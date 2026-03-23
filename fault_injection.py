"""Monte Carlo simulation engine for LEO radiation effects on Flash NAND memory.

Implements stochastic error injection using a LEO radiation
environment model and provides statistical aggregation, convergence
checking, and result reporting compatible with the existing analytical
simulation interfaces.

LEO Radiation Model Parameters
──────────────────────────────
• SEU to Burst SEU ratio: 10 : 1 (single : burst)
  – seu_rate parameter = single-bit SEU rate [events/bit/s]
  – Burst events are an independent Poisson process at 1/10 that rate
• Burst size distribution (SLC, heavy ion):
      2 bits          ~75 %  of burst events
      3–4 bits        ~18 %  of burst events
      5–16 bits       ~5 %   of burst events
      >16 bits        <2 %   (likely SEFI; modelled separately)
• Spatial model: bursts are physically contiguous; start address
  sampled uniformly; truncated at page boundary (partial burst).
"""

from __future__ import annotations

import math
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

# Use dill for multiprocessing serialization to handle closures/lambdas
try:
    import dill.multiprocessing as dill_mp
    HAS_DILL = True
except ImportError:
    HAS_DILL = False


# ══════════════════════════════════════════════════════════
#  LEO RADIATION MODEL PARAMETERS
# ══════════════════════════════════════════════════════════
# These parameters define the LEO burst error distribution.
# Keep synchronized with ecc_simulator.py SIMULATION PARAMETERS.

BURST_SINGLE_TO_BURST_RATIO = 10.0  # 10:1 single-to-burst SEU ratio
BURST_EXCLUDED_PERCENTAGE = 0.02    # >16-bit bursts excluded (<2%)
BURST_BIT_DISTRIBUTION = {
    2: 0.75,                         # 75% of 2-bit bursts
    "3-4": 0.18,                     # 18% uniform over 3-4 bits
    "5-16": 0.05,                    # 5% uniform over 5-16 bits
}

# Derived quantities
_RENORM_FACTOR = 1.0 - BURST_EXCLUDED_PERCENTAGE
_BURST_SEU_RATIO = 1.0 / BURST_SINGLE_TO_BURST_RATIO


# ══════════════════════════════════════════════════════════
#  BURST-SIZE DISTRIBUTION
# ══════════════════════════════════════════════════════════

# Cumulative probability thresholds (SLC, heavy ion), renormalized
# to exclude >16-bit bursts (<2%, likely SEFI).
_BURST_CDF = np.array([
    BURST_BIT_DISTRIBUTION[2] / _RENORM_FACTOR,
    (BURST_BIT_DISTRIBUTION[2] + BURST_BIT_DISTRIBUTION["3-4"]) / _RENORM_FACTOR,
    1.00
])


def sample_burst_size(rng: np.random.Generator) -> int:
    """Sample one burst size from the SLC heavy-ion distribution.

    Returns an integer in [2, 16].
    """
    u = rng.random()
    if u < _BURST_CDF[0]:           # ~76.5 %  →  2 bits
        return 2
    elif u < _BURST_CDF[1]:         # ~18.4 %  →  3–4 bits (uniform)
        return int(rng.integers(3, 5))
    else:                            # ~5.1 %   →  5–16 bits (uniform)
        return int(rng.integers(5, 17))


# ══════════════════════════════════════════════════════════
#  LEO ERROR INJECTION
# ══════════════════════════════════════════════════════════

@dataclass
class InjectionRecord:
    """Diagnostic record for one page corruption event."""
    n_single_seus: int = 0
    n_burst_events: int = 0
    burst_sizes_sampled: list = field(default_factory=list)
    burst_sizes_actual: list = field(default_factory=list)   # after truncation
    n_partial_bursts: int = 0
    total_bits_flipped: int = 0


def inject_errors_leo(flat_page: np.ndarray,
                      seu_rate: float,
                      scrub_interval: float,
                      rng: np.random.Generator,
                      page_size_bits: int | None = None,
                      ) -> Tuple[np.ndarray, InjectionRecord]:
    """Inject errors into a flat page using the LEO radiation model.

    Single-bit SEUs and burst events are modelled as two independent
    Poisson processes.  ``seu_rate`` is the single-bit SEU rate; burst
    events occur at 1/10 of that rate (``_BURST_SEU_RATIO``).
    Burst sizes are sampled from the LEO distribution.

    Args:
        flat_page       : 1-D uint8 array (the full encoded page).
        seu_rate        : single-bit SEU rate  [events / bit / s].
        scrub_interval  : seconds between scrubbing events.
        rng             : numpy random Generator for reproducibility.
        page_size_bits  : override total bit count (default: len * 8).

    Returns:
        (corrupted_page, InjectionRecord)
    """
    out = flat_page.copy()
    n_bytes = len(out)
    total_bits = page_size_bits if page_size_bits is not None else n_bytes * 8
    rec = InjectionRecord()

    if seu_rate < 1e-30 or total_bits == 0:
        return out, rec

    # Single-bit SEUs and burst events are independent Poisson processes.
    # seu_rate is the single-bit SEU rate; burst rate is 1/10 of that.
    lam_single = seu_rate * total_bits * scrub_interval
    lam_burst  = lam_single * _BURST_SEU_RATIO

    n_singles = int(rng.poisson(lam_single))
    n_bursts  = int(rng.poisson(lam_burst))

    if n_singles == 0 and n_bursts == 0:
        return out, rec

    rec.n_single_seus = n_singles
    rec.n_burst_events = n_bursts

    # ── Apply single-bit SEUs ────────────────────────────
    if n_singles > 0:
        # Sample unique bit positions (if n_singles > total_bits, cap it)
        n_flip = min(n_singles, total_bits)
        positions = rng.choice(total_bits, size=n_flip, replace=False)
        for pos in positions:
            byte_idx = pos >> 3           # pos // 8
            bit_idx = pos & 7             # pos % 8
            if byte_idx < n_bytes:
                out[byte_idx] ^= np.uint8(1 << bit_idx)
        rec.total_bits_flipped += n_flip

    # ── Apply burst SEUs ─────────────────────────────────
    # Determine page boundaries (one page = n_bytes bytes)
    # num_sectors is not used here; bursts respect page boundaries only
    for _ in range(n_bursts):
        burst_size = sample_burst_size(rng)
        rec.burst_sizes_sampled.append(burst_size)

        if burst_size <= 0:
            rec.burst_sizes_actual.append(0)
            continue

        # Sample starting bit position uniformly across the page
        start_bit = int(rng.integers(0, total_bits))

        # Truncate burst at page boundary (no wrap-around)
        actual_size = min(burst_size, total_bits - start_bit)
        if actual_size < burst_size:
            rec.n_partial_bursts += 1
        rec.burst_sizes_actual.append(actual_size)

        # Flip contiguous bits
        for i in range(actual_size):
            bit_pos = start_bit + i
            byte_idx = bit_pos >> 3
            bit_idx = bit_pos & 7
            if byte_idx < n_bytes:
                out[byte_idx] ^= np.uint8(1 << bit_idx)
        rec.total_bits_flipped += actual_size

    return out, rec


def inject_errors_leo_conditional(
        flat_page: np.ndarray,
        seu_rate: float,
        scrub_interval: float,
        rng: np.random.Generator,
        page_size_bits: int | None = None,
) -> Tuple[np.ndarray, float]:
    """Importance-sampling variant of inject_errors_leo.

    Forces at least one radiation event (single or burst) by rejection
    sampling from two independent Poisson processes until the total
    count is ≥ 1, then applies the LEO burst model exactly as in
    ``inject_errors_leo``.

    The caller is responsible for weighting the resulting UBER estimate by
    the returned probability ``p_flip = P(≥1 event) = 1 - exp(-λ_total)``
    so that the unconditional UBER is recovered::

        UBER_true = p_flip * mean(UBER | ≥1 event)

    ``λ_total = λ_single + λ_burst = λ_single × (1 + _BURST_SEU_RATIO)``.

    Args:
        flat_page      : 1-D uint8 array (encoded page).
        seu_rate       : single-bit SEU rate [events / bit / s].
        scrub_interval : seconds between scrub events.
        rng            : numpy random Generator.
        page_size_bits : override total bit count (default: len * 8).

    Returns:
        ``(corrupted_page, p_flip)`` — page with ≥1 event applied and the
        importance weight P(at least one event).
    """
    n_bytes = len(flat_page)
    total_bits = page_size_bits if page_size_bits is not None else n_bytes * 8

    # seu_rate is the single-bit SEU rate; burst rate is 1/10 of that.
    lam_single = seu_rate * total_bits * scrub_interval
    lam_burst  = lam_single * _BURST_SEU_RATIO
    lam_total  = lam_single + lam_burst

    # P(at least one radiation event) — the importance weight
    p_flip = 1.0 - math.exp(-lam_total) if lam_total < 700.0 else 1.0

    if lam_total < 1e-300 or total_bits == 0:
        # Essentially zero probability: flip one bit as a degenerate sample.
        # The caller's p_flip weight will suppress this to near zero.
        out = flat_page.copy()
        out[0] ^= np.uint8(1)
        return out, p_flip

    # Zero-truncated: resample until at least one event (single or burst)
    n_singles = 0
    n_bursts  = 0
    while n_singles + n_bursts == 0:
        n_singles = int(rng.poisson(lam_single))
        n_bursts  = int(rng.poisson(lam_burst))

    # Injection logic (same as inject_errors_leo)
    out = flat_page.copy()
    rec = InjectionRecord()

    rec.n_single_seus   = n_singles
    rec.n_burst_events  = n_bursts

    if n_singles > 0:
        n_flip = min(n_singles, total_bits)
        positions = rng.choice(total_bits, size=n_flip, replace=False)
        for pos in positions:
            byte_idx = pos >> 3
            bit_idx  = pos & 7
            if byte_idx < n_bytes:
                out[byte_idx] ^= np.uint8(1 << bit_idx)
        rec.total_bits_flipped += n_flip

    for _ in range(n_bursts):
        burst_size = sample_burst_size(rng)
        if burst_size <= 0:
            continue
        start_bit   = int(rng.integers(0, total_bits))
        actual_size = min(burst_size, total_bits - start_bit)
        for i in range(actual_size):
            bit_pos  = start_bit + i
            byte_idx = bit_pos >> 3
            bit_idx  = bit_pos & 7
            if byte_idx < n_bytes:
                out[byte_idx] ^= np.uint8(1 << bit_idx)
        rec.total_bits_flipped += actual_size

    return out, p_flip


# ══════════════════════════════════════════════════════════
#  ECC ARCHITECTURE ABSTRACTION
# ══════════════════════════════════════════════════════════

class MCArch:
    """ECC architecture wrapper for Monte Carlo simulation.

    Encapsulates encode / decode functions and metadata so the MC engine
    can be architecture-agnostic.
    """

    def __init__(self, label: str, k: int, page_size: int,
                 num_sectors: int,
                 encode_fn: Callable[[np.ndarray], np.ndarray],
                 decode_fn: Callable[[np.ndarray], list],
                 style: dict | None = None):
        """
        Args:
            label       : human-readable name.
            k           : raw data bytes per page.
            page_size   : total encoded page size in bytes.
            num_sectors : number of sectors per page.
            encode_fn   : raw_data (uint8[k]) → flat encoded page (uint8[page_size]).
            decode_fn   : flat encoded page (uint8[page_size]) → list of decoded
                          sector arrays (or None for failed sectors).
            style       : matplotlib style dict for plotting.
        """
        self.label = label
        self.k = k
        self.page_size = page_size
        self.num_sectors = num_sectors
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.style = style or {}
        self.data_bits_per_sector = (k // num_sectors) * 8

    @property
    def rate(self) -> float:
        return self.k / self.page_size


# ══════════════════════════════════════════════════════════
#  POPCOUNT LOOKUP
# ══════════════════════════════════════════════════════════
_POPCOUNT8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.int64)


# ══════════════════════════════════════════════════════════
#  MC SINGLE-WINDOW SIMULATION
# ══════════════════════════════════════════════════════════

@dataclass
class WindowResult:
    """Result of a single MC iteration (one scrub window)."""
    error_bits: int = 0
    total_data_bits: int = 0
    uncorrectable_sectors: int = 0
    injection: InjectionRecord | None = None


def simulate_single_window(arch: MCArch,
                           seu_rate: float,
                           scrub_interval: float,
                           rng: np.random.Generator,
                           use_zero_data: bool = True,
                           ) -> WindowResult:
    """Simulate one scrub window for an ECC architecture.

    Uses the LEO radiation error model to corrupt the encoded page,
    then decodes and counts residual errors.

    Args:
        arch            : ECC architecture.
        seu_rate        : radiation event rate [events/bit/s].
        scrub_interval  : seconds between scrub events.
        rng             : numpy random Generator.
        use_zero_data   : if True, encode all-zero data (valid for linear codes).

    Returns:
        WindowResult with error counts.
    """
    # Encode
    if use_zero_data:
        raw_data = np.zeros(arch.k, dtype=np.uint8)
    else:
        raw_data = rng.integers(0, 256, arch.k, dtype=np.uint8)
    encoded = arch.encode_fn(raw_data)

    # Inject errors
    corrupted, inj_rec = inject_errors_leo(
        encoded, seu_rate, scrub_interval, rng,
        page_size_bits=arch.page_size * 8)

    # Decode
    decoded_sectors = arch.decode_fn(corrupted)

    # Count errors
    wr = WindowResult(injection=inj_rec)
    data_sectors = np.split(raw_data, arch.num_sectors)
    for dec, orig in zip(decoded_sectors, data_sectors):
        wr.total_data_bits += arch.data_bits_per_sector
        if dec is not None:
            wr.error_bits += int(_POPCOUNT8[dec ^ orig].sum())
        else:
            wr.error_bits += arch.data_bits_per_sector
            wr.uncorrectable_sectors += 1
    return wr


# ══════════════════════════════════════════════════════════
#  MC AGGREGATED STATISTICS
# ══════════════════════════════════════════════════════════

@dataclass
class MCSweepPointResult:
    """Aggregated statistics for one SEU-rate point across N iterations."""
    seu_rate: float

    # UBER statistics across iterations
    uber_mean: float = 0.0
    uber_median: float = 0.0
    uber_std: float = 0.0
    uber_p99: float = 0.0

    # Error count statistics
    mean_errors: float = 0.0
    median_errors: float = 0.0
    std_errors: float = 0.0
    p99_errors: float = 0.0

    # ECC overwhelmed fraction
    ecc_overwhelmed_fraction: float = 0.0

    # Burst diagnostics
    burst_sizes_histogram: Dict[int, int] = field(default_factory=dict)
    total_burst_events: int = 0
    total_single_seus: int = 0
    total_partial_bursts: int = 0

    # Raw per-iteration data (for convergence analysis)
    per_iter_uber: np.ndarray = field(default_factory=lambda: np.array([]))
    per_iter_errors: np.ndarray = field(default_factory=lambda: np.array([]))


def _worker_run_iterations(arch_spec: dict, seu_rate: float, scrub_interval: float,
                           worker_id: int, n_iters: int, seed: int,
                           use_zero_data: bool) -> Tuple[np.ndarray, np.ndarray, int, Dict, int, int, int]:
    """Worker process function: run n_iters iterations and return aggregated results.
    
    Args:
        arch_spec: Dictionary with architecture specification (will be used to
                  import and rebuild the MCArch - requires arch type available at worker)
    
    Returns tuple:
        (iter_ubers, iter_errors, n_overwhelmed, burst_hist, total_singles, 
         total_bursts, total_partial)
    """
    rng = np.random.default_rng(seed + worker_id)
    
    # Rebuild arch from specification dict (requires arch params to be available)
    # For now,  we'll just use arch_spec as-is if it's already an MCArch
    # (on fork-based systems this works; on spawn we need to rebuild)
    arch = arch_spec
    if isinstance(arch, dict):
        raise NotImplementedError("Architecture spec dicts not yet supported for workers. "
                                 "Use fork-based multiprocessing (Linux/Mac) or "
                                 "set parallel=False on Windows.")
    
    iter_ubers = np.zeros(n_iters)
    iter_errors = np.zeros(n_iters, dtype=np.int64)
    n_overwhelmed = 0
    burst_hist: Dict[int, int] = {}
    total_singles = 0
    total_bursts = 0
    total_partial = 0
    
    for i in range(n_iters):
        wr = simulate_single_window(arch, seu_rate, scrub_interval,
                                    rng, use_zero_data)
        uber_i = wr.error_bits / wr.total_data_bits if wr.total_data_bits > 0 else 0.0
        iter_ubers[i] = uber_i
        iter_errors[i] = wr.error_bits
        
        if wr.uncorrectable_sectors > 0:
            n_overwhelmed += 1
        
        inj = wr.injection
        if inj is not None:
            total_singles += inj.n_single_seus
            total_bursts += inj.n_burst_events
            total_partial += inj.n_partial_bursts
            for bs in inj.burst_sizes_sampled:
                burst_hist[bs] = burst_hist.get(bs, 0) + 1
    
    return iter_ubers, iter_errors, n_overwhelmed, burst_hist, total_singles, total_bursts, total_partial


def _run_mc_sweep_point_parallel(arch: MCArch,
                                  seu_rate: float,
                                  scrub_interval: float,
                                  n_iterations: int,
                                  seed: int,
                                  use_zero_data: bool,
                                  num_workers: Optional[int]) -> MCSweepPointResult:
    """Parallel implementation: distribute N iterations across worker processes.
    
    Recommended for large CPU counts (8+ cores) and N >= 100 iterations.
    Each worker runs independent iterations with a unique RNG seed.
    
    Uses fork-based multiprocessing on Linux/Mac (no serialization overhead).
    On Windows: uses spawn + dill serialization for closure support.
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    
    result = MCSweepPointResult(seu_rate=seu_rate)
    all_iter_ubers = []
    all_iter_errors = []
    
    # Select context: fork on Unix, spawn on Windows (with dill)
    is_windows = sys.platform == 'win32'
    
    if not is_windows and HAS_DILL:
        # Linux/Mac: prefer fork for zero serialization overhead
        try:
            ctx = __import__('multiprocessing').get_context('fork')
        except ValueError:
            ctx = None
    else:
        ctx = None
    
    # If fork not available, use spawn with dill on Windows
    if ctx is None:
        if not HAS_DILL:
            import warnings
            warnings.warn("dill library not found. Parallel MC requires: pip install dill",
                         RuntimeWarning)
            return _run_mc_sweep_point_sequential(arch, seu_rate, scrub_interval,
                                                  n_iterations, seed, use_zero_data)
        # Use dill-enabled multiprocessing Pool (spawn-based)
        pool_ctx = dill_mp.Pool(num_workers)
    else:
        # Use fork-based Pool (Linux/Mac)
        pool_ctx = ctx.Pool(num_workers)
    
    # Use context manager for pool
    with pool_ctx as pool:
        # Partition iterations among workers
        iters_per_worker = n_iterations // num_workers
        remainder = n_iterations % num_workers
        
        tasks = []
        for w in range(num_workers):
            n_iters = iters_per_worker + (1 if w < remainder else 0)
            tasks.append((arch, seu_rate, scrub_interval, w, n_iters, seed, 
                         use_zero_data))
        
        # Collect results from all workers
        for result_tuple in pool.starmap(_worker_run_iterations, tasks):
            (iter_ubers, iter_errors, n_overwhelmed, burst_hist,
             total_singles, total_bursts, total_partial) = result_tuple
            
            all_iter_ubers.extend(iter_ubers)
            all_iter_errors.extend(iter_errors)
            
            # Accumulate aggregates
            result.ecc_overwhelmed_fraction += n_overwhelmed
            result.total_single_seus += total_singles
            result.total_burst_events += total_bursts
            result.total_partial_bursts += total_partial
            
            # Merge burst histograms
            for size, count in burst_hist.items():
                result.burst_sizes_histogram[size] = (
                    result.burst_sizes_histogram.get(size, 0) + count)
    
    # Convert to arrays and finalize
    iter_ubers_all = np.array(all_iter_ubers)
    iter_errors_all = np.array(all_iter_errors, dtype=np.int64)
    
    # Aggregate UBER statistics
    result.uber_mean = float(np.mean(iter_ubers_all))
    result.uber_median = float(np.median(iter_ubers_all))
    result.uber_std = float(np.std(iter_ubers_all))
    result.uber_p99 = float(np.percentile(iter_ubers_all, 99))
    
    # Error count statistics
    result.mean_errors = float(np.mean(iter_errors_all))
    result.median_errors = float(np.median(iter_errors_all))
    result.std_errors = float(np.std(iter_errors_all))
    result.p99_errors = float(np.percentile(iter_errors_all, 99))
    
    # Normalize overwhelmed fraction
    result.ecc_overwhelmed_fraction /= n_iterations
    
    # Raw data for convergence
    result.per_iter_uber = iter_ubers_all
    result.per_iter_errors = iter_errors_all
    
    return result


def _run_mc_sweep_point_sequential(arch: MCArch,
                                    seu_rate: float,
                                    scrub_interval: float,
                                    n_iterations: int,
                                    seed: int,
                                    use_zero_data: bool) -> MCSweepPointResult:
    """Sequential (non-parallel) MC implementation."""
    rng = np.random.default_rng(seed)
    result = MCSweepPointResult(seu_rate=seu_rate)

    iter_ubers = np.zeros(n_iterations)
    iter_errors = np.zeros(n_iterations, dtype=np.int64)
    n_overwhelmed = 0
    burst_hist: Dict[int, int] = {}

    for i in range(n_iterations):
        wr = simulate_single_window(arch, seu_rate, scrub_interval,
                                    rng, use_zero_data)
        uber_i = wr.error_bits / wr.total_data_bits if wr.total_data_bits > 0 else 0.0
        iter_ubers[i] = uber_i
        iter_errors[i] = wr.error_bits

        if wr.uncorrectable_sectors > 0:
            n_overwhelmed += 1

        # Accumulate burst diagnostics
        inj = wr.injection
        if inj is not None:
            result.total_single_seus += inj.n_single_seus
            result.total_burst_events += inj.n_burst_events
            result.total_partial_bursts += inj.n_partial_bursts
            for bs in inj.burst_sizes_sampled:
                burst_hist[bs] = burst_hist.get(bs, 0) + 1

    # Aggregate UBER statistics
    result.uber_mean = float(np.mean(iter_ubers))
    result.uber_median = float(np.median(iter_ubers))
    result.uber_std = float(np.std(iter_ubers))
    result.uber_p99 = float(np.percentile(iter_ubers, 99))

    # Error count statistics
    result.mean_errors = float(np.mean(iter_errors))
    result.median_errors = float(np.median(iter_errors))
    result.std_errors = float(np.std(iter_errors))
    result.p99_errors = float(np.percentile(iter_errors, 99))

    # ECC overwhelmed fraction
    result.ecc_overwhelmed_fraction = n_overwhelmed / n_iterations

    # Burst histogram
    result.burst_sizes_histogram = burst_hist

    # Raw data for convergence
    result.per_iter_uber = iter_ubers
    result.per_iter_errors = iter_errors

    return result


def run_mc_sweep_point(arch: MCArch,
                       seu_rate: float,
                       scrub_interval: float,
                       n_iterations: int = 1000,
                       seed: int = 42,
                       use_zero_data: bool = True,
                       parallel: bool = False,
                       num_workers: Optional[int] = None,
                       ) -> MCSweepPointResult:
    """Run N Monte Carlo iterations for a single SEU rate.

    For large CPU counts, set parallel=True to distribute iterations across
    worker processes (recommended for N >= 100 iterations or many configurations).

    Args:
        arch            : ECC architecture.
        seu_rate        : radiation event rate [events/bit/s].
        scrub_interval  : seconds between scrub events.
        n_iterations    : number of MC iterations.
        seed            : RNG seed for reproducibility.
        use_zero_data   : encode all-zero data (valid for linear codes).
        parallel        : if True, distribute iterations across worker processes.
        num_workers     : number of worker processes (None = all CPU cores).

    Returns:
        MCSweepPointResult with aggregated statistics.
    """
    if parallel:
        return _run_mc_sweep_point_parallel(arch, seu_rate, scrub_interval,
                                            n_iterations, seed, use_zero_data,
                                            num_workers)
    else:
        return _run_mc_sweep_point_sequential(arch, seu_rate, scrub_interval,
                                              n_iterations, seed, use_zero_data)


# ══════════════════════════════════════════════════════════
#  FULL SEU-RATE SWEEP
# ══════════════════════════════════════════════════════════

@dataclass
class MCSweepResult:
    """Full sweep result across all SEU rates, compatible with analytical output."""
    label: str
    seu_rates: np.ndarray
    uber_values: np.ndarray            # mean UBER per rate (same shape as analytical)
    point_results: List[MCSweepPointResult]

    # Convenience accessors
    @property
    def uber_median(self) -> np.ndarray:
        return np.array([p.uber_median for p in self.point_results])

    @property
    def uber_std(self) -> np.ndarray:
        return np.array([p.uber_std for p in self.point_results])

    @property
    def uber_p99(self) -> np.ndarray:
        return np.array([p.uber_p99 for p in self.point_results])

    @property
    def ecc_overwhelmed(self) -> np.ndarray:
        return np.array([p.ecc_overwhelmed_fraction for p in self.point_results])


def run_mc_sweep(arch: MCArch,
                 seu_rate_sweep: np.ndarray,
                 scrub_interval: float,
                 n_iterations: int = 1000,
                 seed: int = 42,
                 use_zero_data: bool = True,
                 verbose: bool = True,
                 parallel: bool = False,
                 num_workers: Optional[int] = None,
                 ) -> MCSweepResult:
    """Run full Monte Carlo SEU-rate sweep for one ECC architecture.

    Returns an MCSweepResult whose ``uber_values`` array is a drop-in
    replacement for the analytical UBER arrays used by the existing code.

    Args:
        arch            : ECC architecture.
        seu_rate_sweep  : 1-D array of SEU rates to simulate.
        scrub_interval  : seconds between scrub events.
        n_iterations    : MC iterations per SEU-rate point.
        seed            : base RNG seed (varied per rate point).
        use_zero_data   : encode all-zero data (valid for linear codes).
        verbose         : print progress.
        parallel        : if True, distribute iterations across worker processes
                         (recommended for large CPU counts).
        num_workers     : number of worker processes (None = all CPU cores).

    Returns:
        MCSweepResult
    """
    n_rates = len(seu_rate_sweep)
    uber_arr = np.zeros(n_rates)
    points = []

    for ri, seu_rate in enumerate(seu_rate_sweep):
        # Vary seed per rate point for independence, but still reproducible
        point_seed = seed + ri * 1000003
        pt = run_mc_sweep_point(arch, seu_rate, scrub_interval,
                                n_iterations, point_seed, use_zero_data,
                                parallel, num_workers)
        uber_arr[ri] = pt.uber_mean
        points.append(pt)

        if verbose:
            owh = pt.ecc_overwhelmed_fraction
            print(f"  SEU={seu_rate:.2e}  UBER={pt.uber_mean:.3e} "
                  f"(med={pt.uber_median:.3e}, std={pt.uber_std:.3e}, "
                  f"p99={pt.uber_p99:.3e}, overwhelmed={owh:.3f})")

    return MCSweepResult(
        label=arch.label,
        seu_rates=seu_rate_sweep,
        uber_values=uber_arr,
        point_results=points,
    )


# ══════════════════════════════════════════════════════════
#  CONVERGENCE CHECK
# ══════════════════════════════════════════════════════════

def convergence_data(per_iter_uber: np.ndarray,
                     checkpoints: np.ndarray | None = None,
                     ) -> Dict[str, np.ndarray]:
    """Compute running statistics at increasing iteration counts.

    Args:
        per_iter_uber : 1-D array of per-iteration UBER values.
        checkpoints   : iteration counts at which to evaluate
                        (default: logarithmically spaced up to N).

    Returns:
        Dict with keys 'n', 'mean', 'std', 'p99' – arrays of length
        len(checkpoints).
    """
    N = len(per_iter_uber)
    if checkpoints is None:
        checkpoints = np.unique(np.geomspace(10, N, num=min(50, N)).astype(int))
        checkpoints = checkpoints[checkpoints <= N]

    means = np.zeros(len(checkpoints))
    stds = np.zeros(len(checkpoints))
    p99s = np.zeros(len(checkpoints))

    for i, cp in enumerate(checkpoints):
        subset = per_iter_uber[:cp]
        means[i] = np.mean(subset)
        stds[i] = np.std(subset)
        p99s[i] = np.percentile(subset, 99)

    return {'n': checkpoints, 'mean': means, 'std': stds, 'p99': p99s}


def plot_convergence(sweep_result: MCSweepResult,
                     rate_indices: list | None = None,
                     save_path: str | None = None):
    """Plot convergence of UBER mean as N increases.

    For selected SEU-rate points in the sweep, shows how the running-mean
    UBER stabilises with increasing MC iteration count.

    Args:
        sweep_result  : full sweep result.
        rate_indices  : indices into the SEU-rate array to plot
                        (default: 3 evenly spaced).
        save_path     : if given, save figure to this path.
    """
    import matplotlib.pyplot as plt

    pts = sweep_result.point_results
    if rate_indices is None:
        n = len(pts)
        rate_indices = [0, n // 3, 2 * n // 3, n - 1]
        rate_indices = sorted(set(min(i, n - 1) for i in rate_indices))

    fig, axes = plt.subplots(len(rate_indices), 1,
                             figsize=(9, 3.0 * len(rate_indices)),
                             sharex=False)
    if len(rate_indices) == 1:
        axes = [axes]

    for ax, ri in zip(axes, rate_indices):
        pt = pts[ri]
        cd = convergence_data(pt.per_iter_uber)
        ax.plot(cd['n'], cd['mean'], label='running mean', linewidth=1.5)
        ax.fill_between(cd['n'],
                        cd['mean'] - cd['std'],
                        cd['mean'] + cd['std'],
                        alpha=0.2, label='± 1 σ')
        ax.axhline(pt.uber_mean, color='red', linestyle='--', linewidth=0.8,
                    label=f'final mean = {pt.uber_mean:.3e}')
        ax.set_ylabel('UBER')
        ax.set_xlabel('N iterations')
        ax.set_title(f'SEU rate = {pt.seu_rate:.2e}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if pt.uber_mean > 0:
            ax.set_yscale('log')

    fig.suptitle(f'MC Convergence — {sweep_result.label}', fontsize=11)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.35)
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Convergence plot saved to {save_path}")
    return fig


# ══════════════════════════════════════════════════════════
#  BURST-SIZE HISTOGRAM
# ══════════════════════════════════════════════════════════

def aggregate_burst_histogram(sweep_result: MCSweepResult) -> Dict[int, int]:
    """Aggregate burst-size histogram across all sweep points."""
    combined: Dict[int, int] = {}
    for pt in sweep_result.point_results:
        for bs, cnt in pt.burst_sizes_histogram.items():
            combined[bs] = combined.get(bs, 0) + cnt
    return combined


def plot_burst_histogram(sweep_result: MCSweepResult,
                         save_path: str | None = None):
    """Plot histogram of burst sizes actually sampled during the sweep."""
    import matplotlib.pyplot as plt

    hist = aggregate_burst_histogram(sweep_result)
    if not hist:
        print("  No burst events recorded – nothing to plot.")
        return None

    sizes = sorted(hist.keys())
    counts = [hist[s] for s in sizes]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(sizes, counts, width=0.8, edgecolor='black', linewidth=0.5,
           color='#4477AA', alpha=0.8)
    ax.set_xlabel('Burst size  [bits]')
    ax.set_ylabel('Count')
    ax.set_title(f'Burst-Size Histogram — {sweep_result.label}')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Burst histogram saved to {save_path}")
    return fig


# ══════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ══════════════════════════════════════════════════════════

def print_sweep_summary(sweep_result: MCSweepResult):
    """Print a tabular summary matching the existing analytical output style."""
    print(f"\n{'='*80}")
    print(f"  Monte Carlo Sweep Summary -- {sweep_result.label}")
    print(f"{'='*80}")
    print(f"  {'SEU rate':>12s}  {'UBER mean':>11s}  {'UBER med':>11s}  "
          f"{'UBER std':>11s}  {'UBER p99':>11s}  {'ECC fail':>8s}")
    print(f"  {'-'*12}  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*8}")
    for pt in sweep_result.point_results:
        print(f"  {pt.seu_rate:12.2e}  {pt.uber_mean:11.3e}  {pt.uber_median:11.3e}  "
              f"{pt.uber_std:11.3e}  {pt.uber_p99:11.3e}  {pt.ecc_overwhelmed_fraction:8.4f}")

    # Aggregate burst stats
    total_singles = sum(p.total_single_seus for p in sweep_result.point_results)
    total_bursts = sum(p.total_burst_events for p in sweep_result.point_results)
    total_partial = sum(p.total_partial_bursts for p in sweep_result.point_results)
    print(f"\n  Total single SEU events : {total_singles:,}")
    print(f"  Total burst SEU events  : {total_bursts:,}")
    print(f"  Partial bursts (trunc.) : {total_partial:,}")
    print(f"{'='*80}\n")
