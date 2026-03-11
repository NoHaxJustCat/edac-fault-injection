"""Monte-Carlo ECC comparison for multiple NAND page sizes.

Companion script to compare_ecc_pagesizes.py.  Uses the same LEO
radiation environment model and analytical baselines, but drives the
Monte Carlo simulation through a parallel ProcessPoolExecutor that is
safe on Windows (spawn-based multiprocessing).

Two modes are available:

  random  (default)
      Every radiation event is an independent single-bit SEU.
      No burst clustering.  Results closely follow the Poisson-based
      analytical models (thin dotted overlays on the UBER plot).

  burst
      Uses the validated LEO radiation environment model: 10:1
      single-to-burst SEU ratio with a realistic burst-size
      distribution (see monte_carlo.py for details).  This shows
      the real-world degradation caused by burst errors.

Both modes sweep SEU rate (events/bit/s) on the x-axis and plot
UBER on the y-axis, with a configurable scrub interval.

Page sizes analysed: 4224 B, 8640 B

Usage
-----
    python compare_ecc_burst_pagesizes.py               # random mode (default)
    python compare_ecc_burst_pagesizes.py --mode burst  # LEO burst model
    python compare_ecc_burst_pagesizes.py --mode random # explicit random-only

BCH on large sectors (e.g. 8640 B page → 1080 B sector = 8640 bits) exceeds
the GF(2^13) single-codeword limit (8191 bits), so each sector is automatically
split into sub-sector chunks that each fit within the BCH codeword length.

To adjust the simulation, edit the constants in the SIMULATION PARAMETERS
section at the top of the file.
"""

import argparse
import os
import sys
import multiprocessing
from math import ceil, exp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.lines import Line2D

from utils import (
    compute_data_bytes, _make_bch,
    encode_with_rs, decode_with_rs,
    encode_with_bch, decode_with_bch,
    encode_with_bch_chunked, decode_with_bch_chunked,
)
from monte_carlo import inject_errors_leo, inject_errors_leo_conditional, _POPCOUNT8

# ===========================================================
#  OUTPUT DIRECTORY
# ===========================================================
_BASE_OUTPUT_DIR = os.path.join("images", "burst_pagesizes")


def _get_output_dir(mode):
    """Return (and create) the per-mode output subdirectory."""
    path = os.path.join(_BASE_OUTPUT_DIR, mode)
    os.makedirs(path, exist_ok=True)
    return path

# ===========================================================
#  SIMULATION PARAMETERS
# ===========================================================
PAGE_SIZES     = [4224, 8640]   # NAND page sizes to evaluate [bytes]
NUM_SECTORS    = 8              # equal-sized sectors per page

# SEU rate sweep -- same x-axis as compare_ecc_pagesizes
SEU_RATE_SWEEP = np.logspace(-8, -5, 40)

# Scrubbing
SCRUB_INTERVAL = 3600           # seconds between scrub events

# UBER target line drawn on plots
UBER_REQ       = 1e-12

# Monte Carlo settings
#
# IMPORTANT — resolution floor:
#   The finest non-zero UBER this estimator can return for a sector-based
#   code (NUM_SECTORS sectors per page) is:
#
#       UBER_min = 1 / (NUM_ITERS × NUM_SECTORS)
#
#   With NUM_ITERS=10, NUM_SECTORS=8 → UBER_min ≈ 1.25×10⁻², which
#   causes all curves to go flat once they hit that floor.
#   Required iterations for a given target floor:
#       NUM_ITERS ≥ 1 / (UBER_floor × NUM_SECTORS)
#   e.g. 10⁻⁴ floor → ≥1 250 iters,  10⁻⁵ floor → ≥12 500 iters.
#
NUM_ITERS      = 10          # iterations per (architecture × SEU-rate point)
                               # → floor ≈ 1.25×10⁻⁴  (smooth down to ~10⁻⁴)
# Cap at 61: Python 3.13 on Windows enforces a strict 61-worker handle limit.
_CPU_COUNT  = os.cpu_count() or 1
NUM_WORKERS = min(_CPU_COUNT, 61)

# ===========================================================
#  PLOT STYLE
# ===========================================================
# Paul Tol "Bright" palette -- colorblind-safe, publication standard
TOL_COLORS = {
    "blue":   "#4477AA",
    "cyan":   "#66CCEE",
    "green":  "#228833",
    "yellow": "#CCBB44",
    "red":    "#EE6677",
    "purple": "#AA3377",
    "orange": "#EE7733",
}

mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       15,
    "axes.titlesize":  17,
    "axes.labelsize":  15,
    "legend.fontsize": 13,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth":  1.2,
    "grid.linewidth":  0.8,
    "lines.linewidth": 2.2,
    "figure.dpi":      200,
})


# ===========================================================
#  ANALYTICAL UBER MODELS
# ===========================================================

def _poisson_sf(t, lam):
    """P(Poisson(lam) > t)."""
    if lam < 1e-30:
        return 0.0
    from scipy.stats import poisson as poisson_rv  # lazy import (avoids pagefile issue in workers)
    return float(poisson_rv.sf(t, lam))


def uber_rs_only(seu_rate, rs_t, nsym, sector_bytes,
                 scrub=SCRUB_INTERVAL):
    """Analytical UBER for RS-only: Poisson symbol-error model."""
    lam_bit = seu_rate * scrub
    if lam_bit < 1e-30:
        return 0.0
    p_byte = 1.0 - np.exp(-8.0 * lam_bit)
    num_chunks = max(1, ceil(sector_bytes / 255))
    chunk_bytes = sector_bytes // num_chunks
    lam_chunk = chunk_bytes * p_byte
    p_chunk_fail = _poisson_sf(rs_t, lam_chunk)
    return 1.0 - (1.0 - p_chunk_fail) ** num_chunks


def uber_bch_only(seu_rate, bch_t, sector_bits,
                  scrub=SCRUB_INTERVAL):
    """Analytical UBER for BCH-only: Poisson bit-error model."""
    lam = seu_rate * sector_bits * scrub
    return _poisson_sf(bch_t, lam)


def _gallager_a_threshold(dv, dc, max_de_iter=2000, bisect_steps=80):
    """Density evolution threshold for Gallager-A LDPC decoding."""
    def _converges(p):
        q = p
        for _ in range(max_de_iter):
            r = 0.5 * (1.0 - (1.0 - 2.0 * q) ** (dc - 1))
            q_new = (p * (1.0 - (1.0 - r) ** (dv - 1))
                     + (1.0 - p) * r ** (dv - 1))
            if q_new < 1e-15:
                return True
            if q_new > q + 1e-15:
                return False
            q = q_new
        return False

    lo, hi = 0.0, 0.5
    for _ in range(bisect_steps):
        mid = (lo + hi) / 2.0
        if _converges(mid):
            lo = mid
        else:
            hi = mid
    return lo


def uber_ldpc(seu_rate, t_eff, page_bits, scrub=SCRUB_INTERVAL):
    """Analytical UBER for LDPC: Poisson threshold model."""
    lam = seu_rate * page_bits * scrub
    return _poisson_sf(t_eff, lam)


# ===========================================================
#  ECC ARCHITECTURE CONTAINER
# ===========================================================

class Arch:
    """Container for one ECC architecture.

    Stores both human-readable metadata and the picklable ``arch_type`` /
    ``arch_params`` fields that workers use to reconstruct the codec without
    passing closures across process boundaries (required on Windows / spawn).
    ``analytical_fn`` is an optional callable(seu_rate) -> uber used only
    in the main process for the analytical overlay.
    """

    def __init__(self, label, k, page_size, arch_type, arch_params,
                 encode_fn, decode_fn, style=None, analytical_fn=None):
        self.label       = label
        self.k           = k
        self.page_size   = page_size
        self.rate        = k / page_size
        self.arch_type   = arch_type   # 'rs', 'bch', or 'ldpc'
        self.arch_params = arch_params # picklable dict for workers
        self.encode_fn   = encode_fn
        self.decode_fn   = decode_fn
        self.style       = style or {}
        self.analytical_fn = analytical_fn
        self.data_bits_per_sector = (k // NUM_SECTORS) * 8


# ===========================================================
#  ARCHITECTURE BUILDERS
# ===========================================================

_RS_COLORS  = [TOL_COLORS["blue"], TOL_COLORS["cyan"]]
_BCH_COLORS = [TOL_COLORS["green"], TOL_COLORS["yellow"], "#BBCC33"]


def _make_rs_arch(nsym, page_size, color):
    """Build an RS-only architecture with *nsym* parity symbols."""
    k = compute_data_bytes(page_size, NUM_SECTORS, rs_nsyms=[nsym])
    sector_len = page_size // NUM_SECTORS
    sector_bytes = sector_len
    rs_t = nsym // 2

    def encode(data):
        return np.concatenate(encode_with_rs(data, nsym, NUM_SECTORS))

    def decode(flat):
        sectors = [flat[i * sector_len:(i + 1) * sector_len]
                   for i in range(NUM_SECTORS)]
        decoded, _ = decode_with_rs(sectors, nsym)
        return decoded

    def analytical(r):
        return uber_rs_only(r, rs_t, nsym, sector_bytes)

    return Arch(
        label=f"RS nsym={nsym} (t={rs_t})",
        k=k, page_size=page_size,
        arch_type="rs",
        arch_params=dict(nsym=nsym, page_size=page_size,
                         num_sectors=NUM_SECTORS),
        encode_fn=encode, decode_fn=decode,
        style=dict(linestyle="--", color=color, linewidth=1.8),
        analytical_fn=analytical,
    )


def _make_bch_arch(ecc_bytes, page_size, color):
    """Build a BCH-only architecture, chunking the sector when it exceeds the
    GF(2^13) BCH codeword limit (8191 bits / 1023 bytes).  Returns None only
    if the chunk overhead still makes the configuration infeasible.
    """
    sector_bytes       = page_size // NUM_SECTORS
    bch_obj            = _make_bch(ecc_bytes)
    max_codeword_bytes = 8191 // 8           # = 1023 bytes
    max_data_bytes     = (8191 - bch_obj.ecc_bits) // 8

    # Find the smallest n_chunks such that each chunk fits in the BCH field.
    # n_chunks must also divide sector_bytes evenly for equal-sized chunks.
    n_chunks = ceil(sector_bytes / max_codeword_bytes)
    while sector_bytes % n_chunks != 0:
        n_chunks += 1

    chunk_encoded_bytes = sector_bytes // n_chunks
    chunk_data_bytes    = chunk_encoded_bytes - bch_obj.ecc_bytes

    if chunk_data_bytes <= 0 or chunk_data_bytes > max_data_bytes:
        return None  # ECC overhead exceeds chunk capacity

    data_len   = n_chunks * chunk_data_bytes
    k          = data_len * NUM_SECTORS
    chunk_bits = chunk_encoded_bytes * 8
    bch_t      = bch_obj.t

    _nc = n_chunks  # captured by closures
    if n_chunks == 1:
        def encode(data):
            return encode_with_bch(data, ecc_bytes, NUM_SECTORS)

        def decode(flat):
            sl = page_size // NUM_SECTORS
            sectors_list = [flat[i * sl:(i + 1) * sl] for i in range(NUM_SECTORS)]
            decoded, _ = decode_with_bch(sectors_list, ecc_bytes)
            return decoded
    else:
        def encode(data):
            return encode_with_bch_chunked(data, ecc_bytes, NUM_SECTORS, _nc)

        def decode(flat):
            sl = page_size // NUM_SECTORS
            sectors_list = [flat[i * sl:(i + 1) * sl] for i in range(NUM_SECTORS)]
            decoded, _ = decode_with_bch_chunked(sectors_list, ecc_bytes, _nc)
            return decoded

    def analytical(r):
        # Sector fails if any of its n_chunks codewords fails (independent)
        p_chunk_fail = uber_bch_only(r, bch_t, chunk_bits)
        return 1.0 - (1.0 - p_chunk_fail) ** _nc

    return Arch(
        label=f"BCH {ecc_bytes}B (t={bch_t})",
        k=k, page_size=page_size,
        arch_type="bch",
        arch_params=dict(ecc_bytes=ecc_bytes, page_size=page_size,
                         num_sectors=NUM_SECTORS, n_chunks=n_chunks),
        encode_fn=encode, decode_fn=decode,
        style=dict(linestyle=":", color=color, linewidth=2),
        analytical_fn=analytical,
    )


def _make_ldpc_arch(page_size):
    """Build LDPC threshold architecture (Gallager-A, dv=3, dc=30, R~0.90)."""
    dv, dc = 3, 30
    pstar = _gallager_a_threshold(dv, dc)
    page_bits = page_size * 8
    t_eff = int(pstar * page_bits)

    k = int(page_size * (1 - dv / dc))
    k -= k % NUM_SECTORS   # must be divisible by num_sectors

    def encode(data):
        out = np.zeros(page_size, dtype=np.uint8)
        out[:len(data)] = data
        return out

    def decode(flat):
        # LDPC codeword spans the whole page -> check total page errors
        n_page_errors = int(_POPCOUNT8[flat].sum())
        sector_data_bytes = k // NUM_SECTORS
        if n_page_errors <= t_eff:
            return [np.zeros(sector_data_bytes, dtype=np.uint8)
                    for _ in range(NUM_SECTORS)]
        else:
            return [None] * NUM_SECTORS

    def analytical(r):
        return uber_ldpc(r, t_eff, page_bits)

    return Arch(
        label=f"LDPC dv={dv},dc={dc} (t_eff~{t_eff})",
        k=k, page_size=page_size,
        arch_type="ldpc",
        arch_params=dict(t_eff=t_eff, k=k, page_size=page_size,
                         num_sectors=NUM_SECTORS),
        encode_fn=encode, decode_fn=decode,
        style=dict(linestyle="-.", color=TOL_COLORS["orange"], linewidth=2.0),
        analytical_fn=analytical,
    )


def build_archs(page_size):
    """Return all Arch instances for the given page size.

    Builds two RS-only (nsym in {8, 16}), up to three BCH-only
    (ecc_bytes in {13, 22, 31}), and one LDPC threshold architecture.
    BCH configs that exceed the GF(2^13) block size are silently skipped.
    """
    archs = []

    for nsym, color in zip([8, 16], _RS_COLORS):
        try:
            archs.append(_make_rs_arch(nsym, page_size, color))
        except ValueError:
            pass

    for ecc_bytes, color in zip([13, 22, 31], _BCH_COLORS):
        try:
            arch = _make_bch_arch(ecc_bytes, page_size, color)
            if arch is not None:
                archs.append(arch)
        except ValueError:
            pass

    archs.append(_make_ldpc_arch(page_size))
    return archs


# ===========================================================
#  ERROR INJECTION  (module-level -> picklable on Windows)
# ===========================================================

def _inject_random(flat_page, seu_rate, scrub_interval, rng):
    """Inject independent single-bit SEUs (no burst clustering).

    Same total event rate as the LEO model, but ALL radiation events are
    independent single-bit flips.  This matches the Poisson analytical
    model and serves as an optimistic baseline.
    """
    out = flat_page.copy()
    n_bytes = len(out)
    total_bits = n_bytes * 8

    if seu_rate < 1e-30 or total_bits == 0:
        return out

    lam_events = seu_rate * total_bits * scrub_interval
    n_events = int(rng.poisson(lam_events))
    if n_events == 0:
        return out

    n_flip = min(n_events, total_bits)
    positions = rng.choice(total_bits, size=n_flip, replace=False)
    for pos in positions:
        out[pos >> 3] ^= np.uint8(1 << (pos & 7))
    return out


def _inject_burst(flat_page, seu_rate, scrub_interval, rng):
    """LEO radiation model: 10:1 single-to-burst SEU ratio.

    Delegates to monte_carlo.inject_errors_leo which implements the
    validated LEO burst-size distribution.
    """
    corrupted, _ = inject_errors_leo(flat_page, seu_rate, scrub_interval, rng)
    return corrupted


def _inject_random_conditional(flat_page, seu_rate, scrub_interval, rng):
    """Conditional (importance-sampling) variant of _inject_random.

    Forces at least one radiation event via zero-truncated Poisson sampling.
    The caller must scale the resulting UBER by
    ``p_flip = 1 - exp(-lambda)`` to recover the unconditional UBER.
    """
    out = flat_page.copy()
    n_bytes = len(out)
    total_bits = n_bytes * 8

    if seu_rate < 1e-30 or total_bits == 0:
        out[0] ^= np.uint8(1)
        return out

    lam_events = seu_rate * total_bits * scrub_interval
    # Zero-truncated Poisson: resample until n_events >= 1
    n_events = 0
    while n_events == 0:
        n_events = int(rng.poisson(lam_events))

    n_flip = min(n_events, total_bits)
    positions = rng.choice(total_bits, size=n_flip, replace=False)
    for pos in positions:
        out[pos >> 3] ^= np.uint8(1 << (pos & 7))
    return out


def _inject_burst_conditional(flat_page, seu_rate, scrub_interval, rng):
    """Conditional (importance-sampling) variant of _inject_burst.

    Forces at least one radiation event via zero-truncated Poisson sampling
    and applies the full LEO burst model.  The caller must scale the resulting
    UBER by ``p_flip = 1 - exp(-lambda)`` to recover the unconditional UBER.
    """
    corrupted, _ = inject_errors_leo_conditional(
        flat_page, seu_rate, scrub_interval, rng)
    return corrupted


# ===========================================================
#  WORKER  (top-level -> picklable on Windows / spawn)
# ===========================================================

def _worker(task):
    """Simulate one (architecture, SEU-rate) point in a subprocess.

    Parameters
    ----------
    task : tuple
        ``(arch_index, arch_type, arch_params, seu_rate, seed,
           num_iters, mode, scrub_interval)``

    Returns
    -------
    tuple
        ``(arch_index, seu_rate, weighted_error_bits, total_data_bits)``

        ``weighted_error_bits`` is a float equal to
        ``p_flip * conditional_error_bits`` where
        ``p_flip = 1 - exp(-\u03bb)`` is the probability that the page sees at
        least one radiation event in the scrub window.  Dividing by
        ``total_data_bits`` gives the true (unconditional) UBER.
    """
    try:
        (arch_index, arch_type, arch_params,
         seu_rate, seed, num_iters, mode, scrub_interval) = task

        popcount8 = _POPCOUNT8               # module-level LUT from monte_carlo
        rng       = np.random.default_rng(seed)
        ns        = arch_params["num_sectors"]
        page_size = arch_params["page_size"]
        sector_len = page_size // ns
        error_bits = 0
        total_bits = 0

        # ── Importance-sampling weight ─────────────────────────────────────
        # P(at least one radiation event on the full encoded page during one
        # scrub window).  Every simulated iteration is conditioned on having
        # ≥1 event (zero-truncated Poisson), so the resulting UBER estimate
        # must be multiplied by p_flip to recover the unconditional UBER:
        #     UBER_true = p_flip × UBER_conditional
        # This eliminates zero-event iterations and greatly improves resolution
        # at low SEU rates where λ << 1 and most Poisson draws would be zero.
        lam_page = seu_rate * (page_size * 8) * scrub_interval
        p_flip = 1.0 - exp(-lam_page) if lam_page < 700.0 else 1.0

        inject = (_inject_burst_conditional if mode == "burst"
                  else _inject_random_conditional)

        # -- RS-only simulation path -------------------------
        if arch_type == "rs":
            nsym     = arch_params["nsym"]
            k        = compute_data_bytes(page_size, ns, rs_nsyms=[nsym])
            raw_data = np.zeros(k, dtype=np.uint8)
            encoded  = np.concatenate(encode_with_rs(raw_data, nsym, ns))
            dbps     = (k // ns) * 8

            for _ in range(num_iters):
                corrupted = inject(encoded, seu_rate, scrub_interval, rng)
                sectors = [corrupted[i * sector_len:(i + 1) * sector_len]
                           for i in range(ns)]
                decoded_sectors, _ = decode_with_rs(sectors, nsym)
                for dec in decoded_sectors:
                    total_bits += dbps
                    if dec is not None:
                        error_bits += int(popcount8[dec].sum())
                    else:
                        error_bits += dbps

        # -- BCH-only simulation path ------------------------
        elif arch_type == "bch":
            ecc_bytes = arch_params["ecc_bytes"]
            n_chunks  = arch_params.get("n_chunks", 1)
            bch_obj   = _make_bch(ecc_bytes)
            chunk_encoded_bytes = sector_len // n_chunks
            chunk_data_bytes    = chunk_encoded_bytes - bch_obj.ecc_bytes
            data_len            = n_chunks * chunk_data_bytes
            k                   = data_len * ns
            raw_data            = np.zeros(k, dtype=np.uint8)
            if n_chunks == 1:
                encoded = encode_with_bch(raw_data, ecc_bytes, ns)
            else:
                encoded = encode_with_bch_chunked(raw_data, ecc_bytes, ns, n_chunks)
            dbps = data_len * 8

            for _ in range(num_iters):
                corrupted = inject(encoded, seu_rate, scrub_interval, rng)
                sectors = [corrupted[i * sector_len:(i + 1) * sector_len]
                           for i in range(ns)]
                if n_chunks == 1:
                    decoded_sectors, _ = decode_with_bch(sectors, ecc_bytes)
                else:
                    decoded_sectors, _ = decode_with_bch_chunked(
                        sectors, ecc_bytes, n_chunks)
                for dec in decoded_sectors:
                    total_bits += dbps
                    if dec is not None:
                        error_bits += int(popcount8[dec].sum())
                    else:
                        error_bits += dbps

        # -- LDPC threshold simulation path ------------------
        # LDPC codeword spans the whole page; t_eff is the page-level
        # correction threshold, so we count errors across ALL bytes.
        elif arch_type == "ldpc":
            t_eff = arch_params["t_eff"]
            k     = arch_params["k"]
            encoded = np.zeros(page_size, dtype=np.uint8)
            sector_data_bytes = k // ns
            dbps = sector_data_bytes * 8
            page_data_bits = k * 8

            for _ in range(num_iters):
                corrupted = inject(encoded, seu_rate, scrub_interval, rng)
                n_page_errors = int(popcount8[corrupted].sum())
                total_bits += page_data_bits
                if n_page_errors > t_eff:
                    error_bits += page_data_bits   # whole page uncorrectable

        else:
            return arch_index, seu_rate, 0, 1

        # Scale by the importance weight: true UBER = p_flip × conditional UBER
        return arch_index, seu_rate, p_flip * error_bits, total_bits

    except Exception as e:
        # Catch any unhandled exception to prevent worker process from crashing
        import sys
        import traceback
        print(f"ERROR in _worker[arch={task[0]}, seu_rate={task[3]:.1e}]: {e}",
              file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Return safe fallback: zero errors (conservative estimate)
        return task[0], float(task[3]), 0, 1


# ===========================================================
#  ANALYSIS RUNNER
# ===========================================================

def run_analysis(page_size, mode, pool=None):
    """Run Monte Carlo SEU-rate sweep and save plots for one page size.

    Parameters
    ----------
    page_size : int
        Total encoded page size in bytes (e.g. 4224 or 8640).
    mode : str
        ``'random'`` -- independent single-bit SEUs only.
        ``'burst'``  -- LEO radiation model with bursts.
    pool : ProcessPoolExecutor, optional
        An existing pool to submit tasks to.  If None, a new pool is
        created and destroyed within this call.

    Returns
    -------
    tuple
        ``(archs, results_map)``
    """
    archs = build_archs(page_size)
    sector_bytes = page_size // NUM_SECTORS

    # -- Print configuration table -----------------------
    print(f"\n{'=' * 72}")
    print(f"  {'BURST (LEO)' if mode == 'burst' else 'RANDOM'} MODE  --  "
          f"page = {page_size} B  ({NUM_SECTORS} sectors x {sector_bytes} B)  "
          f"scrub = {SCRUB_INTERVAL} s")
    print(f"{'=' * 72}")
    print(f"  {'Architecture':<45s}  {'k':>5s}  {'rate':>6s}")
    print(f"  {'-' * 60}")
    for arch in archs:
        print(f"  {arch.label:<45s}  {arch.k:>5d}  {arch.rate:>.4f}")
    print(f"\n  SEU sweep : {SEU_RATE_SWEEP[0]:.1e} -- {SEU_RATE_SWEEP[-1]:.1e}"
          f"  ({len(SEU_RATE_SWEEP)} points)  |  iters : {NUM_ITERS}"
          f"  |  workers : {NUM_WORKERS}\n")

    # -- Build picklable task list -----------------------
    tasks = [
        (ai, arch.arch_type, arch.arch_params,
         float(sr),
         ai * len(SEU_RATE_SWEEP) + si,   # deterministic seed
         NUM_ITERS, mode, SCRUB_INTERVAL)
        for ai, arch in enumerate(archs)
        for si, sr in enumerate(SEU_RATE_SWEEP)
    ]

    # -- Run parallel simulation -------------------------
    results_map = {ai: {} for ai in range(len(archs))}
    arch_done   = {ai: 0  for ai in range(len(archs))}
    total_tasks = len(tasks)
    completed   = 0

    print(f"  Spawning {NUM_WORKERS} worker(s) for {total_tasks} tasks ...")

    def _run_with_pool(p):
        nonlocal completed
        future_to_task = {p.submit(_worker, task): task for task in tasks}
        for future in as_completed(future_to_task):
            try:
                ai, sr, error_bits, total_bits = future.result(timeout=300)
                uber = error_bits / total_bits if total_bits > 0 else 0.0
                results_map[ai][sr] = uber
                completed += 1
                arch_done[ai] += 1
                if arch_done[ai] == len(SEU_RATE_SWEEP):
                    print(f"    [{completed:>4d}/{total_tasks}]  {archs[ai].label}")
            except Exception as e:
                task = future_to_task[future]
                _ai = task[0]
                _sr = task[3]
                print(f"  WARNING: Task failed (arch={_ai}, seu_rate={_sr:.1e}): {e}")
                completed += 1

    if pool is not None:
        # Reuse the caller-supplied pool.
        _run_with_pool(pool)
    else:
        # On Windows, ProcessPoolExecutor automatically uses spawn.
        # Avoid passing mp_context directly due to Python 3.13+ compatibility.
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as p:
            _run_with_pool(p)

    # -- Check if any data was collected -----------------
    total_results = sum(len(results_map[ai]) for ai in range(len(archs)))
    if total_results == 0:
        print("\n  ERROR: No results collected! All worker tasks failed.")
        print("  This is typically caused by:")
        print("    - Insufficient pagefile/memory on system")
        print("    - Incompatible scipy/numpy binary wheels")
        print("    - Missing dependencies in worker environment")
        print("\nTip: Try with NUM_WORKERS = 1 or 2 to reduce memory pressure,")
        print("or manually increase Windows pagefile size.")
        return archs, results_map

    # -- Save plots --------------------------------------
    out_dir = _get_output_dir(mode)
    _plot_uber(page_size, archs, results_map, mode, out_dir)
    _plot_pareto(page_size, archs, results_map, mode, out_dir)

    return archs, results_map


# ===========================================================
#  PLOT HELPERS
# ===========================================================

_MARKERS = {"rs": "^", "bch": "s", "ldpc": "D"}


def _uber_legend_name(arch):
    """Return concise code name for UBER legends.

    Keeps UBER legends short and stable (no t-values or code rate), while
    preserving richer labels for configuration tables and Pareto annotations.
    """
    if arch.arch_type == "rs":
        return f"RS nsym_{arch.arch_params.get('nsym', '?')}"
    if arch.arch_type == "bch":
        ecc = arch.arch_params.get("ecc_bytes", "?")
        n_chunks = arch.arch_params.get("n_chunks", 1)
        suffix = f"_x{n_chunks}cw" if n_chunks > 1 else ""
        return f"BCH {ecc}B{suffix}"
    if arch.arch_type == "ldpc":
        return "LDPC"
    return arch.label.split("(")[0].strip()


def _analytical_results_map(archs):
    """Build a results_map from analytical functions (no MC needed)."""
    results = {}
    for ai, arch in enumerate(archs):
        results[ai] = {}
        if arch.analytical_fn is not None:
            for sr in SEU_RATE_SWEEP:
                results[ai][float(sr)] = arch.analytical_fn(float(sr))
    return results


def _plot_uber(page_size, archs, results_map, mode, output_dir):
    """UBER vs SEU rate -- log-log, with analytical overlay.

    MC curves are solid/dashed; thin dotted lines of the same colour
    show the analytical (Poisson) prediction for comparison.
    """
    mode_label = "LEO burst model" if mode == "burst" else "random (no bursts)"
    fig, ax = plt.subplots(figsize=(14, 8))

    uber_floor = 1.0 / (NUM_ITERS * NUM_SECTORS)

    for ai, arch in enumerate(archs):
        srs   = np.array(sorted(results_map[ai].keys()))
        if len(srs) == 0:
            continue  # Skip if no results for this architecture
        ubers = np.array([results_map[ai][s] for s in srs])
        mask  = ubers > uber_floor   # clip at MC resolution floor
        legend_name = _uber_legend_name(arch)
        kw    = dict(arch.style)

        # MC curve
        if np.any(mask):
            ax.loglog(srs[mask], ubers[mask],
                      marker=".", markersize=5,
                      label=f"MC  {legend_name}", **kw)

        # Analytical overlay (thin dotted line, same colour)
        if arch.analytical_fn is not None:
            uber_a = np.array([arch.analytical_fn(r) for r in srs])
            mask_a = uber_a > 0
            if np.any(mask_a):
                ax.loglog(srs[mask_a], uber_a[mask_a],
                          linewidth=1.2, linestyle=":", alpha=0.5,
                          color=kw.get("color", "gray"),
                          label=f"{legend_name}")

    ax.axhline(UBER_REQ, color="red", linestyle="--", linewidth=1.8,
               alpha=0.8, label=f"UBER requirement ({UBER_REQ:.0e})")

    ax.axhline(uber_floor, color="#888888", linestyle=":", linewidth=1.4,
               alpha=0.9,
               label=f"MC floor")

    ax.set_xlabel("SEU rate  [events / bit / s]")
    ax.set_ylabel("UBER  (Monte Carlo)")
    ax.set_title(
        f"ECC UBER vs SEU Rate  --  page = {page_size} B, "
        f"scrub = {SCRUB_INTERVAL} s,  N = {NUM_ITERS}")
    ax.set_ylim(bottom=1e-30)
    ax.legend(loc="upper left", fontsize=13,
              bbox_to_anchor=(0.01, 0.99), borderaxespad=0)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    fname = f"uber_{page_size}_{mode}.png"
    save_path = os.path.join(output_dir, fname)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)   # release figure; avoids Tk/Qt hang on plt.show()
    print(f"\n  UBER plot   ->  {save_path}")


def _plot_analytical_only(page_size, archs, mode, output_dir):
    """Plot only the analytical (Poisson) UBER curves -- no MC needed."""
    mode_label = "LEO burst model" if mode == "burst" else "random (no bursts)"
    fig, ax = plt.subplots(figsize=(14, 8))

    for arch in archs:
        if arch.analytical_fn is None:
            continue
        uber_a = np.array([arch.analytical_fn(r) for r in SEU_RATE_SWEEP])
        mask   = uber_a > 0
        if not np.any(mask):
            continue
        kw = dict(arch.style)
        legend_name = _uber_legend_name(arch)
        ax.loglog(SEU_RATE_SWEEP[mask], uber_a[mask],
                  label=f"{legend_name}", **kw)

    ax.axhline(UBER_REQ, color="red", linestyle="--", linewidth=1.8,
               alpha=0.8, label=f"UBER RQT")

    ax.set_xlabel("SEU rate  [events / bit / s]")
    ax.set_ylabel("UBER  (analytical)")
    ax.set_title(
        f"ECC UBER vs SEU Rate  --  page = {page_size} B, "
        f"{NUM_SECTORS} sectors, scrub = {SCRUB_INTERVAL} s")
    ax.set_ylim(bottom=1e-30)
    ax.legend(loc="upper left", fontsize=13,
              bbox_to_anchor=(0.01, 0.99), borderaxespad=0)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    fname = f"uber_{page_size}.png"
    save_path = os.path.join(output_dir, fname)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"\n  Analytical UBER plot  ->  {save_path}")


def _plot_pareto(page_size, archs, results_map, mode, output_dir):
    """Pareto -- max tolerable SEU rate (UBER < UBER_REQ) vs code rate.

    Uses log-log linear interpolation to find the crossing point.
    Architectures whose UBER never exceeds UBER_REQ within the sweep
    (i.e. always within spec) are omitted -- they have no meaningful
    crossing and placing them at the sweep edge would be misleading.
    """
    mode_label = "LEO burst model" if mode == "burst" else "random (no bursts)"
    fig, ax = plt.subplots(figsize=(11, 7))

    present_types = set()
    always_ok_labels = []

    for ai, arch in enumerate(archs):
        srs   = np.array(sorted(results_map[ai].keys()))
        if len(srs) == 0:
            continue  # Skip if no results for this architecture
        ubers = np.array([results_map[ai][s] for s in srs])

        # Log-log interpolated crossing point
        above = np.where(ubers > UBER_REQ)[0]
        if len(above) == 0:
            # UBER never exceeded UBER_REQ -- no valid crossing to plot
            always_ok_labels.append(arch.label.split("(")[0].strip())
            continue

        idx = above[0]
        if idx > 0:
            x0 = np.log10(srs[idx - 1])
            x1 = np.log10(srs[idx])
            y0 = np.log10(max(ubers[idx - 1], 1e-50))
            y1 = np.log10(ubers[idx])
            y_req = np.log10(UBER_REQ)
            frac = (y_req - y0) / (y1 - y0) if y1 != y0 else 0.5
            crossing = 10 ** (x0 + frac * (x1 - x0))
        else:
            # UBER already above UBER_REQ at first point -- use leftmost rate
            crossing = srs[0]

        group = arch.arch_type
        present_types.add(group)
        marker = _MARKERS.get(group, "o")
        color  = arch.style.get("color", "black")

        ax.semilogy(arch.rate, crossing,
                    marker=marker, color=color, markersize=13, zorder=5)
        ax.annotate(
            arch.label.split("(")[0].strip(),
            xy=(arch.rate, crossing), fontsize=10,
            textcoords="offset points", xytext=(8, 4),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))

    ax.set_xlabel("Code rate  (k / page_size)")
    ax.set_ylabel(
        f"Max tolerable SEU rate for UBER < {UBER_REQ:.0e}  [events/bit/s]")
    ax.set_title(
        f"ECC Pareto  --  page = {page_size} B, "
        f"scrub = {SCRUB_INTERVAL} s")
    ax.grid(True, which="both", alpha=0.3)

    _legend_defs = [
        ("rs",   "^", TOL_COLORS["blue"],   "RS-only"),
        ("bch",  "s", TOL_COLORS["green"],  "BCH-only"),
        ("ldpc", "D", TOL_COLORS["orange"], "LDPC"),
    ]
    legend_elements = [
        Line2D([0], [0], marker=mk, color="w",
               markerfacecolor=col, markersize=11, label=lbl)
        for model, mk, col, lbl in _legend_defs
        if model in present_types
    ]

    # Collect footnote lines
    footnotes = []
    if always_ok_labels:
        names = ", ".join(always_ok_labels)
        footnotes.append(
            f"Not plotted (always UBER < {UBER_REQ:.0e} within sweep):\n  {names}")

    if footnotes:
        ax.text(
            0.98, 0.02, "\n".join(footnotes),
            transform=ax.transAxes, fontsize=9, color="gray",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="lightgray", alpha=0.8))

    if not present_types:
        ax.text(0.5, 0.5,
                f"No architecture exceeds UBER_REQ = {UBER_REQ:.0e}\nwithin the simulated SEU-rate range.",
                transform=ax.transAxes, fontsize=13, color="gray",
                ha="center", va="center")

    if legend_elements:
        ax.legend(handles=legend_elements, loc="lower left", fontsize=11)
    plt.tight_layout()

    fname = f"pareto_{page_size}.png"
    save_path = os.path.join(output_dir, fname)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)   # release figure; avoids Tk/Qt hang on plt.show()
    print(f"  Pareto plot ->  {save_path}")


# ===========================================================
#  MULTI-PAGE RUNNER
# ===========================================================

def run_all_analyses(page_sizes, mode):
    """Run all page sizes in a single ProcessPoolExecutor batch.

    Submitting every task before entering ``as_completed`` keeps the
    worker pool continuously busy and avoids the Windows hang that
    occurs when the pool goes idle between sequential batches.
    """
    # ------------------------------------------------------------------
    # Phase 1: build archs + tasks for every page size
    # ------------------------------------------------------------------
    page_archs   = {}   # page_size -> list[Arch]
    page_offsets = {}   # page_size -> global arch-index base
    g_offset     = 0
    all_tasks    = []

    for page_size in page_sizes:
        archs = build_archs(page_size)
        sector_bytes = page_size // NUM_SECTORS
        page_archs[page_size]   = archs
        page_offsets[page_size] = g_offset

        print(f"\n{'=' * 72}")
        print(f"  {'BURST (LEO)' if mode == 'burst' else 'RANDOM'} MODE  --  "
              f"page = {page_size} B  ({NUM_SECTORS} sectors x {sector_bytes} B)  "
              f"scrub = {SCRUB_INTERVAL} s")
        print(f"{'=' * 72}")
        print(f"  {'Architecture':<45s}  {'k':>5s}  {'rate':>6s}")
        print(f"  {'-' * 60}")
        for arch in archs:
            print(f"  {arch.label:<45s}  {arch.k:>5d}  {arch.rate:>.4f}")

        for ai, arch in enumerate(archs):
            for si, sr in enumerate(SEU_RATE_SWEEP):
                seed = (g_offset + ai) * len(SEU_RATE_SWEEP) + si
                all_tasks.append((
                    g_offset + ai, arch.arch_type, arch.arch_params,
                    float(sr), seed, NUM_ITERS, mode, SCRUB_INTERVAL
                ))
        g_offset += len(archs)

    n_total_archs = g_offset
    total_tasks   = len(all_tasks)
    print(f"\n  SEU sweep : {SEU_RATE_SWEEP[0]:.1e} -- {SEU_RATE_SWEEP[-1]:.1e}"
          f"  ({len(SEU_RATE_SWEEP)} points)  |  iters : {NUM_ITERS}"
          f"  |  workers : {NUM_WORKERS}")
    print(f"  Total tasks across all page sizes: {total_tasks}")
    print(f"  Spawning {NUM_WORKERS} worker(s) ...")

    # ------------------------------------------------------------------
    # Phase 2: one pool – all tasks submitted before as_completed starts
    # ------------------------------------------------------------------
    all_results  = {i: {} for i in range(n_total_archs)}
    arch_done_g  = {i: 0  for i in range(n_total_archs)}
    completed    = 0

    # Build a reverse lookup: global arch index -> (page_size, local arch index)
    g_to_page = {}
    for page_size in page_sizes:
        off = page_offsets[page_size]
        for ai in range(len(page_archs[page_size])):
            g_to_page[off + ai] = (page_size, ai)

    # Do NOT use a 'with' block here: on Windows the context-manager calls
    # pool.shutdown(wait=True) which blocks until worker *processes* exit.
    # Some Windows installations (especially Windows Store Python) never
    # release the handles, causing an infinite hang.  Instead we collect all
    # results ourselves and then call shutdown(wait=False) to detach.
    pool = ProcessPoolExecutor(max_workers=NUM_WORKERS)
    try:
        future_to_task = {pool.submit(_worker, task): task for task in all_tasks}
        for future in as_completed(future_to_task):
            try:
                ai_g, sr, error_bits, total_bits = future.result(timeout=300)
                uber = error_bits / total_bits if total_bits > 0 else 0.0
                all_results[ai_g][sr] = uber
                arch_done_g[ai_g] += 1
                completed += 1
                ps, ai_local = g_to_page[ai_g]
                archs = page_archs[ps]
                if arch_done_g[ai_g] == len(SEU_RATE_SWEEP):
                    print(f"    [{completed:>5d}/{total_tasks}]  "
                          f"[{ps}B]  {archs[ai_local].label}")
            except Exception as e:
                task = future_to_task[future]
                print(f"  WARNING: Task failed "
                      f"(arch={task[0]}, seu_rate={task[3]:.1e}): {e}")
                completed += 1
    finally:
        # wait=False: all futures already collected; don't block on process exit.
        pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    # Phase 3: plot per page size
    # ------------------------------------------------------------------
    for page_size in page_sizes:
        archs  = page_archs[page_size]
        offset = page_offsets[page_size]
        results_map = {ai: all_results[offset + ai] for ai in range(len(archs))}

        total_results = sum(len(results_map[ai]) for ai in range(len(archs)))
        if total_results == 0:
            print(f"\n  ERROR: No results for page_size={page_size}. "
                  "All worker tasks failed.")
            continue

        out_dir = _get_output_dir(mode)
        _plot_uber(page_size, archs, results_map, mode, out_dir)
        _plot_pareto(page_size, archs, results_map, mode, out_dir)

    # Close all matplotlib figures to prevent hanging on Windows
    plt.close('all')


# ===========================================================
#  ENTRY POINT
# ===========================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Monte-Carlo ECC comparison for multiple NAND page sizes.")
    parser.add_argument(
        "--mode", choices=["random", "burst"], default="random",
        help=(
            "random (default) -- independent single-bit SEUs only.  "
            "burst -- LEO radiation model with 10:1 single-to-burst ratio."))
    parser.add_argument(
        "--analytical-only", action="store_true",
        help=(
            "Skip Monte Carlo simulation entirely and plot only the "
            "analytical (Poisson) UBER curves.  Runs instantly."))
    parser.add_argument(
        "--mc-iter", type=int, default=None, metavar="N",
        help=(
            f"Monte Carlo iterations per (arch × SEU-rate) point "
            f"(default: {NUM_ITERS}).  "
            "Resolution floor = 1 / (N × NUM_SECTORS).  "
            "Use ≥1250 for a 10⁻⁴ floor, ≥12500 for a 10⁻⁵ floor."))
    args = parser.parse_args()

    if args.mc_iter is not None:
        NUM_ITERS = args.mc_iter
        if NUM_ITERS < 1:
            parser.error("--mc-iter must be ≥ 1")
        print(f"  [mc-iter override] NUM_ITERS = {NUM_ITERS}  "
              f"(UBER floor ≈ {1/(NUM_ITERS*NUM_SECTORS):.2e})")

    if args.analytical_only:
        print("  [analytical-only] Skipping Monte Carlo -- plotting Poisson models.")
        out_dir = _get_output_dir("analytical")
        for page_size in PAGE_SIZES:
            archs = build_archs(page_size)
            _plot_analytical_only(page_size, archs, args.mode, out_dir)
            results_map = _analytical_results_map(archs)
            _plot_pareto(page_size, archs, results_map, args.mode, out_dir)
        plt.close("all")
    else:
        # All page sizes are processed in a single executor lifetime so the
        # worker pool never goes idle between batches (avoids Windows hang).
        run_all_analyses(PAGE_SIZES, args.mode)
    # plt.show() intentionally omitted: figures are saved as PNGs above.
    # plt.show() with Tk/Qt backends on Windows Store Python can hang.
    sys.exit(0)

