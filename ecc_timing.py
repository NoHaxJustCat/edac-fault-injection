"""ECC Timing Estimator for MCU-based NAND Flash Protection.

Estimates the encoding (write) and decoding (read) latency for each ECC
architecture considered in compare_ecc_pagesizes.py and
compare_ecc_burst_pagesizes.py, given the clock frequency of the host MCU.

ECC architectures analysed
--------------------------
  RS-only    : nsym ∈ {8, 16}, GF(2^8), sector-wise (same as other scripts)
  BCH-only   : ecc_bytes ∈ {13, 22, 31}, GF(2^13), sector-wise (auto-chunked
               into ×N sub-codewords when the sector exceeds the 1023 B / 8191-bit
               GF(2^13) codeword limit, matching compare_ecc_single.py)
  LDPC       : Gallager-A, dv=3, dc=30, page-wide (threshold model)

Timing model
------------
All estimates are based on a **software** implementation without hardware ECC
acceleration.  The dominant cost is Galois-Field arithmetic, parameterised by
the clock-cycle costs below.  These match typical ARM Cortex-M4/M7 benchmarks
using precomputed lookup tables (GF(2^8)) or software CLMUL emulation
(GF(2^13)).

  Constant        Default  Description
  ─────────────── ───────  ────────────────────────────────────────────────
  C_GF8_MUL          4    Cycles per GF(2^8)  multiply (log/antilog LUT)
  C_GF8_ADD          1    Cycles per GF(2^8)  addition (XOR)
  C_GF13_MUL        25    Cycles per GF(2^13) multiply (soft CLMUL emu)
  C_GF13_ADD         1    Cycles per GF(2^13) addition (XOR)
  C_GF13_LFSR        3    Cycles per GF(2^13) LFSR step (fixed α^j multiply)
  C_LDPC_MSG         4    Cycles per LDPC msg-passing update
  C_LDPC_ENC_EDGE   2    Cycles per sparse-matrix XOR edge (LDPC encode)
  LDPC_MAX_ITER     50    Maximum Gallager-A decoder iterations
  LDPC_AVG_ITER     15    Average Gallager-A iterations (benign case)

Reed-Solomon encoding model (per sector, systematic):
  Each data byte requires nsym GF(2^8) multiply-accumulate LFSR steps.
  cycles_enc  =  k_sector × nsym × (C_GF8_MUL + C_GF8_ADD)
  where k_sector = raw data bytes per sector.

Reed-Solomon decoding model (per sector, one codeword chunk):
  1. Syndrome     : n_sector × 2t × C_GF8_MUL          (Horner eval, 2t syndromes)
  2. BM algorithm : 2t² × (C_GF8_MUL + C_GF8_ADD)      (error-locator polynomial)
  3. Chien search : n_sector × t × C_GF8_MUL            (roots of error-locator)
  4. Forney       : t² × (C_GF8_MUL + C_GF8_ADD)        (error magnitudes)
  cycles_dec  ≈  n_sector × 3t × C_GF8_MUL  +  3t² × C_GF8_MUL
  where n_sector = encoded bytes per sector, t = nsym // 2.

BCH encoding model (per sector, or per chunk when chunked):
  Systematic LFSR division of data polynomial by the generator.  Processing
  one bit at a time with t feedback taps (degree-2t generator):
  cycles_enc  =  k_sector × 8 × t × C_GF2_BIT
  where C_GF2_BIT ≈ 1 (single boolean XOR per feedback tap per bit).
  For chunked sectors (n_chunks > 1): total = n_chunks × cycles_enc_per_chunk.

BCH decoding model (per sector, or per chunk when chunked):
  1. Syndrome     : n_sector_bits × 2t × C_GF13_LFSR     (LFSR shift per syndrome register)
  2. BM algorithm : n_chunks × 2t² × C_GF13_MUL          (arbitrary GF(2^m) multiplies)
  3. Chien search : n_sector_bits × t × C_GF13_LFSR      (fixed α^j update per coefficient)
  4. Bit correction: t × C_GF13_ADD
  cycles_dec  ≈  n_sector_bits × 3t × C_GF13_LFSR  +  n_chunks × 2t² × C_GF13_MUL
  where m = BCH_GF_EXP = 13.  C_GF13_LFSR models a fixed-constant multiply
  by α^j implemented as a register-only shift+conditional-XOR (no memory access),
  which is much cheaper than an arbitrary GF(2^m) multiply (C_GF13_MUL).

LDPC (Gallager-A, dv=3, dc=30, page-wide) model:
  Each iteration: every variable node sends dv messages to check nodes
                  every check node  sends dc messages to variable nodes.
  n_vn = page_bits, n_cn ≈ page_bits × (dv/dc)
  messages_per_iter = n_vn × dv  +  n_cn × dc  ≈  n_vn × 2dv
  cycles_enc  =  page_bits × dv × C_LDPC_ENC_EDGE  (sparse parity-check generation)
  cycles_dec  =  iters × page_bits × 2dv × C_LDPC_MSG

Usage
-----
    python ecc_timing.py                          # 168 MHz, all page sizes
    python ecc_timing.py --freq 400e6             # 400 MHz (STM32H7)
    python ecc_timing.py --freq 64e6              # 64 MHz  (STM32G0)
    python ecc_timing.py --freq 168e6 --page 4224 # single page size
    python ecc_timing.py --plot                   # generate bar chart
    python ecc_timing.py --worst-case             # use LDPC max iterations
"""

import argparse
import math
import os
import sys

import numpy as np

from utils import compute_data_bytes, _make_bch, BCH_BITS_FROM_ECC

# ---------------------------------------------------------------------------
#  GF arithmetic cycle costs (tunable)
# ---------------------------------------------------------------------------
C_GF8_MUL   = 4    # GF(2^8)  multiply  (log/antilog LUT, 2 reads + 1 XOR)
C_GF8_ADD   = 1    # GF(2^8)  add        (XOR)
C_GF13_MUL  = 25   # GF(2^13) multiply  (software CLMUL emulation)
C_GF13_ADD  = 1    # GF(2^13) add        (XOR)
C_GF13_LFSR = 3    # GF(2^13) fixed-constant multiply (LFSR shift+XOR, register-only)
BCH_GF_EXP  = 13   # BCH field exponent (bchlib uses GF(2^13))

C_LDPC_MSG      = 4   # cycles per Gallager-A message update
C_LDPC_ENC_EDGE = 2   # cycles per sparse-matrix XOR edge (LDPC systematic encode)
LDPC_MAX_ITER   = 50  # worst-case iterations
LDPC_AVG_ITER  = 15  # average iterations (typical benign channel)

# ---------------------------------------------------------------------------
#  Architecture definitions (mirrors compare_ecc_burst_pagesizes.py)
# ---------------------------------------------------------------------------
PAGE_SIZES   = [4224, 8640]
NUM_SECTORS  = 8

RS_NSYMS     = [8, 16]   
BCH_ECC_BYTES = [13, 22, 31]

LDPC_DV = 3
LDPC_DC = 30

# ---------------------------------------------------------------------------
#  NAND device profiles (page read / program / erase timings)
# ---------------------------------------------------------------------------
#  Each device maps page_size → {read_us, program_us, erase_us}.
#  read_us / program_us are per-page; erase_us is per-block.
NAND_DEVICES = {
    "MT29F256G08": {
        "desc": "MT29F256G08AUCABPB-10ITZ:A",
        "page_size": 8640,
        "read_us":    35.0,
        "program_us": 350.0,
        "erase_us":   1500.0,
    },
    "MT29F512G08": {
        "desc": "MT29F512G08CUAAAC5:A",
        "page_size": 8640,
        "read_us":    75.0,
        "program_us": 1300.0,
        "erase_us":   3800.0,
    },
    "3DFN128G08": {
        "desc": "3DFN128G08US8761",
        "page_size": 4224,
        "read_us":    27.0,
        "program_us": 230.0,
        "erase_us":   700.0,
    },
}

def nand_devices_for_page(page_size: int) -> list:
    """Return list of NAND device keys whose page_size matches."""
    return [k for k, v in NAND_DEVICES.items() if v["page_size"] == page_size]


# ---------------------------------------------------------------------------
#  Cycle-count models
# ---------------------------------------------------------------------------

def rs_encode_cycles(k_sector: int, nsym: int) -> int:
    """RS encoding cycles for one sector.

    LFSR-based systematic encoding in GF(2^8): each of the k_sector data
    symbols feeds through an nsym-tap LFSR, requiring nsym multiply-
    accumulate operations per symbol.

    Args:
        k_sector : raw data bytes per sector
        nsym     : RS parity symbols (code redundancy)

    Returns:
        Estimated clock cycles to encode one sector.
    """
    mac_per_symbol = nsym  # nsym GF mults + nsym GF adds (fused)
    cycles = k_sector * mac_per_symbol * (C_GF8_MUL + C_GF8_ADD)
    # Overhead: output the nsym parity symbols (one read each) + housekeeping
    cycles += nsym * C_GF8_ADD + 50
    return int(cycles)


def rs_decode_cycles(n_sector: int, nsym: int) -> int:
    """RS decoding cycles for one sector (worst case: errors present).

    Pipeline: syndrome → Berlekamp-Massey → Chien search → Forney.

    Args:
        n_sector : total encoded bytes per sector (data + parity)
        nsym     : RS parity symbols; correction capability t = nsym // 2

    Returns:
        Estimated clock cycles to decode one sector (error-present path).
    """
    t = nsym // 2
    # 1. Syndrome computation: n_sector symbol evaluations × 2t syndromes
    cyc_syndrome = n_sector * 2 * t * C_GF8_MUL
    # 2. Berlekamp-Massey: ~2t^2 GF multiplications
    cyc_bm = 2 * t * t * (C_GF8_MUL + C_GF8_ADD)
    # 3. Chien search: n_sector evaluations × polynomial degree t
    cyc_chien = n_sector * t * C_GF8_MUL
    # 4. Forney algorithm: t^2 GF operations
    cyc_forney = t * t * (C_GF8_MUL + C_GF8_ADD)
    # 5. Error correction: t symbol XORs (trivial)
    cyc_correct = t * C_GF8_ADD
    # Overhead / branch logic
    overhead = 200
    # NOTE: This models the worst-case (errors present) path.  In the error-free
    # case, all syndromes are zero and the decoder can early-exit after step 1,
    # skipping BM, Chien, and Forney (saving ~60-80% of decode cycles at low BER).
    return int(cyc_syndrome + cyc_bm + cyc_chien + cyc_forney + cyc_correct + overhead)


def bch_encode_cycles(k_sector: int, t: int) -> int:
    """BCH encoding cycles for one sector.

    Systematic LFSR polynomial division: k_sector × 8 one-bit steps
    through a degree-2t shift register.  Each bit step requires t
    feedback XOR operations (one per tap in the generator polynomial).

    Args:
        k_sector : raw data bytes per sector
        t        : BCH error-correction capability (BCH_BITS from bchlib)

    Returns:
        Estimated clock cycles to encode one sector.
    """
    # Bit-serial LFSR: one XOR per feedback tap per bit
    cycles = k_sector * 8 * t * C_GF8_ADD  # GF(2) — just XOR
    # Overhead for byte extraction and parity output
    cycles += 100
    return int(cycles)


def bch_decode_cycles(n_sector: int, k_sector: int, t: int) -> int:
    """BCH decoding cycles for one sector (worst case: errors present).

    Pipeline: syndrome → Berlekamp-Massey → Chien search → bit correction.
    BCH operates over GF(2^m), m = BCH_GF_EXP = 13.

    Syndrome computation and Chien search use fixed-constant multiplies
    by powers of α (LFSR shift + conditional XOR), costing C_GF13_LFSR
    per step — a register-only operation independent of memory placement.
    Only Berlekamp-Massey requires full arbitrary GF(2^m) multiplies.

    Args:
        n_sector : total encoded bytes per sector (data + parity)
        k_sector : raw data bytes per sector
        t        : BCH error-correction capability (BCH_BITS from bchlib)

    Returns:
        Estimated clock cycles to decode one sector (error-present path).
    """
    n_bits = n_sector * 8
    # 1. Syndrome: n_bits LFSR steps, 2t syndrome registers.
    #    Each step is a fixed multiply by α^j (shift + conditional XOR).
    cyc_syndrome = n_bits * 2 * t * C_GF13_LFSR
    # 2. BM over GF(2^m): ~2t^2 full GF(2^m) multiplications
    cyc_bm = 2 * t * t * (C_GF13_MUL + C_GF13_ADD)
    # 3. Chien search: incremental α^j update per coefficient per position.
    #    Each step is a fixed-constant multiply (LFSR shift), not a full GF mult.
    cyc_chien = n_bits * t * C_GF13_LFSR
    # 4. Bit correction: t XOR operations (trivial)
    cyc_correct = t * C_GF13_ADD
    # Overhead
    overhead = 300
    return int(cyc_syndrome + cyc_bm + cyc_chien + cyc_correct + overhead)


def ldpc_encode_cycles(page_size: int) -> int:
    """LDPC encoding cycles (full page, Gallager-A dv=3, dc=30).

    Sparse parity-check matrix with variable-node degree dv=3: each
    coded bit participates in dv parity equations.  A single forward
    pass computes all check-node parities.

    Args:
        page_size : total page size in bytes

    Returns:
        Estimated clock cycles to encode one page.
    """
    page_bits = page_size * 8
    # Systematic encode via sparse parity-check matrix: each variable node
    # contributes to dv parity equations (one XOR + sparse index access per edge).
    # Uses C_LDPC_ENC_EDGE (not C_LDPC_MSG) since encoding is a single-pass
    # sparse matrix-vector product, not iterative message-passing.
    cycles = page_bits * LDPC_DV * C_LDPC_ENC_EDGE
    cycles += 300  # overhead
    return int(cycles)


def ldpc_decode_cycles(page_size: int, n_iterations: int) -> int:
    """LDPC decoding cycles (Gallager-A, hard-decision, dv=3, dc=30).

    Per iteration:
      - Variable nodes → check nodes: n_vn × dv messages
      - Check nodes → variable nodes: n_cn × dc messages
    With n_cn = n_vn × (dv / dc), message counts balance:
      total messages = n_vn × dv + n_cn × dc = 2 × n_vn × dv

    Args:
        page_size    : total page size in bytes
        n_iterations : number of decoder iterations to model

    Returns:
        Estimated clock cycles to decode one page.
    """
    page_bits = page_size * 8
    msgs_per_iter = 2 * page_bits * LDPC_DV
    cycles = n_iterations * msgs_per_iter * C_LDPC_MSG
    cycles += 500  # overhead
    return int(cycles)


# ---------------------------------------------------------------------------
#  Architecture builder
# ---------------------------------------------------------------------------

class EccTiming:
    """Holds timing breakdown for one ECC architecture."""

    def __init__(self, label: str, code_type: str, page_size: int,
                 k: int, n: int, redundancy_pct: float,
                 enc_cycles_total: int, dec_cycles_total: int,
                 enc_cycles_per_sector: int, dec_cycles_per_sector: int,
                 n_chunks: int = 1):
        self.label              = label
        self.code_type          = code_type
        self.page_size          = page_size
        self.k                  = k       # raw data bytes per page
        self.n                  = n       # encoded bytes per page (= page_size)
        self.redundancy_pct     = redundancy_pct
        self.enc_cycles_total   = enc_cycles_total
        self.dec_cycles_total   = dec_cycles_total
        self.enc_cycles_sector  = enc_cycles_per_sector
        self.dec_cycles_sector  = dec_cycles_per_sector
        self.n_chunks           = n_chunks

    def enc_time_us(self, freq_hz: float) -> float:
        return self.enc_cycles_total / freq_hz * 1e6

    def dec_time_us(self, freq_hz: float) -> float:
        return self.dec_cycles_total / freq_hz * 1e6

    def enc_throughput_mbits(self, freq_hz: float,
                              nand_program_us: float = 0.0) -> float:
        """Effective write throughput in Mbit/s.

        Throughput = k * 8 / (t_ecc_encode + t_nand_program).
        """
        t_ecc_s  = self.enc_cycles_total / freq_hz
        t_nand_s = nand_program_us * 1e-6
        t_total  = t_ecc_s + t_nand_s
        if t_total <= 0:
            return float("inf")
        return (self.k * 8 / t_total) / 1e6

    def dec_throughput_mbits(self, freq_hz: float,
                              nand_read_us: float = 0.0) -> float:
        """Effective read throughput in Mbit/s.

        Throughput = k * 8 / (t_ecc_decode + t_nand_read).
        """
        t_ecc_s  = self.dec_cycles_total / freq_hz
        t_nand_s = nand_read_us * 1e-6
        t_total  = t_ecc_s + t_nand_s
        if t_total <= 0:
            return float("inf")
        return (self.k * 8 / t_total) / 1e6


class InfeasibleEntry:
    """Placeholder for an ECC configuration that cannot be realised for this page size.

    Two root causes are tracked:
      RS nsym=N infeasible : the encoded sector size does not decompose into
                             an integer number of GF(2^8) RS chunks of ≤255-N bytes.
      BCH infeasible       : the sector size in bits exceeds the GF(2^13)
                             maximum codeword length of 2^13-1 = 8191 bits.
    """

    feasible = False   # class-level sentinel used by callers

    def __init__(self, label: str, code_type: str, page_size: int, reason: str):
        self.label      = label
        self.code_type  = code_type
        self.page_size  = page_size
        self.reason     = reason


def build_rs_timing(nsym: int, page_size: int):
    """Compute RS-only timing for the given page size and nsym."""
    t      = nsym // 2
    label  = f"RS nsym={nsym} (t={t})"
    sector = page_size // NUM_SECTORS
    try:
        k = compute_data_bytes(page_size, NUM_SECTORS, rs_nsyms=[nsym])
    except ValueError:
        max_chunk = 255 - nsym
        return InfeasibleEntry(
            label=label, code_type="rs", page_size=page_size,
            reason=(
                f"sector {sector} B cannot be split into GF(2^8) "
                f"RS-({255},{max_chunk}) chunks with integer count"
            ),
        )

    k_sector = k // NUM_SECTORS
    n_sector = page_size // NUM_SECTORS
    t = nsym // 2
    label = f"RS nsym={nsym} (t={t})"

    enc_per_sector = rs_encode_cycles(k_sector, nsym)
    dec_per_sector = rs_decode_cycles(n_sector, nsym)

    enc_total = enc_per_sector * NUM_SECTORS
    dec_total = dec_per_sector * NUM_SECTORS

    redundancy = (page_size - k) / page_size * 100.0

    return EccTiming(
        label=label,
        code_type="rs",
        page_size=page_size,
        k=k, n=page_size,
        redundancy_pct=redundancy,
        enc_cycles_total=enc_total,
        dec_cycles_total=dec_total,
        enc_cycles_per_sector=enc_per_sector,
        dec_cycles_per_sector=dec_per_sector,
    )


def build_bch_timing(ecc_bytes: int, page_size: int):
    """Compute BCH-only timing for the given page size and ecc_bytes.

    When the sector size exceeds the GF(2^13) codeword limit (8191 bits /
    1023 bytes), the sector is automatically split into the minimum number
    of equal-sized sub-sector chunks, each independently BCH-encoded.
    This mirrors the chunking strategy in compare_ecc_single.py.
    """
    t_bits = BCH_BITS_FROM_ECC.get(ecc_bytes)
    label  = f"BCH {ecc_bytes}B (t={t_bits})" if t_bits else f"BCH {ecc_bytes}B"
    try:
        bch_obj = _make_bch(ecc_bytes)
    except ValueError as exc:
        return InfeasibleEntry(label=label, code_type="bch",
                               page_size=page_size, reason=str(exc))

    label              = f"BCH {ecc_bytes}B (t={bch_obj.t})"
    sector_bytes       = page_size // NUM_SECTORS
    max_codeword_bytes = 8191 // 8           # = 1023 B
    max_data_bytes     = (8191 - bch_obj.ecc_bits) // 8

    # Find the smallest n_chunks so every chunk fits within the GF(2^13) limit.
    # n_chunks must also divide sector_bytes evenly for equal-sized sub-blocks.
    n_chunks = math.ceil(sector_bytes / max_codeword_bytes)
    while sector_bytes % n_chunks != 0:
        n_chunks += 1

    chunk_encoded_bytes = sector_bytes // n_chunks
    chunk_data_bytes    = chunk_encoded_bytes - bch_obj.ecc_bytes

    if chunk_data_bytes <= 0:
        return InfeasibleEntry(
            label=label, code_type="bch", page_size=page_size,
            reason=(
                f"chunk {chunk_encoded_bytes} B ≤ BCH parity "
                f"{bch_obj.ecc_bytes} B — no room for data"
            ),
        )
    if chunk_data_bytes > max_data_bytes:
        chunk_bits = chunk_encoded_bytes * 8
        gf_max     = 2 ** BCH_GF_EXP - 1
        return InfeasibleEntry(
            label=label, code_type="bch", page_size=page_size,
            reason=(
                f"chunk {chunk_encoded_bytes} B = {chunk_bits} bits "
                f"> GF(2^{BCH_GF_EXP}) max codeword {gf_max} bits"
            ),
        )

    t         = bch_obj.t
    k         = chunk_data_bytes * n_chunks * NUM_SECTORS

    enc_per_sector = n_chunks * bch_encode_cycles(chunk_data_bytes, t)
    dec_per_sector = n_chunks * bch_decode_cycles(chunk_encoded_bytes, chunk_data_bytes, t)

    enc_total  = enc_per_sector * NUM_SECTORS
    dec_total  = dec_per_sector * NUM_SECTORS
    redundancy = (page_size - k) / page_size * 100.0

    return EccTiming(
        label=f"BCH {ecc_bytes}B (t={t})",
        code_type="bch",
        page_size=page_size,
        k=k, n=page_size,
        redundancy_pct=redundancy,
        enc_cycles_total=enc_total,
        dec_cycles_total=dec_total,
        enc_cycles_per_sector=enc_per_sector,
        dec_cycles_per_sector=dec_per_sector,
    )


def build_ldpc_timing(page_size: int, worst_case: bool = False) -> EccTiming:
    """Compute LDPC (Gallager-A, dv=3, dc=30) timing."""
    dv, dc = LDPC_DV, LDPC_DC
    k      = int(page_size * (1 - dv / dc))
    k     -= k % NUM_SECTORS

    n_iters    = LDPC_MAX_ITER if worst_case else LDPC_AVG_ITER
    enc_total  = ldpc_encode_cycles(page_size)
    dec_total  = ldpc_decode_cycles(page_size, n_iters)

    redundancy = (page_size - k) / page_size * 100.0
    iter_label = "max" if worst_case else "avg"

    return EccTiming(
        label=f"LDPC dv={dv},dc={dc} ({iter_label}={n_iters} iter)",
        code_type="ldpc",
        page_size=page_size,
        k=k, n=page_size,
        redundancy_pct=redundancy,
        enc_cycles_total=enc_total,
        dec_cycles_total=dec_total,
        enc_cycles_per_sector=enc_total // NUM_SECTORS,
        dec_cycles_per_sector=dec_total // NUM_SECTORS,
    )


def build_all_timings(page_size: int, worst_case: bool = False) -> list:
    """Build the full canonical ECC timing list for *page_size*.

    Every RS, BCH and LDPC configuration is always included.  Configurations
    that are not algebraically realisable for *page_size* are represented as
    :class:`InfeasibleEntry` objects so callers can display them as N/A rather
    than silently omitting them.
    """
    results = []
    for nsym in RS_NSYMS:
        results.append(build_rs_timing(nsym, page_size))
    for eb in BCH_ECC_BYTES:
        results.append(build_bch_timing(eb, page_size))
    results.append(build_ldpc_timing(page_size, worst_case=worst_case))
    return results


# ---------------------------------------------------------------------------
#  Table formatting helpers
# ---------------------------------------------------------------------------

COL_W = {
    "label":    28,
    "k":         7,
    "red":       7,
    "enc_cyc":  12,
    "dec_cyc":  12,
    "enc_us":   10,
    "dec_us":   10,
    "enc_mbits": 12,
    "dec_mbits": 12,
}


def _row(label, k, red, enc_cyc, dec_cyc, enc_us, dec_us, enc_mbits, dec_mbits):
    return (
        f"  {label:<{COL_W['label']}}"
        f"  {k:>{COL_W['k']}}"
        f"  {red:>{COL_W['red']}}"
        f"  {enc_cyc:>{COL_W['enc_cyc']}}"
        f"  {dec_cyc:>{COL_W['dec_cyc']}}"
        f"  {enc_us:>{COL_W['enc_us']}}"
        f"  {dec_us:>{COL_W['dec_us']}}"
        f"  {enc_mbits:>{COL_W['enc_mbits']}}"
        f"  {dec_mbits:>{COL_W['dec_mbits']}}"
    )


_NA = "N/A"


def _row_infeasible(label: str, reason: str) -> str:
    """Format a single N/A row for an infeasible ECC configuration."""
    # Truncate reason so it fits roughly within the row width
    reason_col = f"(infeasible: {reason})"
    return (
        f"  {label:<{COL_W['label']}}"
        f"  {_NA:>{COL_W['k']}}"
        f"  {_NA:>{COL_W['red']}}"
        f"  {_NA:>{COL_W['enc_cyc']}}"
        f"  {_NA:>{COL_W['dec_cyc']}}"
        f"  {reason_col}"
    )


def print_timing_table(timings: list, freq_hz: float, page_size: int,
                       nand_key: str = None):
    """Print the tabular timing summary.

    If *nand_key* is given, throughput columns include the NAND read/program
    latency in addition to the ECC compute time.
    """
    nand = NAND_DEVICES.get(nand_key) if nand_key else None
    read_us    = nand["read_us"]    if nand else 0.0
    program_us = nand["program_us"] if nand else 0.0
    thr_label  = "Mbit/s" if nand else "Mbit/s"

    header = _row(
        "Architecture", "k (B)", "Red. %",
        "Enc cycles", "Dec cycles",
        "Enc (µs)", "Dec (µs)",
        f"Enc {thr_label}", f"Dec {thr_label}",
    )
    sep = "  " + "-" * (len(header) - 2)

    label_freq = f"{freq_hz/1e6:.0f} MHz" if freq_hz >= 1e6 else f"{freq_hz/1e3:.0f} kHz"
    print()
    print(f"  Page size : {page_size} B ({page_size // NUM_SECTORS} B/sector, "
          f"{NUM_SECTORS} sectors)")
    print(f"  MCU clock : {label_freq}")
    if nand:
        print(f"  NAND      : {nand['desc']}  "
              f"(read={read_us} µs, program={program_us} µs, "
              f"erase={nand['erase_us']} µs)")
        print(f"  Throughput: k×8 / (t_ECC + t_NAND)  [Mbit/s]")
    else:
        print(f"  Throughput: k×8 / t_ECC  [Mbit/s, no NAND I/O]")
    print(f"  Dec model : worst-case (errors present) for RS/BCH")
    print()
    print(header)
    print(sep)
    prev_type = None
    for t in timings:
        if prev_type and t.code_type != prev_type:
            print(sep)
        prev_type = t.code_type
        if isinstance(t, InfeasibleEntry):
            print(_row_infeasible(t.label, t.reason))
        else:
            print(_row(
                t.label,
                f"{t.k}",
                f"{t.redundancy_pct:.1f}",
                f"{t.enc_cycles_total:,}",
                f"{t.dec_cycles_total:,}",
                f"{t.enc_time_us(freq_hz):.2f}",
                f"{t.dec_time_us(freq_hz):.2f}",
                f"{t.enc_throughput_mbits(freq_hz, program_us):.2f}",
                f"{t.dec_throughput_mbits(freq_hz, read_us):.2f}",
            ))
    print(sep)
    print()


# ---------------------------------------------------------------------------
#  Optional bar chart
# ---------------------------------------------------------------------------

def plot_timing_bars(all_timings: dict, freq_hz: float,
                     worst_case: bool, output_dir: str = "images/timing"):
    """Generate encode / decode latency bar charts for all page sizes.

    Feasible entries are drawn as solid colour bars with value labels.
    Infeasible entries are drawn as white/hatched N/A markers so the full
    canonical set of ECC schemes is always visible on the x-axis.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.patches as mpatches
    except ImportError:
        print("[warning] matplotlib not available — skipping plot generation")
        return

    mpl.rcParams.update({
        "font.family":    "serif",
        "font.size":      9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi":     150,
    })

    label_freq = f"{freq_hz/1e6:.0f} MHz"
    os.makedirs(output_dir, exist_ok=True)

    for page_size, timings in all_timings.items():
        labels = [t.label for t in timings]
        x      = np.arange(len(labels))
        w      = 0.38
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 4.5))

        # Separate feasible / infeasible indices
        feasible_idx   = [i for i, t in enumerate(timings) if not isinstance(t, InfeasibleEntry)]
        infeasible_idx = [i for i, t in enumerate(timings) if isinstance(t, InfeasibleEntry)]

        enc_times_us = [timings[i].enc_time_us(freq_hz) for i in feasible_idx]
        dec_times_us = [timings[i].dec_time_us(freq_hz) for i in feasible_idx]

        # Feasible bars
        bars_enc = ax.bar(x[feasible_idx] - w / 2, enc_times_us, width=w,
                          label="Encode (write)", color="#4477AA", alpha=0.85)
        bars_dec = ax.bar(x[feasible_idx] + w / 2, dec_times_us, width=w,
                          label="Decode (read)",  color="#EE6677", alpha=0.85)

        # Value labels above feasible bars
        for bar, h in zip(bars_enc, enc_times_us):
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7, color="#4477AA")
        for bar, h in zip(bars_dec, dec_times_us):
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7, color="#EE6677")

        # Infeasible placeholder bars
        if infeasible_idx:
            y_max = max(dec_times_us) if dec_times_us else 1.0
            stub_h = y_max * 0.06   # short stub so the hatch is visible
            for i in infeasible_idx:
                for offset in (-w / 2, w / 2):
                    ax.bar(x[i] + offset, stub_h, width=w,
                           color="none", edgecolor="#888888",
                           hatch="//", linewidth=0.8)
                ax.text(x[i], stub_h * 1.3, "N/A",
                        ha="center", va="bottom", fontsize=7,
                        color="#888888", style="italic")

        iter_tag = "worst-case" if worst_case else "avg-iter"
        ax.set_title(
            f"ECC Encoding / Decoding Latency — {page_size} B page @ {label_freq}\n"
            f"(software model, {iter_tag} LDPC)",
            pad=8,
        )
        ax.set_ylabel("Latency (µs)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

        # Legend: add N/A patch if any infeasible entries exist
        handles, leg_labels = ax.get_legend_handles_labels()
        if infeasible_idx:
            na_patch = mpatches.Patch(
                facecolor="none", edgecolor="#888888",
                hatch="//", linewidth=0.8, label="N/A (infeasible)"
            )
            handles.append(na_patch)
        ax.legend(handles=handles, framealpha=0.9)

        y_top = max(dec_times_us) * 1.15 if dec_times_us else 1.0
        ax.set_ylim(0, y_top)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        fig.tight_layout()
        fname = os.path.join(
            output_dir,
            f"ecc_timing_{page_size}B_{int(freq_hz/1e6)}MHz.png",
        )
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_throughput_bars(all_timings: dict, freq_hz: float,
                         worst_case: bool, output_dir: str = "images/timing"):
    """Generate encode / decode throughput (Mbit/s) bar charts per NAND device.

    For each page size, one chart is produced per matching NAND device.
    Each chart has paired encode/decode bars per ECC architecture.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.patches as mpatches
    except ImportError:
        print("[warning] matplotlib not available — skipping throughput plot")
        return

    mpl.rcParams.update({
        "font.family":    "serif",
        "font.size":      9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi":     150,
    })

    label_freq = f"{freq_hz/1e6:.0f} MHz"
    os.makedirs(output_dir, exist_ok=True)

    for page_size, timings in all_timings.items():
        nand_keys = nand_devices_for_page(page_size)
        if not nand_keys:
            continue
        for nand_key in nand_keys:
            nand = NAND_DEVICES[nand_key]
            read_us    = nand["read_us"]
            program_us = nand["program_us"]

            labels = [t.label for t in timings]
            x      = np.arange(len(labels))
            w      = 0.38

            feasible_idx   = [i for i, t in enumerate(timings)
                              if not isinstance(t, InfeasibleEntry)]
            infeasible_idx = [i for i, t in enumerate(timings)
                              if isinstance(t, InfeasibleEntry)]

            enc_thr = [timings[i].enc_throughput_mbits(freq_hz, program_us)
                       for i in feasible_idx]
            dec_thr = [timings[i].dec_throughput_mbits(freq_hz, read_us)
                       for i in feasible_idx]

            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 4.5))

            bars_enc = ax.bar(x[feasible_idx] - w / 2, enc_thr, width=w,
                              label="Write (enc + program)", color="#4477AA",
                              alpha=0.85)
            bars_dec = ax.bar(x[feasible_idx] + w / 2, dec_thr, width=w,
                              label="Read (dec + page read)", color="#EE6677",
                              alpha=0.85)

            for bar, h in zip(bars_enc, enc_thr):
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=7,
                        color="#4477AA")
            for bar, h in zip(bars_dec, dec_thr):
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=7,
                        color="#EE6677")

            if infeasible_idx:
                all_vals = enc_thr + dec_thr
                y_max  = max(all_vals) if all_vals else 1.0
                stub_h = y_max * 0.06
                for i in infeasible_idx:
                    for offset in (-w / 2, w / 2):
                        ax.bar(x[i] + offset, stub_h, width=w,
                               color="none", edgecolor="#888888",
                               hatch="//", linewidth=0.8)
                    ax.text(x[i], stub_h * 1.3, "N/A",
                            ha="center", va="bottom", fontsize=7,
                            color="#888888", style="italic")

            iter_tag = "worst-case" if worst_case else "avg-iter"
            ax.set_title(
                f"Effective Throughput — {nand['desc']}\n"
                f"{page_size} B page @ {label_freq}  "
                f"(read={read_us} µs, prog={program_us} µs, "
                f"{iter_tag} LDPC)",
                pad=8,
            )
            ax.set_ylabel("Throughput (Mbit/s)")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

            handles, _ = ax.get_legend_handles_labels()
            if infeasible_idx:
                handles.append(mpatches.Patch(
                    facecolor="none", edgecolor="#888888",
                    hatch="//", linewidth=0.8, label="N/A (infeasible)"))
            ax.legend(handles=handles, framealpha=0.9)

            all_vals = enc_thr + dec_thr
            y_top = max(all_vals) * 1.15 if all_vals else 1.0
            ax.set_ylim(0, y_top)
            ax.yaxis.grid(True, linestyle="--", alpha=0.5)
            ax.set_axisbelow(True)

            fig.tight_layout()
            fname = os.path.join(
                output_dir,
                f"ecc_throughput_{nand_key}_{page_size}B_"
                f"{int(freq_hz/1e6)}MHz.png",
            )
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
#  Detailed per-operation breakdown
# ---------------------------------------------------------------------------

def print_operation_breakdown(timings: list, freq_hz: float):
    """Print a detailed per-operation cycle breakdown for each architecture."""
    print("  ── Detailed cycle breakdown (per sector) ──────────────────────")
    for arch in timings:
        if isinstance(arch, InfeasibleEntry):
            print(f"\n  {arch.label}  (page {arch.page_size} B)")
            print(f"    INFEASIBLE: {arch.reason}")
            continue
        print(f"\n  {arch.label}  (page {arch.page_size} B)")
        k_s = arch.k // NUM_SECTORS
        n_s = arch.n // NUM_SECTORS

        if arch.code_type == "rs":
            nsym = int(arch.label.split("nsym=")[1].split()[0])
            t    = nsym // 2
            lbl_enc = [
                ("LFSR MACs (k×nsym×(C_mul+C_add))",
                 k_s * nsym * (C_GF8_MUL + C_GF8_ADD)),
                ("Parity output overhead", nsym * C_GF8_ADD + 50),
            ]
            lbl_dec = [
                (f"Syndrome (n×2t×C_mul)",
                 n_s * 2 * t * C_GF8_MUL),
                (f"Berlekamp-Massey (2t²×(C_mul+C_add))",
                 2 * t * t * (C_GF8_MUL + C_GF8_ADD)),
                (f"Chien search (n×t×C_mul)",
                 n_s * t * C_GF8_MUL),
                (f"Forney (t²×(C_mul+C_add))",
                 t * t * (C_GF8_MUL + C_GF8_ADD)),
                ("Correction + overhead", t * C_GF8_ADD + 200),
            ]

        elif arch.code_type == "bch":
            t_str    = arch.label.split("(t=")[1].split(")")[0]
            t        = int(t_str)
            n_chunks = arch.n_chunks
            ecc_b    = n_s - k_s
            n_bits   = n_s * 8
            m        = BCH_GF_EXP
            lbl_enc = [
                (f"LFSR steps (k_bits×t×C_add)",
                 k_s * 8 * t * C_GF8_ADD),
                ("Output + overhead", 100 * n_chunks),
            ]
            lbl_dec = [
                (f"Syndrome (n_bits×2t×C_gf13_mul/m)",
                 n_bits * 2 * t * (C_GF13_MUL / m)),
                (f"Berlekamp-Massey ({n_chunks}×2t²×(C_gf13_mul+add))",
                 n_chunks * 2 * t * t * (C_GF13_MUL + C_GF13_ADD)),
                (f"Chien search (n_bits×t×C_gf13_mul/m)",
                 n_bits * t * (C_GF13_MUL / m)),
                ("Bit correction + overhead", t * C_GF13_ADD + 300 * n_chunks),
            ]

        else:  # ldpc
            page_bits = arch.n * 8
            iter_str  = arch.label.split("=")[2].split()[0]
            n_iters   = int(iter_str)
            lbl_enc = [
                (f"Parity generation (page_bits×dv×C_enc_edge)",
                 page_bits * LDPC_DV * C_LDPC_ENC_EDGE),
                ("Overhead", 300),
            ]
            lbl_dec = [
                (f"Message passing ({n_iters} iter × page_bits×2×dv×C_msg)",
                 n_iters * 2 * page_bits * LDPC_DV * C_LDPC_MSG),
                ("Overhead", 500),
            ]

        print(f"    Encode:")
        for desc, cyc in lbl_enc:
            us = cyc / freq_hz * 1e6
            print(f"      {desc:<48}  {cyc:>10,} cyc  ({us:.3f} µs)")
        total_enc = sum(v for _, v in lbl_enc)
        print(f"      {'TOTAL (per sector)':<48}  {total_enc:>10,} cyc  ({total_enc/freq_hz*1e6:.3f} µs)")

        print(f"    Decode (worst case):")
        for desc, cyc in lbl_dec:
            us = cyc / freq_hz * 1e6
            print(f"      {desc:<48}  {cyc:>10,} cyc  ({us:.3f} µs)")
        total_dec = sum(v for _, v in lbl_dec)
        print(f"      {'TOTAL (per sector)':<48}  {total_dec:>10,} cyc  ({total_dec/freq_hz*1e6:.3f} µs)")

    print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    global C_GF8_MUL, C_GF13_MUL, C_LDPC_MSG, C_LDPC_ENC_EDGE

    parser = argparse.ArgumentParser(
        description="Estimate ECC encode/decode latency for an MCU given its clock frequency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--freq", type=float, default=168e6,
        metavar="HZ",
        help="MCU clock frequency in Hz (default: 168e6 = 168 MHz, STM32F4)",
    )
    parser.add_argument(
        "--page", type=int, choices=PAGE_SIZES, default=None,
        metavar="BYTES",
        help=f"Analyse a single NAND page size. "
             f"Choices: {PAGE_SIZES}. Default: all.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate bar-chart PNG(s) in the images/timing/ directory.",
    )
    parser.add_argument(
        "--worst-case", action="store_true",
        help=f"Use LDPC_MAX_ITER={LDPC_MAX_ITER} iterations for LDPC "
             f"(default: LDPC_AVG_ITER={LDPC_AVG_ITER}).",
    )
    parser.add_argument(
        "--breakdown", action="store_true",
        help="Print a per-operation cycle breakdown for every architecture.",
    )
    parser.add_argument(
        "--c-gf8-mul", type=int, default=C_GF8_MUL,
        metavar="CYC",
        help=f"Override GF(2^8) multiply cycle cost (default {C_GF8_MUL}).",
    )
    parser.add_argument(
        "--c-gf13-mul", type=int, default=C_GF13_MUL,
        metavar="CYC",
        help=f"Override GF(2^13) multiply cycle cost (default {C_GF13_MUL}).",
    )
    parser.add_argument(
        "--c-ldpc-msg", type=int, default=C_LDPC_MSG,
        metavar="CYC",
        help=f"Override LDPC message-update cycle cost (default {C_LDPC_MSG}).",
    )
    parser.add_argument(
        "--c-ldpc-enc-edge", type=int, default=C_LDPC_ENC_EDGE,
        metavar="CYC",
        help=f"Override LDPC encode per-edge cycle cost (default {C_LDPC_ENC_EDGE}).",
    )
    parser.add_argument(
        "--nand", type=str, choices=list(NAND_DEVICES), default=None,
        metavar="DEVICE",
        help=(
            "NAND device for throughput calculation. "
            "Adds page-read/program latency to ECC time. "
            f"Choices: {', '.join(NAND_DEVICES)}. "
            "Default: none (ECC-only throughput)."
        ),
    )

    args = parser.parse_args()

    # Apply overrides to module-level constants
    C_GF8_MUL       = args.c_gf8_mul
    C_GF13_MUL      = args.c_gf13_mul
    C_LDPC_MSG      = args.c_ldpc_msg
    C_LDPC_ENC_EDGE = args.c_ldpc_enc_edge

    freq_hz    = args.freq
    worst_case = args.worst_case
    nand_arg   = args.nand
    sizes      = [args.page] if args.page else PAGE_SIZES

    # If a NAND device is specified, restrict page sizes to the matching one
    if nand_arg:
        nand_ps = NAND_DEVICES[nand_arg]["page_size"]
        if args.page and args.page != nand_ps:
            print(f"  ERROR: --nand {nand_arg} has page_size={nand_ps} B "
                  f"but --page {args.page} was specified.")
            sys.exit(1)
        sizes = [nand_ps]

    if freq_hz >= 400e6:
        print()
        print("  WARNING: Clock frequency >= 400 MHz suggests an STM32H7-class target.")
        print("  The default GF cycle constants here are tuned for a generic Cortex-M4.")
        print("  For STM32H753-specific memory hierarchy models (DTCM, D-cache, AXI SRAM),")
        print("  use stm32h753_ecc_timing.py instead, or override constants with")
        print("  --c-gf8-mul / --c-gf13-mul.")
        print()

    freq_label = (f"{freq_hz/1e9:.3f} GHz" if freq_hz >= 1e9
                  else f"{freq_hz/1e6:.1f} MHz" if freq_hz >= 1e6
                  else f"{freq_hz/1e3:.1f} kHz")

    print()
    print("=" * 100)
    print(f"  ECC TIMING ESTIMATOR  ·  Clock: {freq_label}  ·  "
          f"GF8-mul={C_GF8_MUL} cyc  GF13-mul={C_GF13_MUL} cyc  "
          f"LDPC-msg={C_LDPC_MSG} cyc  LDPC-enc={C_LDPC_ENC_EDGE} cyc")
    print("=" * 100)
    print()
    print("  NOTE: All cycle counts are order-of-magnitude estimates for a "
          "pure-software")
    print("  implementation on ARM Cortex-class hardware (no hardware ECC "
          "accelerator).")
    print("  Actual performance will differ based on cache behaviour, compiler "
          "optimisations,")
    print("  and library implementation quality. Adjust --c-gf8-mul / "
          "--c-gf13-mul to calibrate.")
    print()

    all_timings: dict = {}
    for page_size in sizes:
        timings = build_all_timings(page_size, worst_case=worst_case)
        all_timings[page_size] = timings

        if nand_arg:
            # Single NAND device specified — one table with its timings
            print_timing_table(timings, freq_hz, page_size, nand_key=nand_arg)
        else:
            # Print one table per matching NAND device, plus a bare ECC-only table
            device_keys = nand_devices_for_page(page_size)
            for nk in device_keys:
                print_timing_table(timings, freq_hz, page_size, nand_key=nk)
            if not device_keys:
                print_timing_table(timings, freq_hz, page_size)

        if args.breakdown:
            print_operation_breakdown(timings, freq_hz)

    if args.plot:
        print("  Generating bar charts …")
        plot_timing_bars(all_timings, freq_hz, worst_case)
        plot_throughput_bars(all_timings, freq_hz, worst_case)

    # Print model parameter summary
    print("  Model parameters")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  C_GF8_MUL   = {C_GF8_MUL:3d} cyc   (GF(2^8)  multiply, log/antilog LUT)")
    print(f"  C_GF8_ADD   = {C_GF8_ADD:3d} cyc   (GF(2^8)  add, XOR)")
    print(f"  C_GF13_MUL  = {C_GF13_MUL:3d} cyc   (GF(2^13) multiply, soft CLMUL emulation)")
    print(f"  C_GF13_ADD  = {C_GF13_ADD:3d} cyc   (GF(2^13) add, XOR)")
    print(f"  C_LDPC_MSG  = {C_LDPC_MSG:3d} cyc   (Gallager-A message update)")
    print(f"  C_LDPC_ENC  = {C_LDPC_ENC_EDGE:3d} cyc   (LDPC encode per-edge XOR)")
    print(f"  LDPC iters  = {LDPC_MAX_ITER if worst_case else LDPC_AVG_ITER:3d}        "
          f"({'worst-case' if worst_case else 'average'})")
    print(f"  NUM_SECTORS = {NUM_SECTORS:3d}")
    print()


if __name__ == "__main__":
    main()
