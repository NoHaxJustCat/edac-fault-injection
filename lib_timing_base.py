"""
ECC Timing Estimator for MCU-based NAND Flash Protection.

Estimates the encoding (write) and decoding (read) latency for each ECC
architecture, given the clock frequency of the host MCU.
"""

import argparse
import math
import os

from lib_ecc_codecs import compute_data_bytes, _make_bch, BCH_BITS_FROM_ECC

# ---------------------------------------------------------------------------
#  GF arithmetic cycle costs (tunable)
# ---------------------------------------------------------------------------
C_GF8_MUL   = 4
C_GF8_ADD   = 1
C_GF13_MUL  = 25
C_GF13_ADD  = 1
C_GF13_LFSR = 3
BCH_GF_EXP  = 13

C_LDPC_MSG      = 4
C_LDPC_ENC_EDGE = 2
LDPC_MAX_ITER   = 50
LDPC_AVG_ITER   = 15

# ---------------------------------------------------------------------------
#  Architecture definitions
# ---------------------------------------------------------------------------
PAGE_SIZES   = [4224, 8640]
NUM_SECTORS  = 8

RS_NSYMS     = [8, 16]   
BCH_ECC_BYTES = [13, 22, 31]

LDPC_DV = 3
LDPC_DC = 30


def rs_encode_cycles(k_sector: int, nsym: int) -> int:
    cycles = k_sector * nsym * (C_GF8_MUL + C_GF8_ADD)
    return int(cycles + nsym * C_GF8_ADD + 50)


def rs_decode_cycles(n_sector: int, nsym: int) -> int:
    t = nsym // 2
    cyc_syndrome = n_sector * 2 * t * C_GF8_MUL
    cyc_bm = 2 * t * t * (C_GF8_MUL + C_GF8_ADD)
    cyc_chien = n_sector * t * C_GF8_MUL
    cyc_forney = t * t * (C_GF8_MUL + C_GF8_ADD)
    cyc_correct = t * C_GF8_ADD
    return int(cyc_syndrome + cyc_bm + cyc_chien + cyc_forney + cyc_correct + 200)


def bch_encode_cycles(k_sector: int, t: int) -> int:
    return int(k_sector * 8 * t * C_GF8_ADD + 100)


def bch_decode_cycles(n_sector: int, k_sector: int, t: int) -> int:
    n_bits = n_sector * 8
    cyc_syndrome = n_bits * 2 * t * C_GF13_LFSR
    cyc_bm = 2 * t * t * (C_GF13_MUL + C_GF13_ADD)
    cyc_chien = n_bits * t * C_GF13_LFSR
    cyc_correct = t * C_GF13_ADD
    return int(cyc_syndrome + cyc_bm + cyc_chien + cyc_correct + 300)


def ldpc_encode_cycles(page_size: int) -> int:
    return int((page_size * 8) * LDPC_DV * C_LDPC_ENC_EDGE + 300)


def ldpc_decode_cycles(page_size: int, n_iterations: int) -> int:
    msgs_per_iter = 2 * (page_size * 8) * LDPC_DV
    return int(n_iterations * msgs_per_iter * C_LDPC_MSG + 500)


class EccTiming:
    def __init__(self, label: str, code_type: str, page_size: int, k: int, n: int, 
                 redundancy_pct: float, enc_cycles_total: int, dec_cycles_total: int,
                 enc_cycles_per_sector: int, dec_cycles_per_sector: int, n_chunks: int = 1):
        self.label = label
        self.code_type = code_type
        self.page_size = page_size
        self.k = k
        self.n = n
        self.redundancy_pct = redundancy_pct
        self.enc_cycles_total = enc_cycles_total
        self.dec_cycles_total = dec_cycles_total
        self.enc_cycles_sector = enc_cycles_per_sector
        self.dec_cycles_sector = dec_cycles_per_sector
        self.n_chunks = n_chunks

    def enc_time_us(self, freq_hz: float) -> float:
        return self.enc_cycles_total / freq_hz * 1e6

    def dec_time_us(self, freq_hz: float) -> float:
        return self.dec_cycles_total / freq_hz * 1e6

    def enc_throughput_mbits(self, freq_hz: float, nand_program_us: float = 0.0) -> float:
        t_total = (self.enc_cycles_total / freq_hz) + (nand_program_us * 1e-6)
        return (self.k * 8 / t_total) / 1e6 if t_total > 0 else float("inf")

    def dec_throughput_mbits(self, freq_hz: float, nand_read_us: float = 0.0) -> float:
        t_total = (self.dec_cycles_total / freq_hz) + (nand_read_us * 1e-6)
        return (self.k * 8 / t_total) / 1e6 if t_total > 0 else float("inf")


class InfeasibleEntry:
    feasible = False
    def __init__(self, label: str, code_type: str, page_size: int, reason: str):
        self.label = label
        self.code_type = code_type
        self.page_size = page_size
        self.reason = reason


def build_rs_timing(nsym: int, page_size: int):
    t = nsym // 2
    label = f"RS nsym={nsym} (t={t})"
    sector = page_size // NUM_SECTORS
    try:
        k = compute_data_bytes(page_size, NUM_SECTORS, rs_nsyms=[nsym])
    except ValueError:
        return InfeasibleEntry(label, "rs", page_size, f"invalid chunking")

    k_sector = k // NUM_SECTORS
    n_sector = page_size // NUM_SECTORS

    enc_per_sector = rs_encode_cycles(k_sector, nsym)
    dec_per_sector = rs_decode_cycles(n_sector, nsym)
    
    return EccTiming(
        label=label,
        code_type="rs",
        page_size=page_size,
        k=k, n=page_size,
        redundancy_pct=(page_size - k) / page_size * 100.0,
        enc_cycles_total=enc_per_sector * NUM_SECTORS,
        dec_cycles_total=dec_per_sector * NUM_SECTORS,
        enc_cycles_per_sector=enc_per_sector,
        dec_cycles_per_sector=dec_per_sector,
    )


def build_bch_timing(ecc_bytes: int, page_size: int):
    t_bits = BCH_BITS_FROM_ECC.get(ecc_bytes)
    label = f"BCH {ecc_bytes}B (t={t_bits})" if t_bits else f"BCH {ecc_bytes}B"
    try:
        bch_obj = _make_bch(ecc_bytes)
    except ValueError as exc:
        return InfeasibleEntry(label, "bch", page_size, str(exc))

    label = f"BCH {ecc_bytes}B (t={bch_obj.t})"
    sector_bytes = page_size // NUM_SECTORS
    max_codeword_bytes = 8191 // 8
    max_data_bytes = (8191 - bch_obj.ecc_bits) // 8

    n_chunks = math.ceil(sector_bytes / max_codeword_bytes)
    while sector_bytes % n_chunks != 0:
        n_chunks += 1

    chunk_encoded_bytes = sector_bytes // n_chunks
    chunk_data_bytes = chunk_encoded_bytes - bch_obj.ecc_bytes

    if chunk_data_bytes <= 0 or chunk_data_bytes > max_data_bytes:
        return InfeasibleEntry(label, "bch", page_size, "invalid capacity")

    t = bch_obj.t
    k = chunk_data_bytes * n_chunks * NUM_SECTORS

    enc_per_sector = n_chunks * bch_encode_cycles(chunk_data_bytes, t)
    dec_per_sector = n_chunks * bch_decode_cycles(chunk_encoded_bytes, chunk_data_bytes, t)

    return EccTiming(
        label=label,
        code_type="bch",
        page_size=page_size,
        k=k, n=page_size,
        redundancy_pct=(page_size - k) / page_size * 100.0,
        enc_cycles_total=enc_per_sector * NUM_SECTORS,
        dec_cycles_total=dec_per_sector * NUM_SECTORS,
        enc_cycles_per_sector=enc_per_sector,
        dec_cycles_per_sector=dec_per_sector,
    )


def build_ldpc_timing(page_size: int, worst_case: bool = False):
    iters = LDPC_MAX_ITER if worst_case else LDPC_AVG_ITER
    dv, dc = LDPC_DV, LDPC_DC
    
    k = int(page_size * (1 - dv / dc))
    k -= k % NUM_SECTORS

    enc_total = ldpc_encode_cycles(page_size)
    dec_total = ldpc_decode_cycles(page_size, iters)
    
    return EccTiming(
        label=f"LDPC dv={dv},dc={dc}",
        code_type="ldpc",
        page_size=page_size,
        k=k, n=page_size,
        redundancy_pct=(page_size - k) / page_size * 100.0,
        enc_cycles_total=enc_total,
        dec_cycles_total=dec_total,
        enc_cycles_per_sector=enc_total // NUM_SECTORS,
        dec_cycles_per_sector=dec_total // NUM_SECTORS,
    )


def build_all_timings(page_size: int, worst_case: bool = False):
    timings = []
    for nsym in RS_NSYMS:
        timings.append(build_rs_timing(nsym, page_size))
    for ecc in BCH_ECC_BYTES:
        timings.append(build_bch_timing(ecc, page_size))
    timings.append(build_ldpc_timing(page_size, worst_case))
    return timings


def print_timing_table(timings, freq_hz: float, page_size: int):
    print(f"\n  Performance Estimates — {page_size} B page @ {freq_hz/1e6:.1f} MHz")
    print("  " + "─" * 85)
    print(f"  {'Architecture':<22} | {'Data':<6} | {'Redund.':<7} | {'Encode':<10} | {'Decode':<10}")
    print("  " + "─" * 85)
    
    for t in timings:
        if isinstance(t, InfeasibleEntry):
            print(f"  {t.label:<22} | N/A    | N/A     | N/A        | N/A")
            continue
        print(f"  {t.label:<22} | {t.k:<6} | {t.redundancy_pct:>6.2f}% | {t.enc_time_us(freq_hz):>7.2f} µs | {t.dec_time_us(freq_hz):>7.2f} µs")
    print()


def print_operation_breakdown(timings, freq_hz: float):
    print("\n  Cycle count breakdown")
    print("  " + "─" * 70)
    for t in timings:
        if isinstance(t, InfeasibleEntry):
            continue
        print(f"  {t.label}")
        print(f"    Encode : {t.enc_cycles_total:>9,} cycles")
        print(f"    Decode : {t.dec_cycles_total:>9,} cycles")
    print()


def plot_timing_bars(all_timings, freq_hz: float, worst_case: bool, output_dir: str = "results/timing"):
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    for page_size, timings in all_timings.items():
        valid = [t for t in timings if not isinstance(t, InfeasibleEntry)]
        if not valid: continue
        
        labels = [t.label for t in valid]
        enc_us = [t.enc_time_us(freq_hz) for t in valid]
        dec_us = [t.dec_time_us(freq_hz) for t in valid]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], enc_us, width=0.4, label="Encode", color="#4477AA")
        ax.bar([i + 0.2 for i in x], dec_us, width=0.4, label="Decode", color="#EE6677")
        
        ax.set_ylabel("Latency (µs)")
        ax.set_title(f"ECC Latency — {page_size}B @ {freq_hz/1e6:.1f}MHz")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/timing_{page_size}B.png")
        plt.close()


def plot_throughput_bars(all_timings, freq_hz: float, worst_case: bool, output_dir: str = "results/timing"):
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    for page_size, timings in all_timings.items():
        valid = [t for t in timings if not isinstance(t, InfeasibleEntry)]
        if not valid: continue
        
        labels = [t.label for t in valid]
        enc_tp = [t.enc_throughput_mbits(freq_hz) for t in valid]
        dec_tp = [t.dec_throughput_mbits(freq_hz) for t in valid]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], enc_tp, width=0.4, label="Encode (Write)", color="#228833")
        ax.bar([i + 0.2 for i in x], dec_tp, width=0.4, label="Decode (Read)", color="#CCBB44")
        
        ax.set_ylabel("Throughput (Mbit/s)")
        ax.set_title(f"ECC Throughput — {page_size}B @ {freq_hz/1e6:.1f}MHz")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_{page_size}B.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECC timing estimator limit generator.")
    parser.add_argument("--freq", type=float, default=168e6, help="Core clock in Hz.")
    parser.add_argument("--page", type=int, choices=PAGE_SIZES, help="Single page size.")
    parser.add_argument("--plot", action="store_true", help="Generate bar charts.")
    parser.add_argument("--worst-case", action="store_true", help="Worst case LDPC.")
    
    args = parser.parse_args()
    sizes = [args.page] if args.page else PAGE_SIZES

    all_t = {sz: build_all_timings(sz, args.worst_case) for sz in sizes}
    for sz, timings in all_t.items():
        print_timing_table(timings, args.freq, sz)

    if args.plot:
        plot_timing_bars(all_t, args.freq, args.worst_case)
        plot_throughput_bars(all_t, args.freq, args.worst_case)

