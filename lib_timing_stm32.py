"""
STM32H753 ECC Performance Simulator.

Extends ecc_timing.py with cycle-cost constants derived from the STM32H753
ARM Cortex-M7 microcontroller. Uses the DTCM memory placement profile to model
the effect of fast zero-wait-state memory on GF arithmetic LUT access latency.

STM32H753 key facts
-------------------
  Core          : ARM Cortex-M7, 6-stage in-order pipeline, superscalar
  Clock         : up to 480 MHz (default here)
  ITCM          : 128 KB - zero-wait-state, 64-bit wide, for code
  DTCM          : 128 KB - zero-wait-state, 64-bit wide, for data / LUTs
"""

import argparse
import os

import lib_timing_base as _base

from lib_timing_base import (
    PAGE_SIZES,
    LDPC_MAX_ITER,
    LDPC_AVG_ITER,
    build_all_timings,
    print_timing_table,
    plot_timing_bars,
    plot_throughput_bars,
    print_operation_breakdown,
)

STM32H753_FREQ_MAX_HZ = 480e6
STM32H753_DTCM_KB = 128

GF8_LOG_TABLE_BYTES = 256
GF8_EXP_TABLE_BYTES = 512
GF8_LUT_TOTAL_BYTES = GF8_LOG_TABLE_BYTES + GF8_EXP_TABLE_BYTES

# Apply DTCM profile constants directly
_base.C_GF8_MUL = 5
_base.C_GF8_ADD = 1
_base.C_GF13_MUL = 60
_base.C_GF13_ADD = 1
_base.C_GF13_LFSR = 3
_base.C_LDPC_MSG = 3
_base.C_LDPC_ENC_EDGE = 2


def print_mcu_header(freq_hz: float):
    """Prints the STM32H753 target header block."""
    freq_label = _fmt_freq(freq_hz)
    
    print("\n" + "=" * 100)
    print("  TARGET: STM32H753 - ARM Cortex-M7 (6-stage in-order, superscalar,")
    print(f"          DSP ext, double-FPU, no CLMUL) | Clock: {freq_label} | Memory model: DTCM")
    print("=" * 100 + "\n")
    print("  GF(2^8) log/antilog LUT (768 B) placed in DTCM.")
    print("  1-cycle zero-wait-state loads; two independent loads may dual-issue")
    print("  on the 64-bit TCM bus. GF(2^13) uses software shift-XOR loop.")
    print("  LDPC graph data in DTCM.\n")
    print("  Cycle constants:")
    print("    C_GF8_MUL   =   5 cyc   GF(2^8) log/antilog LUT multiply")
    print("    C_GF8_ADD   =   1 cyc   GF(2^8) addition (XOR)")
    print("    C_GF13_MUL  =  60 cyc   GF(2^13) SW shift-XOR multiply")
    print("    C_GF13_ADD  =   1 cyc   GF(2^13) addition (XOR)")
    print("    C_LDPC_MSG  =   3 cyc   Gallager-A message update")
    print("    C_LDPC_ENC  =   2 cyc   LDPC encode per-edge XOR\n")


def print_memory_footprint():
    """Prints the LUT memory footprint and DTCM budget analysis."""
    dtcm_kb = STM32H753_DTCM_KB * 1024
    total_lut = GF8_LUT_TOTAL_BYTES
    remaining = dtcm_kb - total_lut
    
    print("  LUT memory footprint")
    print("  ─────────────────────────────────────────────────────────────────")
    print(f"    GF(2^8)  log table       {GF8_LOG_TABLE_BYTES:>7,} B")
    print(f"    GF(2^8)  exp table       {GF8_EXP_TABLE_BYTES:>7,} B")
    print(f"    GF(2^8)  total           {GF8_LUT_TOTAL_BYTES:>7,} B")
    print(f"    GF(2^13)                 software only")
    print("    ─────────────────────────────────────────────────────────")
    print(f"    Total LUT footprint      {total_lut:>7,} B  ({total_lut/1024:.1f} KB)")
    print(f"    DTCM capacity            {dtcm_kb:>7,} B  ({STM32H753_DTCM_KB} KB)")
    print(f"    Fits in DTCM?            {'YES' if total_lut < dtcm_kb else 'NO'}")
    print(f"    Remaining DTCM           {remaining:>7,} B  ({remaining/1024:.1f} KB)\n")


def _fmt_freq(hz: float) -> str:
    """Formats frequency into readable string."""
    if hz >= 1e9: 
        return f"{hz/1e9:.3f} GHz"
    if hz >= 1e6: 
        return f"{hz/1e6:.1f} MHz"
    return f"{hz/1e3:.1f} kHz"


def main():
    parser = argparse.ArgumentParser(
        description="ECC timing simulator for the STM32H753 Cortex-M7 using DTCM profile."
    )
    parser.add_argument(
        "--freq", type=float, default=STM32H753_FREQ_MAX_HZ,
        metavar="HZ", help=f"Core clock in Hz (default: {STM32H753_FREQ_MAX_HZ:.0f})."
    )
    parser.add_argument(
        "--page", type=int, choices=PAGE_SIZES, default=None,
        metavar="BYTES", help=f"Single page size to analyze. Choices: {PAGE_SIZES}."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate bar-chart PNGs in the results/timing/ directory."
    )
    parser.add_argument(
        "--worst-case", action="store_true",
        help="Use worst case LDPC max iterations."
    )
    parser.add_argument(
        "--breakdown", action="store_true",
        help="Print per-operation cycle breakdown."
    )
    parser.add_argument(
        "--footprint", action="store_true",
        help="Print LUT memory footprint and DTCM budget analysis."
    )

    args = parser.parse_args()
    sizes = [args.page] if args.page else PAGE_SIZES
    
    print_mcu_header(args.freq)

    if args.footprint:
        print_memory_footprint()

    print("  NOTE: All cycle counts are order-of-magnitude estimates for a")
    print("  pure-software implementation. Actual performance depends on compiler")
    print("  optimisations (-O2 / -O3), loop unrolling, and LUT cache/TCM state.\n")

    all_timings = {}
    for page_size in sizes:
        timings = build_all_timings(page_size, worst_case=args.worst_case)
        all_timings[page_size] = timings
        
        print_timing_table(timings, args.freq, page_size)
        if args.breakdown:
            print_operation_breakdown(timings, args.freq)

    if args.plot:
        print("  Generating bar charts...")
        os.makedirs("results/timing", exist_ok=True)
        plot_timing_bars(all_timings, args.freq, args.worst_case, output_dir="results/timing")
        plot_throughput_bars(all_timings, args.freq, args.worst_case, output_dir="results/timing")

    print("\n  STM32H753 model parameters")
    print("  ─────────────────────────────────────────────────────────────────────")
    print("  Profile        : DTCM")
    print("  C_GF8_MUL      = 5 cyc")
    print("  C_GF8_ADD      = 1 cyc")
    print("  C_GF13_MUL     = 60 cyc")
    print("  C_GF13_ADD     = 1 cyc")
    print("  C_LDPC_MSG     = 3 cyc")
    print("  C_LDPC_ENC     = 2 cyc")
    print(f"  LDPC iters     = {LDPC_MAX_ITER if args.worst_case else LDPC_AVG_ITER} "
          f"({'worst-case' if args.worst_case else 'average'})")
    print(f"  MCU clock      : {_fmt_freq(args.freq)}\n")


if __name__ == "__main__":
    main()
