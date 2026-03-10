"""STM32H753 ECC Performance Simulator.

Extends ecc_timing.py with cycle-cost constants derived from the STM32H753
ARM Cortex-M7 microcontroller. Four memory placement profiles are available
to model the effect of DTCM, AXI SRAM, and the L1 D-cache on GF arithmetic
LUT access latency.

STM32H753 key facts
-------------------
  Core          : ARM Cortex-M7, 6-stage in-order pipeline, superscalar
  Clock         : up to 480 MHz (default here)
  I-cache       : 32 KB (4-way set-associative)
  D-cache       : 32 KB (4-way set-associative)
  ITCM          : 128 KB — zero-wait-state, 64-bit wide, for code
  DTCM          : 128 KB — zero-wait-state, 64-bit wide, for data / LUTs
  AXI SRAM      : up to 512 KB (D1 domain) — 1 wait state (2-cycle access)
  Flash         : up to 2 MB — 6–8 wait states at 480 MHz; bypassed via cache
  Accelerators  : CRYP, HASH, RNG — NO dedicated ECC engine
  DSP / FPU     : DSP extension (UMULL/SMULL), double-precision FPU
  CLMUL         : absent → GF(2^13) requires pure XOR-based SW emulation
  Pipeline IPC  : ~1 instruction/cycle for dependent chains; up to 2 for
                  independent pairs (dual-issue arithmetic + load/store)

GF cycle-cost derivation
-------------------------
  GF(2^8) multiply — log/antilog LUT (exp_table 512 B + log_table 256 B = 768 B):
  ┌──────────────────────────────┬────────────────────────────────────────┐
  │ Profile                      │ Derivation                             │
  ├──────────────────────────────┼────────────────────────────────────────┤
  │ DTCM   (1-cyc load)          │ LDR log[a]      → 1 cyc               │
  │                              │ LDR log[b]      → 1 cyc (independent,  │
  │                              │   may dual-issue on 64-bit TCM bus)    │
  │                              │ ADD + mod-255   → 2 cyc               │
  │                              │ LDR exp[sum]    → 1 cyc               │
  │                              │ Total ≈ 5 cyc                          │
  ├──────────────────────────────┼────────────────────────────────────────┤
  │ AXI SRAM (2-cyc load)        │ Same sequence, 2 cyc/load:            │
  │                              │ 2 + 2 + 2 + 2 = 8 +overhead ≈ 9 cyc  │
  ├──────────────────────────────┼────────────────────────────────────────┤
  │ D-cache  (4-cyc hit latency) │ Cortex-M7 L1 hit: 4-cyc load-to-use  │
  │                              │ Two loads can be issued back-to-back   │
  │                              │ but stall on the add: 4+2+4 ≈ 11 cyc  │
  └──────────────────────────────┴────────────────────────────────────────┘

  GF(2^13) multiply — no CLMUL, schoolbook carry-less multiply + reduction:
  ┌──────────────────────────────┬────────────────────────────────────────┐
  │ Software CLMUL emulation     │ 13-iteration shift-XOR loop (poly 8219)│
  │  (all profiles except LUT)   │  per iteration: shift(1) + tst(1)     │
  │                              │  + CSEL/IT-XOR(1) + tst-bit13(1)      │
  │                              │  + conditional reduce(1) = ~5 cyc/bit │
  │                              │  13 × 5 = 65 cyc; Cortex-M7 IT-block  │
  │                              │  and branch-prediction: ~60 cyc        │
  ├──────────────────────────────┼────────────────────────────────────────┤
  │ DTCM LUT  (1-cyc load)       │ log/antilog tables, 2-byte entries:   │
  │  (dtcm_gf13lut profile only) │ LDRH log[a] + LDRH log[b] (1+1)      │
  │                              │ ADD + mod-8191 (1+2) + LDRH exp[s](1)  │
  │                              │ Total ≈ 7 cyc                          │
  │                              │ LUT footprint: ~32 KB of DTCM budget  │
  └──────────────────────────────┴────────────────────────────────────────┘

  LDPC Gallager-A message update (dv=3, dc=30, binary messages):
  ┌──────────────────────────────┬────────────────────────────────────────┐
  │ DTCM                         │ BIT ops + graph accesses from DTCM    │
  │                              │ ~3 cyc/msg (simple XOR/AND, hot data) │
  ├──────────────────────────────┼────────────────────────────────────────┤
  │ D-cache                      │ Graph topology causes L1 pressure at  │
  │                              │ large page sizes → ~4 cyc/msg          │
  ├──────────────────────────────┼────────────────────────────────────────┤
  │ AXI SRAM                     │ Graph structure in AXI SRAM:          │
  │                              │ ~5 cyc/msg (2-cyc loads + overhead)   │
  └──────────────────────────────┴────────────────────────────────────────┘

Memory profiles
---------------
  dtcm         GF(2^8)  LUT in DTCM (768 B).  GF(2^13) software.
               Best balance: minimal TCM usage, fast RS/BCH encode.
  dtcm_gf13lut Both GF(2^8) and GF(2^13) LUTs in DTCM (~32 KB total).
               Fastest BCH decode; requires ≥33 KB free DTCM.
  dcache       All LUTs accessed via 32 KB D-cache.  GF(2^13) software.
               Realistic when DTCM is consumed by application data/stack.
  axisram      LUTs in AXI SRAM (D1 domain, 1 wait state), no D-cache.
               Worst-case conservative estimate.

Usage
-----
    python stm32h753_ecc_timing.py                        # DTCM profile, 480 MHz
    python stm32h753_ecc_timing.py --freq 400e6           # lower clock
    python stm32h753_ecc_timing.py --memory-model dcache  # D-cache profile
    python stm32h753_ecc_timing.py --compare              # all 4 profiles side-by-side
    python stm32h753_ecc_timing.py --plot                 # bar charts to images/
    python stm32h753_ecc_timing.py --breakdown            # per-operation cycle counts
    python stm32h753_ecc_timing.py --worst-case           # LDPC max iterations
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
#  Import the base timing engine and patch its module-level constants
# ---------------------------------------------------------------------------
import ecc_timing as _base

# Re-export constants we override so callers can inspect them
from ecc_timing import (
    PAGE_SIZES, NUM_SECTORS,
    RS_NSYMS, BCH_ECC_BYTES, BCH_GF_EXP,
    LDPC_DV, LDPC_DC, LDPC_MAX_ITER, LDPC_AVG_ITER,
    build_all_timings, print_timing_table, plot_timing_bars,
    print_operation_breakdown, EccTiming, InfeasibleEntry,
)

# ---------------------------------------------------------------------------
#  STM32H753 fixed hardware constants
# ---------------------------------------------------------------------------
STM32H753_FREQ_MAX_HZ = 480e6   # maximum core clock
STM32H753_DTCM_KB     = 128     # DTCM size
STM32H753_ITCM_KB     = 128     # ITCM size
STM32H753_DCACHE_KB   = 32      # L1 data cache
STM32H753_ICACHE_KB   = 32      # L1 instruction cache

# GF LUT memory footprints
GF8_LOG_TABLE_BYTES   = 256     # log_table[256], uint8
GF8_EXP_TABLE_BYTES   = 512     # exp_table[512], uint8 (doubled for wrap)
GF8_LUT_TOTAL_BYTES   = GF8_LOG_TABLE_BYTES + GF8_EXP_TABLE_BYTES   # 768 B

GF13_LOG_TABLE_BYTES  = 8192 * 2  # log_table[8192], uint16
GF13_EXP_TABLE_BYTES  = 8192 * 2  # exp_table[8192], uint16 (doubled trick)
GF13_LUT_TOTAL_BYTES  = GF13_LOG_TABLE_BYTES + GF13_EXP_TABLE_BYTES  # 32 KB

# ---------------------------------------------------------------------------
#  Memory / configuration profiles
# ---------------------------------------------------------------------------
#  Each profile is a dict with keys:
#    gf8_mul     : cycles per GF(2^8)  multiply (log/antilog LUT)
#    gf8_add     : cycles per GF(2^8)  addition (XOR)
#    gf13_mul    : cycles per GF(2^13) multiply
#    gf13_add    : cycles per GF(2^13) addition (XOR)
#    ldpc_msg    : cycles per Gallager-A message update
#    gf13_lut    : bool — True if GF(2^13) also uses LUT (not SW emulation)
#    description : human-readable summary
# ---------------------------------------------------------------------------
PROFILES = {
    "dtcm": {
        "gf8_mul":  5,
        "gf8_add":  1,
        "gf13_mul": 60,
        "gf13_add": 1,
        "ldpc_msg": 3,
        "gf13_lut": False,
        "short":    "DTCM",
        "description": (
            "GF(2^8) log/antilog LUT (768 B) placed in DTCM. "
            "1-cycle zero-wait-state loads; two independent loads may dual-issue "
            "on the 64-bit TCM bus. GF(2^13) uses software shift-XOR loop "
            "(no carryless-multiply instruction on Cortex-M7). "
            "LDPC graph data in DTCM."
        ),
    },
    "dtcm_gf13lut": {
        "gf8_mul":  5,
        "gf8_add":  1,
        "gf13_mul": 7,
        "gf13_add": 1,
        "ldpc_msg": 3,
        "gf13_lut": True,
        "short":    "DTCM + GF13-LUT",
        "description": (
            "Both GF(2^8) (768 B) and GF(2^13) (~32 KB) log/antilog LUTs "
            "placed in DTCM (total ~33 KB of 128 KB budget). "
            "GF(2^13) multiply: 2×LDRH (1 cyc each) + ADD + mod-8191 + LDRH = ~7 cyc. "
            "Fastest BCH performance; reserve at least 33 KB of DTCM for LUTs."
        ),
    },
    "dcache": {
        "gf8_mul":  11,
        "gf8_add":  1,
        "gf13_mul": 65,
        "gf13_add": 1,
        "ldpc_msg": 4,
        "gf13_lut": False,
        "short":    "D-cache",
        "description": (
            "LUTs reside in Flash or AXI SRAM, accessed through the 32 KB D-cache. "
            "Cortex-M7 L1 hit latency: 4 cycles (load-to-use). Three dependent "
            "loads in the GF(2^8) multiply chain → effective ~11 cycles with "
            "pipelining of the two independent log lookups. GF(2^13) software. "
            "LDPC graph may cause cache pressure at larger page sizes (~4 cyc/msg)."
        ),
    },
    "axisram": {
        "gf8_mul":  14,
        "gf8_add":  1,
        "gf13_mul": 68,
        "gf13_add": 1,
        "ldpc_msg": 5,
        "gf13_lut": False,
        "short":    "AXI SRAM (no cache)",
        "description": (
            "Conservative worst-case: LUTs in AXI D1 SRAM (1 wait state = "
            "2-cycle loads) with D-cache disabled or bypassed. "
            "3 loads × 2 cyc + overhead ≈ 14 cyc for GF(2^8) multiply. "
            "GF(2^13) software, working set scattered across SRAM (68 cyc). "
            "Suitable for safety-critical configurations where caching is banned."
        ),
    },
}

DEFAULT_PROFILE = "dtcm"


# ---------------------------------------------------------------------------
#  Profile application — patches ecc_timing module-level constants in place
# ---------------------------------------------------------------------------

def apply_profile(name: str) -> dict:
    """Patch ecc_timing's module-level cycle constants with *name* profile.

    Args:
        name : key from PROFILES dict

    Returns:
        The profile dict that was applied.
    """
    if name not in PROFILES:
        raise ValueError(
            f"Unknown profile '{name}'. "
            f"Valid choices: {list(PROFILES)}"
        )
    p = PROFILES[name]
    _base.C_GF8_MUL  = p["gf8_mul"]
    _base.C_GF8_ADD  = p["gf8_add"]
    _base.C_GF13_MUL = p["gf13_mul"]
    _base.C_GF13_ADD = p["gf13_add"]
    _base.C_LDPC_MSG = p["ldpc_msg"]
    return p


# ---------------------------------------------------------------------------
#  STM32H753 info / memory footprint helpers
# ---------------------------------------------------------------------------

def print_mcu_header(freq_hz: float, profile_name: str):
    """Print the STM32H753 target header block."""
    p = PROFILES[profile_name]
    freq_label = _fmt_freq(freq_hz)
    print()
    print("=" * 100)
    print("  TARGET: STM32H753  —  ARM Cortex-M7  (6-stage in-order, superscalar,")
    print("          DSP ext, double-FPU, no CLMUL)  |  "
          f"Clock: {freq_label}  |  "
          f"Memory model: {p['short']}")
    print("=" * 100)
    print()
    print(f"  {p['description']}")
    print()
    print(f"  Cycle constants (patched into ecc_timing module):")
    print(f"    C_GF8_MUL   = {p['gf8_mul']:3d} cyc   GF(2^8)  log/antilog LUT multiply")
    print(f"    C_GF8_ADD   = {p['gf8_add']:3d} cyc   GF(2^8)  addition (XOR)")
    print(f"    C_GF13_MUL  = {p['gf13_mul']:3d} cyc   GF(2^13) "
          f"{'log/antilog LUT multiply' if p['gf13_lut'] else 'SW shift-XOR multiply (no CLMUL)'}")
    print(f"    C_GF13_ADD  = {p['gf13_add']:3d} cyc   GF(2^13) addition (XOR)")
    print(f"    C_LDPC_MSG  = {p['ldpc_msg']:3d} cyc   Gallager-A message update")
    print()


def print_memory_footprint(profile_name: str):
    """Print the LUT memory footprint and DTCM budget analysis."""
    p        = PROFILES[profile_name]
    dtcm_kb  = STM32H753_DTCM_KB * 1024  # bytes)

    gf8_bytes  = GF8_LUT_TOTAL_BYTES
    gf13_bytes = GF13_LUT_TOTAL_BYTES if p["gf13_lut"] else 0
    total_lut  = gf8_bytes + gf13_bytes
    remaining  = dtcm_kb - total_lut

    print("  LUT memory footprint")
    print("  ─────────────────────────────────────────────────────────────────")
    print(f"    GF(2^8)  log table       {GF8_LOG_TABLE_BYTES:>7,} B  (256 × uint8)")
    print(f"    GF(2^8)  exp table       {GF8_EXP_TABLE_BYTES:>7,} B  (512 × uint8, doubled)")
    print(f"    GF(2^8)  total           {gf8_bytes:>7,} B")
    if p["gf13_lut"]:
        print(f"    GF(2^13) log table       {GF13_LOG_TABLE_BYTES:>7,} B  (8192 × uint16)")
        print(f"    GF(2^13) exp table       {GF13_EXP_TABLE_BYTES:>7,} B  (8192 × uint16, doubled)")
        print(f"    GF(2^13) total           {gf13_bytes:>7,} B  ({gf13_bytes//1024} KB)")
    else:
        print(f"    GF(2^13)                 software only (no LUT, saves {GF13_LUT_TOTAL_BYTES//1024} KB)")
    print(f"    ─────────────────────────────────────────────────────────")
    print(f"    Total LUT footprint      {total_lut:>7,} B  ({total_lut/1024:.1f} KB)")
    print(f"    DTCM capacity            {dtcm_kb:>7,} B  ({STM32H753_DTCM_KB} KB)")
    if profile_name in ("dtcm", "dtcm_gf13lut"):
        fit = "YES" if total_lut < dtcm_kb else "NO — DTCM overflow!"
        print(f"    Fits in DTCM?            {fit}")
        print(f"    Remaining DTCM           {remaining:>7,} B  ({remaining/1024:.1f} KB)"
              f"  [stack + application data]")
    print()


def _fmt_freq(hz: float) -> str:
    if hz >= 1e9:
        return f"{hz/1e9:.3f} GHz"
    if hz >= 1e6:
        return f"{hz/1e6:.1f} MHz"
    return f"{hz/1e3:.1f} kHz"


# ---------------------------------------------------------------------------
#  Cross-profile comparison table
# ---------------------------------------------------------------------------

def compare_all_profiles(freq_hz: float, page_size: int, worst_case: bool):
    """Print a compact cross-profile comparison for *page_size* at *freq_hz*."""
    rows = {}
    for pname in PROFILES:
        apply_profile(pname)
        timings = build_all_timings(page_size, worst_case=worst_case)
        rows[pname] = {t.label: t for t in timings}

    all_labels = list(next(iter(rows.values())).keys())
    p_names    = list(PROFILES.keys())
    p_shorts   = [PROFILES[p]["short"] for p in p_names]

    freq_label = _fmt_freq(freq_hz)
    wc_tag     = "worst-case" if worst_case else "avg-iter"

    print()
    print(f"  Cross-profile comparison — {page_size} B page @ {freq_label} "
          f"({wc_tag} LDPC)")
    print()

    hdr_cols = "".join(f"  {s:>16}" for s in p_shorts)
    sep_cols = "  " + "─" * (26 + len(p_names) * 18)

    def _cell_enc(t):
        if t is None or isinstance(t, InfeasibleEntry):
            return f"  {'N/A':>14}   "
        return f"  {t.enc_time_us(freq_hz):>14.2f} µs"

    def _cell_dec(t):
        if t is None or isinstance(t, InfeasibleEntry):
            return f"  {'N/A':>14}   "
        return f"  {t.dec_time_us(freq_hz):>14.2f} µs"

    def _cell_thr(t):
        if t is None or isinstance(t, InfeasibleEntry):
            return f"  {'N/A':>14}   "
        return f"  {t.dec_throughput_mbps(freq_hz):>13.3f} M/s"

    # ── Encode latency ──────────────────────────────────────────────────────
    print(f"  {'Encode latency (µs)'}")    
    print(f"  {'Architecture':<26}" + hdr_cols)
    print(sep_cols)
    for lbl in all_labels:
        row_str = f"  {lbl:<26}"
        for pname in p_names:
            row_str += _cell_enc(rows[pname].get(lbl))
        # annotate infeasibility reason (same for every profile)
        any_t = next((rows[p].get(lbl) for p in p_names), None)
        if isinstance(any_t, InfeasibleEntry):
            row_str += f"  ← {any_t.reason}"
        print(row_str)
    print(sep_cols)
    print()

    # ── Decode latency ──────────────────────────────────────────────────────
    print(f"  {'Decode latency (µs)  [worst-case RS/BCH, ' + wc_tag + ' LDPC]'}")
    print(f"  {'Architecture':<26}" + hdr_cols)
    print(sep_cols)
    for lbl in all_labels:
        row_str = f"  {lbl:<26}"
        for pname in p_names:
            row_str += _cell_dec(rows[pname].get(lbl))
        any_t = next((rows[p].get(lbl) for p in p_names), None)
        if isinstance(any_t, InfeasibleEntry):
            row_str += f"  ← {any_t.reason}"
        print(row_str)
    print(sep_cols)
    print()

    # ── Decode throughput ───────────────────────────────────────────────────
    print(f"  {'Decode throughput (MB/s)'}")
    print(f"  {'Architecture':<26}" + hdr_cols)
    print(sep_cols)
    for lbl in all_labels:
        row_str = f"  {lbl:<26}"
        for pname in p_names:
            row_str += _cell_thr(rows[pname].get(lbl))
        any_t = next((rows[p].get(lbl) for p in p_names), None)
        if isinstance(any_t, InfeasibleEntry):
            row_str += f"  ← {any_t.reason}"
        print(row_str)
    print(sep_cols)
    print()


def plot_profile_comparison(freq_hz: float, page_size: int,
                             worst_case: bool, output_dir: str = "images"):
    """Side-by-side grouped bar chart: encode and decode latency per architecture,
    one bar group per ECC scheme, one bar colour per memory profile.
    Infeasible configurations are shown as hatched N/A stubs."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.patches as mpatches
    except ImportError:
        print("[warning] matplotlib not available — skipping plot")
        return

    mpl.rcParams.update({
        "font.family":    "serif",
        "font.size":      9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi":     150,
    })

    # Collect data
    profile_data = {}
    for pname in PROFILES:
        apply_profile(pname)
        timings = build_all_timings(page_size, worst_case=worst_case)
        profile_data[pname] = timings

    labels = [t.label for t in next(iter(profile_data.values()))]
    x      = np.arange(len(labels))
    n_prof = len(PROFILES)
    w      = 0.18
    offsets = np.linspace(-(n_prof - 1) / 2 * w, (n_prof - 1) / 2 * w, n_prof)

    # Paul Tol Bright palette (colourblind-safe)
    colours = ["#4477AA", "#228833", "#EE6677", "#AA3377"]

    # Feasible / infeasible index sets (same across all profiles)
    ref_timings    = next(iter(profile_data.values()))
    feasible_idx   = [i for i, t in enumerate(ref_timings) if not isinstance(t, InfeasibleEntry)]
    infeasible_idx = [i for i, t in enumerate(ref_timings) if isinstance(t, InfeasibleEntry)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (metric, ylabel, attr) in enumerate([
        ("Encode latency (µs)", "Latency (µs)", "enc_time_us"),
        ("Decode latency (µs)", "Latency (µs)", "dec_time_us"),
    ]):
        ax = axes[ax_idx]

        all_vals = []
        for j, (pname, col) in enumerate(zip(PROFILES, colours)):
            timings = profile_data[pname]
            # Feasible bars
            feas_vals = [getattr(timings[i], attr)(freq_hz) for i in feasible_idx]
            ax.bar(x[feasible_idx] + offsets[j], feas_vals, width=w,
                   label=PROFILES[pname]["short"], color=col, alpha=0.85)
            all_vals.extend(feas_vals)

        # Infeasible placeholder stubs (drawn once on top of the loop)
        if infeasible_idx:
            y_max  = max(all_vals) if all_vals else 1.0
            stub_h = y_max * 0.05
            for i in infeasible_idx:
                for offset in offsets:
                    ax.bar(x[i] + offset, stub_h, width=w,
                           color="none", edgecolor="#888888",
                           hatch="//", linewidth=0.8)
                ax.text(x[i], stub_h * 1.4, "N/A",
                        ha="center", va="bottom", fontsize=7,
                        color="#555555", style="italic")

        iter_tag = "worst-case" if worst_case else "avg-iter"
        ax.set_title(
            f"{metric} — STM32H753  {page_size} B page "
            f"@ {_fmt_freq(freq_hz)}\n"
            f"(software ECC, {iter_tag} LDPC)",
            pad=6,
        )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)

        handles, leg_labels = ax.get_legend_handles_labels()
        if infeasible_idx:
            handles.append(mpatches.Patch(
                facecolor="none", edgecolor="#888888",
                hatch="//", linewidth=0.8, label="N/A (infeasible)"
            ))
        ax.legend(handles=handles, title="Memory model", framealpha=0.9)
        y_top = max(all_vals) * 1.15 if all_vals else 1.0
        ax.set_ylim(0, y_top)
        ax.yaxis.grid(True, linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(
        output_dir,
        f"stm32h753_ecc_{page_size}B_{int(freq_hz/1e6)}MHz_profiles.png",
    )
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "ECC timing simulator for the STM32H753 Cortex-M7.\n"
            "Extends ecc_timing.py with target-specific memory hierarchy models."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--freq", type=float, default=STM32H753_FREQ_MAX_HZ,
        metavar="HZ",
        help=f"Core clock in Hz (default: {STM32H753_FREQ_MAX_HZ:.0f} = 480 MHz).",
    )
    parser.add_argument(
        "--memory-model",
        choices=list(PROFILES),
        default=DEFAULT_PROFILE,
        dest="memory_model",
        metavar="MODEL",
        help=(
            "Memory placement profile for LUTs. "
            f"Choices: {', '.join(PROFILES)}. "
            f"Default: {DEFAULT_PROFILE}. "
            "Use --compare to see all profiles at once."
        ),
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Print cross-profile comparison tables for all memory models.",
    )
    parser.add_argument(
        "--page", type=int, choices=PAGE_SIZES, default=None,
        metavar="BYTES",
        help=f"Single page size to analyse. Choices: {PAGE_SIZES}. Default: all.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate bar-chart PNGs in the images/ directory. "
             "With --compare, also produces a cross-profile comparison chart.",
    )
    parser.add_argument(
        "--worst-case", action="store_true",
        help=f"Use LDPC_MAX_ITER={LDPC_MAX_ITER} for LDPC "
             f"(default: LDPC_AVG_ITER={LDPC_AVG_ITER}).",
    )
    parser.add_argument(
        "--breakdown", action="store_true",
        help="Print per-operation cycle breakdown for every ECC architecture.",
    )
    parser.add_argument(
        "--footprint", action="store_true",
        help="Print LUT memory footprint and DTCM budget analysis.",
    )

    args   = parser.parse_args()
    sizes  = [args.page] if args.page else PAGE_SIZES
    freq   = args.freq
    wc     = args.worst_case

    # ------------------------------------------------------------------
    # If --compare: table across all profiles, then optional plot, done.
    # ------------------------------------------------------------------
    if args.compare:
        print()
        print("=" * 100)
        print(f"  STM32H753 ECC TIMING — MEMORY-MODEL COMPARISON  "
              f"|  Clock: {_fmt_freq(freq)}")
        print("=" * 100)
        for page_size in sizes:
            compare_all_profiles(freq, page_size, wc)

        if args.footprint:
            for pname in PROFILES:
                print(f"  [{PROFILES[pname]['short']}]")
                print_memory_footprint(pname)

        if args.plot:
            print("  Generating comparison charts …")
            for page_size in sizes:
                plot_profile_comparison(freq, page_size, wc)
        return

    # ------------------------------------------------------------------
    # Single-profile mode (default)
    # ------------------------------------------------------------------
    p = apply_profile(args.memory_model)
    print_mcu_header(freq, args.memory_model)

    if args.footprint:
        print_memory_footprint(args.memory_model)

    print("  NOTE: All cycle counts are order-of-magnitude estimates for a")
    print("  pure-software implementation. Actual performance depends on compiler")
    print("  optimisations (-O2 / -O3), loop unrolling, and LUT cache/TCM state.")
    print("  Adjust memory model with --memory-model {" +
          ", ".join(PROFILES) + "}.")
    print()

    all_timings = {}
    for page_size in sizes:
        timings = build_all_timings(page_size, worst_case=wc)
        all_timings[page_size] = timings
        print_timing_table(timings, freq, page_size)
        if args.breakdown:
            print_operation_breakdown(timings, freq)

    if args.plot:
        print("  Generating bar charts …")
        label_suffix = f"stm32h753_{args.memory_model}"
        # Temporarily redirect output dir naming
        plot_timing_bars(all_timings, freq, wc, output_dir="images")

    # Parameter summary
    print("  STM32H753 model parameters")
    print("  ─────────────────────────────────────────────────────────────────────")
    print(f"  Profile        : {args.memory_model}  ({p['short']})")
    print(f"  C_GF8_MUL      = {p['gf8_mul']:3d} cyc")
    print(f"  C_GF8_ADD      = {p['gf8_add']:3d} cyc")
    gf13_note = "LUT" if p["gf13_lut"] else "software CLMUL emulation"
    print(f"  C_GF13_MUL     = {p['gf13_mul']:3d} cyc  ({gf13_note})")
    print(f"  C_GF13_ADD     = {p['gf13_add']:3d} cyc")
    print(f"  C_LDPC_MSG     = {p['ldpc_msg']:3d} cyc")
    print(f"  LDPC iters     = "
          f"{LDPC_MAX_ITER if wc else LDPC_AVG_ITER:3d}  "
          f"({'worst-case' if wc else 'average'})")
    print(f"  MCU clock      : {_fmt_freq(freq)}")
    print(f"  DTCM           : {STM32H753_DTCM_KB} KB  |  "
          f"D-cache: {STM32H753_DCACHE_KB} KB  |  "
          f"ITCM: {STM32H753_ITCM_KB} KB")
    print()


if __name__ == "__main__":
    main()
