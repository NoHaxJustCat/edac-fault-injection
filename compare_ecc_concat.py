"""Monte Carlo ECC comparison for multiple NAND page sizes.

Runs the same analysis as compare_ecc.py but parameterised by page size.
Generates separate UBER curve + Pareto plots for each page size.
Uses Monte Carlo simulation with the validated LEO radiation environment
model (10:1 SEU-to-burst ratio, realistic burst-size distribution).

Page sizes analysed: 4224 B, 8640 B

Usage:
    python compare_ecc_pagesizes.py                  # all codes
    python compare_ecc_pagesizes.py --display rs      # 3 RS-only codes
    python compare_ecc_pagesizes.py --display bch     # 3 BCH-only codes
    python compare_ecc_pagesizes.py --display concat_rs   # 3 RS⊗RS-cross
    python compare_ecc_pagesizes.py --display concat_bch  # 3 BCH⊗RS-cross
    python compare_ecc_pagesizes.py --mc-iter 500     # set MC iterations
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import poisson as poisson_rv, binom as binom_rv
from math import comb, ceil

from utils import (
    _make_bch, compute_data_bytes,
    encode_with_rs, decode_with_rs,
    encode_with_bch, decode_with_bch,
    encode_rs_columns, decode_rs_columns,
)
from monte_carlo import (
    MCArch, run_mc_sweep, plot_convergence, plot_burst_histogram,
    print_sweep_summary, _POPCOUNT8,
)

# Academic plot style
mpl.rcParams.update({
    'font.family':        'serif',
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'legend.fontsize':    8,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'axes.linewidth':     0.8,
    'grid.linewidth':     0.5,
    'lines.linewidth':    1.6,
    'figure.dpi':         150,
})

# Paul Tol "Bright" palette — colorblind-safe, standard for publications
_tol = {
    'blue':   '#4477AA',
    'cyan':   '#66CCEE',
    'green':  '#228833',
    'yellow': '#CCBB44',
    'red':    '#EE6677',
    'purple': '#AA3377',
    'orange': '#EE7733',
}

# Output directory for all generated images
OUTPUT_DIR = os.path.join("images", "pagesizes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEU_RATE_SWEEP = np.logspace(-9, -3, 60)
SCRUB_INTERVAL = 3600   # seconds
UBER_REQ       = 1e-12
NUM_SECTORS    = 8

# Monte Carlo configuration
MC_ITERATIONS  = 1000
MC_SEED        = 42


# ══════════════════════════════════════════════════════════
#  ANALYTICAL UBER MODELS  (parameterised — kept for reference)
# ══════════════════════════════════════════════════════════

def _poisson_sf(t, lam):
    if lam < 1e-30:
        return 0.0
    return float(poisson_rv.sf(t, lam))


def uber_bch_only(seu_rate, bch_t, sector_bits, scrub=SCRUB_INTERVAL):
    lam = seu_rate * sector_bits * scrub
    return _poisson_sf(bch_t, lam)


def uber_rs_only(seu_rate, rs_t, nsym, sector_bytes, scrub=SCRUB_INTERVAL):
    lam_bit = seu_rate * scrub
    if lam_bit < 1e-30:
        return 0.0
    p_byte = 1.0 - np.exp(-8.0 * lam_bit)
    num_chunks = max(1, ceil(sector_bytes / 255))
    chunk_bytes = sector_bytes // num_chunks
    lam_chunk   = chunk_bytes * p_byte
    p_chunk_fail = _poisson_sf(rs_t, lam_chunk)
    return 1.0 - (1.0 - p_chunk_fail) ** num_chunks


def uber_rs_product(seu_rate, rs_row_t, rs_cross_t, total_rows, L,
                    rows_per_sector=1, scrub=SCRUB_INTERVAL):
    lam_bit = seu_rate * scrub
    if lam_bit < 1e-30:
        return 0.0
    p_byte = 1.0 - np.exp(-8.0 * lam_bit)
    p_col_fail = float(binom_rv.sf(rs_cross_t, total_rows, p_byte))
    if p_col_fail < 1e-30:
        return 0.0
    p_row_fail = float(binom_rv.sf(rs_row_t, L, p_col_fail))
    return 1.0 - (1.0 - p_row_fail) ** rows_per_sector


def uber_bch_product(seu_rate, bch_t, rs_cross_t, total_rows, L,
                     rows_per_sector=1, scrub=SCRUB_INTERVAL):
    lam_bit = seu_rate * scrub
    if lam_bit < 1e-30:
        return 0.0
    p_byte = 1.0 - np.exp(-8.0 * lam_bit)
    p_col_fail = float(binom_rv.sf(rs_cross_t, total_rows, p_byte))
    if p_col_fail < 1e-30:
        return 0.0
    mu_bits  = 8.0 * lam_bit / p_byte if p_byte > 1e-30 else 1.0
    lam_bits = L * p_col_fail * mu_bits
    p_row_fail = _poisson_sf(bch_t, lam_bits)
    return 1.0 - (1.0 - p_row_fail) ** rows_per_sector


def _gallager_a_threshold(dv, dc, max_de_iter=2000, bisect_steps=80):
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
    lam = seu_rate * page_bits * scrub
    return _poisson_sf(t_eff, lam)


# ══════════════════════════════════════════════════════════
#  MC-READY ARCHITECTURE BUILDERS  (parameterised by page size)
# ══════════════════════════════════════════════════════════

def _build_rs_arch(nsym, page_size, label, style):
    k = compute_data_bytes(page_size, NUM_SECTORS, rs_nsyms=[nsym])
    sector_len = page_size // NUM_SECTORS

    def encode(data):
        sectors = encode_with_rs(data, nsym, NUM_SECTORS)
        return np.concatenate(sectors)

    def decode(flat):
        sectors = [flat[i*sector_len:(i+1)*sector_len]
                   for i in range(NUM_SECTORS)]
        decoded, _ = decode_with_rs(sectors, nsym)
        return decoded

    return MCArch(label, k, page_size, NUM_SECTORS, encode, decode, style)


def _build_bch_arch(bch_ecc, page_size, label, style):
    sector_bytes = page_size // NUM_SECTORS
    bch_obj = _make_bch(bch_ecc)
    k = (sector_bytes - bch_obj.ecc_bytes) * NUM_SECTORS

    def encode(data):
        return encode_with_bch(data, bch_ecc, NUM_SECTORS)

    def decode(flat):
        sector_len = page_size // NUM_SECTORS
        sectors = [flat[i*sector_len:(i+1)*sector_len]
                   for i in range(NUM_SECTORS)]
        decoded, _ = decode_with_bch(sectors, bch_ecc)
        return decoded

    return MCArch(label, k, page_size, NUM_SECTORS, encode, decode, style)


def _build_rs_product_arch(nsym_row, nsym_cross, total_rows, L, page_size,
                           label, style):
    data_rows = total_rows - nsym_cross
    rows_per_sector = data_rows // NUM_SECTORS
    k = (L - nsym_row) * data_rows

    def encode(data):
        row_encoded = encode_with_rs(data, nsym_row, data_rows)
        matrix = np.stack(row_encoded)            # (data_rows, L)
        enc_matrix = encode_rs_columns(matrix, nsym_cross)  # (total_rows, L)
        return enc_matrix.flatten()

    def decode(flat):
        corrupted_matrix = flat.reshape(total_rows, L)
        decoded_matrix, _ = decode_rs_columns(
            corrupted_matrix, nsym_cross, data_rows)
        row_list = [decoded_matrix[i] for i in range(data_rows)]
        decoded_rows, _ = decode_with_rs(row_list, nsym_row)
        sectors = []
        for s in range(NUM_SECTORS):
            chunk = decoded_rows[s * rows_per_sector:(s + 1) * rows_per_sector]
            if any(r is None for r in chunk):
                sectors.append(None)
            else:
                sectors.append(np.concatenate(chunk))
        return sectors

    return MCArch(label, k, page_size, NUM_SECTORS, encode, decode, style)


def _build_bch_product_arch(bch_ecc, nsym_cross, total_rows, L, page_size,
                            label, style):
    bch_obj = _make_bch(bch_ecc)
    data_rows = total_rows - nsym_cross
    rows_per_sector = data_rows // NUM_SECTORS
    k = (L - bch_obj.ecc_bytes) * data_rows

    def encode(data):
        bch_flat = encode_with_bch(data, bch_ecc, data_rows)
        matrix = bch_flat.reshape(data_rows, L)
        enc_matrix = encode_rs_columns(matrix, nsym_cross)
        return enc_matrix.flatten()

    def decode(flat):
        corrupted_matrix = flat.reshape(total_rows, L)
        decoded_matrix, _ = decode_rs_columns(
            corrupted_matrix, nsym_cross, data_rows)
        bch_rows = [decoded_matrix[i] for i in range(data_rows)]
        decoded_rows, _ = decode_with_bch(bch_rows, bch_ecc)
        sectors = []
        for s in range(NUM_SECTORS):
            chunk = decoded_rows[s * rows_per_sector:(s + 1) * rows_per_sector]
            if any(r is None for r in chunk):
                sectors.append(None)
            else:
                sectors.append(np.concatenate(chunk))
        return sectors

    return MCArch(label, k, page_size, NUM_SECTORS, encode, decode, style)


def _build_ldpc_threshold_arch(t_eff, k_ldpc, page_size, label, style):
    sector_bytes = page_size // NUM_SECTORS

    def encode(data):
        out = np.zeros(page_size, dtype=np.uint8)
        out[:len(data)] = data
        return out

    def decode(flat):
        decoded = []
        sector_data_bytes = k_ldpc // NUM_SECTORS
        for i in range(NUM_SECTORS):
            start = i * sector_bytes
            end = start + sector_bytes
            sector = flat[start:end]
            n_errors = int(_POPCOUNT8[sector].sum())
            if n_errors <= t_eff:
                decoded.append(np.zeros(sector_data_bytes, dtype=np.uint8))
            else:
                decoded.append(None)
        return decoded

    return MCArch(label, k_ldpc, page_size, NUM_SECTORS, encode, decode, style)


# ══════════════════════════════════════════════════════════
#  FIND BEST PRODUCT-CODE GEOMETRY
# ══════════════════════════════════════════════════════════

def _best_product_geometry(page_size, nsym_cross=2):
    best = None
    for tr in range(10, 300):
        if page_size % tr != 0:
            continue
        L = page_size // tr
        data_rows = tr - nsym_cross
        if data_rows <= 0 or data_rows % NUM_SECTORS != 0:
            continue
        if L < 4:
            continue
        rps = data_rows // NUM_SECTORS
        product_rate = (data_rows / tr) * ((L - 2) / L)
        if best is None or product_rate > best[-1]:
            best = (tr, L, rps, product_rate)
    return best


# ══════════════════════════════════════════════════════════
#  PARAMETERISED ANALYSIS
# ══════════════════════════════════════════════════════════

def run_analysis(page_size, display=None, n_iter=MC_ITERATIONS, seed=MC_SEED,
                  parallel=False, num_workers=None, scrub_interval=SCRUB_INTERVAL):
    """Run full Monte Carlo ECC comparison for a given page size."""
    sector_bytes = page_size // NUM_SECTORS
    sector_bits  = sector_bytes * 8

    print(f"\n{'='*72}")
    print(f"  PAGE SIZE = {page_size} B   ({NUM_SECTORS} sectors x {sector_bytes} B)")
    if display:
        print(f"  DISPLAY   = {display}")
    print(f"  Monte Carlo: N={n_iter}, seed={seed}, scrub={scrub_interval}s")
    print(f"{'='*72}\n")

    configs = []

    # ── RS-only (3 configs) ───────────────────────────────
    if display in (None, 'rs', 'simple'):
        _rs_cols = [_tol['blue'], _tol['cyan'], '#88CCEE']
        for nsym, col in zip([8, 16, 24], _rs_cols):
            rs_t = nsym // 2
            try:
                k_rs = compute_data_bytes(page_size, NUM_SECTORS, rs_nsyms=[nsym])
            except ValueError:
                print(f"  [skip] RS nsym={nsym} incompatible with "
                      f"sector={sector_bytes}B (chunk regime)")
                continue
            label = f"RS nsym={nsym} (t={rs_t})"
            arch = _build_rs_arch(nsym, page_size, label,
                                  dict(linestyle='--', color=col, linewidth=1.8))
            configs.append(dict(
                label=label, model="rs_only",
                args=dict(rs_t=rs_t, nsym=nsym, sector_bytes=sector_bytes),
                k=k_rs, rate=k_rs / page_size,
                style=dict(linestyle='--', color=col, linewidth=1.8),
                arch=arch,
            ))

    # ── BCH-only (3 configs) ─────────────────────────────
    if display in (None, 'bch', 'simple'):
        _bch_cols = [_tol['green'], _tol['yellow'], '#BBCC33']
        for bch_ecc, col in zip([13, 22, 31], _bch_cols):
            bch_obj = _make_bch(bch_ecc)
            k_b = (sector_bytes - bch_obj.ecc_bytes) * NUM_SECTORS
            if k_b <= 0:
                continue
            label = f"BCH {bch_ecc}B (t={bch_obj.t})"
            arch = _build_bch_arch(bch_ecc, page_size, label,
                                   dict(linestyle=':', color=col, linewidth=2))
            configs.append(dict(
                label=label, model="bch_only",
                args=dict(bch_t=bch_obj.t, sector_bits=sector_bits),
                k=k_b, rate=k_b / page_size,
                style=dict(linestyle=':', color=col, linewidth=2),
                arch=arch,
            ))

    # ── RS ⊗ RS-cross (3 product-code configs) ───────────
    if display in (None, 'concat_rs') and display != 'simple':
        _rsp_cols = [_tol['red'], '#CC3311', '#EE3377']
        for (rs_nsym_row, nsym_cross), col in zip(
            [(2, 2), (4, 2), (2, 4)], _rsp_cols
        ):
            geom = _best_product_geometry(page_size, nsym_cross=nsym_cross)
            if geom is None:
                continue
            tr, L, rps, _ = geom
            rs_k = (L - rs_nsym_row) * (tr - nsym_cross)
            if rs_k <= 0:
                continue
            label = (f"RS(t={rs_nsym_row//2}) x RS-cross(t={nsym_cross//2})"
                     f"  [{tr}x{L}]")
            arch = _build_rs_product_arch(
                rs_nsym_row, nsym_cross, tr, L, page_size,
                label, dict(linewidth=1.8, color=col))
            configs.append(dict(
                label=label, model="rs_product",
                args=dict(rs_row_t=rs_nsym_row // 2,
                          rs_cross_t=nsym_cross // 2,
                          total_rows=tr, L=L, rows_per_sector=rps),
                k=rs_k, rate=rs_k / page_size,
                style=dict(linewidth=1.8, color=col),
                arch=arch,
            ))

    # ── BCH ⊗ RS-cross (3 product-code configs) ──────────
    if display in (None, 'concat_bch') and display != 'simple':
        _bp_cols = [_tol['purple'], '#882255', '#CC6677']
        for (bch_ecc_pc, nsym_cross), col in zip(
            [(2, 2), (4, 2), (2, 4)], _bp_cols
        ):
            geom = _best_product_geometry(page_size, nsym_cross=nsym_cross)
            if geom is None:
                continue
            tr, L, rps, _ = geom
            bch_obj_pc = _make_bch(bch_ecc_pc)
            bch_k_pc   = (L - bch_obj_pc.ecc_bytes) * (tr - nsym_cross)
            if bch_k_pc <= 0:
                continue
            label = (f"BCH(t={bch_obj_pc.t}) x RS-cross(t={nsym_cross//2})"
                     f"  [{tr}x{L}]")
            arch = _build_bch_product_arch(
                bch_ecc_pc, nsym_cross, tr, L, page_size,
                label, dict(linewidth=1.8, color=col))
            configs.append(dict(
                label=label, model="bch_product",
                args=dict(bch_t=bch_obj_pc.t, rs_cross_t=nsym_cross // 2,
                          total_rows=tr, L=L, rows_per_sector=rps),
                k=bch_k_pc, rate=bch_k_pc / page_size,
                style=dict(linewidth=1.8, color=col),
                arch=arch,
            ))

    # ── LDPC (analytical threshold MC: 3,30 → R=0.900) ───
    if display in (None, 'simple'):
        ldpc_dv, ldpc_dc = 3, 30
        ldpc_pstar = _gallager_a_threshold(ldpc_dv, ldpc_dc)
        page_bits  = page_size * 8
        ldpc_teff  = int(ldpc_pstar * page_bits)
        ldpc_k     = int(page_size * (1 - ldpc_dv / ldpc_dc))
        ldpc_k     = ldpc_k - (ldpc_k % NUM_SECTORS)   # must be divisible by num_sectors
        print(f"LDPC ({ldpc_dv},{ldpc_dc}): p*={ldpc_pstar:.5f}, "
              f"t_eff={ldpc_teff}, k={ldpc_k}, rate={ldpc_k/page_size:.3f}")
        label = f"LDPC d_v={ldpc_dv},d_c={ldpc_dc} (t_eff~{ldpc_teff})"
        arch = _build_ldpc_threshold_arch(
            ldpc_teff, ldpc_k, page_size, label,
            dict(linewidth=2.0, color=_tol['orange'], linestyle='-.'))
        configs.append(dict(
            label=label, model="ldpc",
            args=dict(t_eff=ldpc_teff, page_bits=page_bits),
            k=ldpc_k, rate=ldpc_k / page_size,
            style=dict(linewidth=2.0, color=_tol['orange'], linestyle='-.'),
            arch=arch,
        ))

    # ── Print config table ───────────────────────────────
    print(f"{'Config':<52s}  {'k':>5s}  {'rate':>6s}")
    print("-" * 72)
    for c in configs:
        print(f"{c['label']:<52s}  {c['k']:>5d}  {c['rate']:>.4f}")
    print()

    # ── Model dispatch (analytical reference) ────────────
    model_funcs = {
        "bch_only":    uber_bch_only,
        "rs_only":     uber_rs_only,
        "rs_product":  uber_rs_product,
        "bch_product": uber_bch_product,
        "ldpc":        uber_ldpc,
    }

    # ── Run Monte Carlo sweeps ───────────────────────────
    mc_results = {}
    results = {}          # backward-compatible UBER arrays

    for c in configs:
        print(f"\n-- MC sweep: {c['label']} --")
        sweep = run_mc_sweep(
            c['arch'], SEU_RATE_SWEEP, scrub_interval,
            n_iterations=n_iter, seed=seed, verbose=True,
            parallel=parallel, num_workers=num_workers)
        mc_results[c['label']] = sweep
        results[c['label']] = sweep.uber_values

        print_sweep_summary(sweep)

    # ── Compute analytical reference ─────────────────────
    analytical_results = {}
    for c in configs:
        fn   = model_funcs[c["model"]]
        uber = np.array([fn(r, **c["args"]) for r in SEU_RATE_SWEEP])
        analytical_results[c["label"]] = uber
        above = np.where(uber > UBER_REQ)[0]
        if len(above) > 0:
            crossing = SEU_RATE_SWEEP[above[0]]
            print(f"  [Analytical] {c['label']:<52s}  UBER > 1e-12 at SEU ~ {crossing:.2e}")
        else:
            print(f"  [Analytical] {c['label']:<52s}  UBER < 1e-12 across entire sweep")

    # ══════════════════════════════════════════════════════
    #  PLOT 1 — UBER curves
    # ══════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(6, 3))
    for c in configs:
        uber = results[c["label"]]
        mask = uber > 0
        kw   = dict(c.get("style", {}))
        if np.any(mask):
            ax.loglog(SEU_RATE_SWEEP[mask], uber[mask],
                      marker='.', markersize=3,
                      label=f"{c['label']}  (page={page_size}B, rate={c['rate']:.3f})", **kw)
        # Analytical overlay
        if c["label"] in analytical_results:
            uber_a = analytical_results[c["label"]]
            mask_a = uber_a > 0
            if np.any(mask_a):
                ax.loglog(SEU_RATE_SWEEP[mask_a], uber_a[mask_a],
                          linewidth=0.8, linestyle=':', color=kw.get('color', 'gray'),
                          alpha=0.5)

    ax.axhline(UBER_REQ, color='red', linestyle='--', linewidth=1.2, alpha=0.8,
               label='UBER requirement (1e-12)')
    ax.set_xlabel("SEU rate  [events / bit / s]")
    ax.set_ylabel("UBER  (Monte Carlo)")
    ax.set_title(f"ECC Comparison  (page={page_size} B, {NUM_SECTORS} sectors,"
                 f" scrub={scrub_interval} s, N={n_iter}")
    ax.set_ylim(bottom=1e-30)
    ax.legend(loc='upper left', fontsize=7, ncol=1,
              bbox_to_anchor=(0.01, 0.99), borderaxespad=0)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fname1 = f"uber_{page_size}.png"
    if display:
        fname1 = f"uber_{page_size}_{display}.png"
    save1 = os.path.join(OUTPUT_DIR, fname1)
    plt.savefig(save1, dpi=150)
    print(f"\nPlot 1 saved to {save1}")

    # ══════════════════════════════════════════════════════
    #  PLOT 2 — Pareto (max tolerable SEU rate vs code rate)
    # ══════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    _markers = {"bch_only": "s", "rs_only": "^",
                "rs_product": "P", "bch_product": "*",
                "ldpc": "D"}

    for c in configs:
        uber = results[c["label"]]
        above = np.where(uber > UBER_REQ)[0]
        if len(above) > 0:
            idx = above[0]
            if idx > 0:
                x0, x1 = np.log10(SEU_RATE_SWEEP[idx-1]), np.log10(SEU_RATE_SWEEP[idx])
                y0, y1 = np.log10(max(uber[idx-1], 1e-50)), np.log10(uber[idx])
                y_req  = np.log10(UBER_REQ)
                frac   = (y_req - y0) / (y1 - y0) if y1 != y0 else 0.5
                crossing = 10 ** (x0 + frac * (x1 - x0))
            else:
                crossing = SEU_RATE_SWEEP[0]
        else:
            crossing = SEU_RATE_SWEEP[-1]

        ax2.semilogy(c["rate"], crossing,
                     marker=_markers.get(c["model"], "o"),
                     color=c["style"]["color"],
                     markersize=10, zorder=5)
        offset = (8, 4) if c["model"] != "rs_only" else (8, -12)
        ax2.annotate(c["label"].split("(")[0].strip(),
                     xy=(c["rate"], crossing), fontsize=6.5,
                     textcoords="offset points", xytext=offset,
                     arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    ax2.set_xlabel("Code rate  (k / page_size)")
    ax2.set_ylabel("Max tolerable SEU rate for UBER < 1e-12  [events/bit/s]")
    ax2.set_title(f"ECC Pareto  (page={page_size} B, Monte Carlo)")
    ax2.grid(True, which="both", alpha=0.3)

    from matplotlib.lines import Line2D
    _legend_defs = [
        ('rs_only',     '^', _tol['blue'],   'RS'),
        ('bch_only',    's', _tol['green'],  'BCH'),
        ('rs_product',  'P', _tol['red'],    'RS \u2297 RS-cross (product)'),
        ('bch_product', '*', _tol['purple'], 'BCH \u2297 RS-cross (product)'),
        ('ldpc',        'D', _tol['orange'], 'LDPC'),
    ]
    present_models = {c['model'] for c in configs}
    legend_elements = [
        Line2D([0], [0], marker=mk, color='w', markerfacecolor=col,
               markersize=9, label=lbl)
        for model, mk, col, lbl in _legend_defs
        if model in present_models
    ]
    ax2.legend(handles=legend_elements, loc='lower left', fontsize=8)

    plt.tight_layout()
    fname2 = f"pareto_{page_size}.png"
    if display:
        fname2 = f"pareto_{page_size}_{display}.png"
    save2 = os.path.join(OUTPUT_DIR, fname2)
    plt.savefig(save2, dpi=150)
    print(f"Plot 2 saved to {save2}")

    # ══════════════════════════════════════════════════════
    #  PLOT 3 — ECC overwhelmed fraction
    # ══════════════════════════════════════════════════════
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    for c in configs:
        sweep = mc_results[c['label']]
        owh = sweep.ecc_overwhelmed
        mask = owh > 0
        kw = dict(c.get("style", {}))
        if np.any(mask):
            ax3.semilogx(SEU_RATE_SWEEP[mask], owh[mask] * 100,
                         marker='.', markersize=3,
                         label=f"{c['label']}", **kw)
    ax3.set_xlabel("SEU rate  [events / bit / s]")
    ax3.set_ylabel("Windows with uncorrectable errors  [%]")
    ax3.set_title(f"ECC Overwhelmed Fraction  (page={page_size} B)")
    ax3.legend(loc='upper left', fontsize=7)
    ax3.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    save3 = os.path.join(OUTPUT_DIR, f"overwhelmed_{page_size}.png")
    plt.savefig(save3, dpi=150)
    print(f"Plot 3 saved to {save3}")

    # ── Convergence + burst histogram (first config) ─────
    first_label = configs[0]['label']
    plot_convergence(mc_results[first_label],
                     save_path=os.path.join(OUTPUT_DIR, f"convergence_{page_size}.png"))
    plot_burst_histogram(mc_results[first_label],
                         save_path=os.path.join(OUTPUT_DIR, f"burst_hist_{page_size}.png"))

    return configs, results


# ══════════════════════════════════════════════════════════
#  MAIN — run for both page sizes
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monte Carlo ECC comparison for multiple NAND page sizes.")
    parser.add_argument(
        "--display", "-d",
        choices=["rs", "bch", "concat_rs", "concat_bch", "simple"],
        default=None,
        help="Filter display: rs/bch/concat_rs/concat_bch (3 configs each), "
             "simple (RS+BCH+LDPC, no concat), or None (all). Default: all.")
    parser.add_argument("--mc-iter", "-n", type=int, default=MC_ITERATIONS,
                        help=f"MC iterations per SEU-rate point (default: {MC_ITERATIONS})")
    parser.add_argument("--seed", "-s", type=int, default=MC_SEED,
                        help=f"RNG seed for reproducibility (default: {MC_SEED})")
    parser.add_argument("--parallel", action="store_true",
                        help="Use multiprocessing (recommended for N >= 100 or many cores)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of worker processes (None = all CPU cores)")
    parser.add_argument("--scrub-interval", type=float, default=SCRUB_INTERVAL,
                        help=f"Scrub interval in seconds (default: {SCRUB_INTERVAL})")
    args = parser.parse_args()

    for ps in [4224, 8640]:
        run_analysis(ps, display=args.display,
                     n_iter=args.mc_iter, seed=args.seed,
                     parallel=args.parallel, num_workers=args.num_workers,
                     scrub_interval=args.scrub_interval)
    plt.show()
