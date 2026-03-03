"""Shared encode / decode / corrupt routines for concatenated ECC simulations."""

import numpy as np
from reedsolo import RSCodec, ReedSolomonError
import bchlib


# ── Codec caching (avoid re-creating expensive GF look-up tables) ─────
_rs_cache = {}


def _get_rsc(nsym):
    """Return a cached RSCodec(nsym, c_exp=8) instance."""
    try:
        return _rs_cache[nsym]
    except KeyError:
        rsc = RSCodec(nsym, c_exp=8)
        _rs_cache[nsym] = rsc
        return rsc


# ── Column-wise (cross-sector) RS helpers ────────────────

def encode_rs_columns(matrix, nsym):
    """RS-encode each column of a 2D uint8 array (cross-sector encoding).

    Each column of *matrix* is independently RS-encoded, appending *nsym*
    parity symbols.  This effectively applies RS across sectors for each
    byte position, creating a product-code-like interleaved structure.

    Args:
        matrix : numpy array of shape (num_data_rows, num_cols), dtype uint8
        nsym   : RS parity symbols per column

    Returns:
        numpy array of shape (num_data_rows + nsym, num_cols), dtype uint8
    """
    rsc = _get_rsc(nsym)
    num_rows, num_cols = matrix.shape
    encoded = np.empty((num_rows + nsym, num_cols), dtype=np.uint8)
    for j in range(num_cols):
        enc = rsc.encode(matrix[:, j].tobytes())
        encoded[:, j] = np.frombuffer(enc, dtype=np.uint8)
    return encoded


def decode_rs_columns(matrix, nsym, num_data_rows):
    """RS-decode each column of a 2D uint8 array (cross-sector decoding).

    Reverses :func:`encode_rs_columns`.  Columns that cannot be decoded
    have their data portion passed through uncorrected so the downstream
    per-sector RS decoder can still attempt recovery.

    Args:
        matrix         : numpy array of shape (num_data_rows + nsym, num_cols)
        nsym           : RS parity symbols per column
        num_data_rows  : number of data rows (= NUM_SECTORS)

    Returns:
        (decoded, num_column_failures)
        decoded             : numpy array (num_data_rows, num_cols), dtype uint8
        num_column_failures : int, columns that failed RS decoding
    """
    rsc = _get_rsc(nsym)
    _, num_cols = matrix.shape
    decoded = np.empty((num_data_rows, num_cols), dtype=np.uint8)
    failures = 0
    for j in range(num_cols):
        try:
            data, _, _ = rsc.decode(matrix[:, j].tobytes())
            decoded[:, j] = np.frombuffer(data, dtype=np.uint8)
        except Exception:
            decoded[:, j] = matrix[:num_data_rows, j]
            failures += 1
    return decoded, failures


# ── BCH configuration ────────────────────────────────────
# Maps BCH ECC byte count → BCH_BITS parameter for bchlib.
# BCH codes over GF(2^13) with primitive polynomial 8219.
BCH_BITS_FROM_ECC = {
    2: 1,  4: 2,  5: 3,  7: 4,  9: 5,  10: 6,  12: 7,
    13: 8, 15: 9, 17: 10, 18: 11, 20: 12, 22: 13,
    23: 14, 25: 15, 26: 16, 28: 17, 30: 18, 31: 19,
}
BCH_PRIM_POLY = 8219


def _make_bch(ecc_bytes):
    """Create a bchlib.BCH encoder/decoder from the desired ECC byte count."""
    bits = BCH_BITS_FROM_ECC.get(ecc_bytes)
    if bits is None:
        raise ValueError(
            f"No BCH_BITS mapping for {ecc_bytes} ECC bytes. "
            f"Valid values: {sorted(BCH_BITS_FROM_ECC)}"
        )
    return bchlib.BCH(bits, prim_poly=BCH_PRIM_POLY)


_bch_cache = {}


def _get_bch_cached(ecc_bytes):
    """Return a cached bchlib.BCH instance for the given ECC byte count."""
    try:
        return _bch_cache[ecc_bytes]
    except KeyError:
        bch = _make_bch(ecc_bytes)
        _bch_cache[ecc_bytes] = bch
        return bch


# ── Data-size computation ────────────────────────────────

def compute_data_bytes(page_size, num_sectors, rs_nsyms, bch_ecc=0):
    """Compute raw data bytes (k) that fit in a page after all ECC layers.

    Works backwards from the final encoded sector size, peeling off each RS
    layer's overhead (accounting for reedsolo's GF(2^8) chunk splitting) and
    then the BCH overhead.

    Args:
        page_size   : total encoded page size in bytes
        num_sectors : number of equal-sized sectors the page is split into
        rs_nsyms    : list of RS nsym values **in encoding order**
        bch_ecc     : BCH ECC bytes per sector (applied before all RS layers; 0 = no BCH)

    Returns:
        int – number of raw data bytes (k)
    """
    import math
    assert page_size % num_sectors == 0, "page_size must be divisible by num_sectors"
    sector = page_size // num_sectors

    # Peel off RS layers in reverse (last encoded = first to undo)
    for nsym in reversed(rs_nsyms):
        max_chunk_data = 255 - nsym          # max data bytes per RS chunk
        # Find the number of chunks n such that:
        #   data + n * nsym == sector   AND   ceil(data / max_chunk_data) == n
        found = False
        for n_chunks in range(1, 100):
            data = sector - n_chunks * nsym
            if data <= 0:
                break
            if math.ceil(data / max_chunk_data) == n_chunks:
                sector = data
                found = True
                break
        if not found:
            raise ValueError(
                f"No valid RS chunk regime for encoded sector={sector}, nsym={nsym}. "
                f"The page_size/num_sectors combination may not be achievable."
            )

    # Peel off BCH layer
    sector -= bch_ecc
    if sector <= 0:
        raise ValueError(f"No room for data (sector={sector} after removing BCH ECC)")
    return sector * num_sectors


# ── Encode ───────────────────────────────────────────────

def encode_with_rs(data, nsym, num_sectors):
    """RS-encode: split *data* into *num_sectors* sectors, add *nsym* parity
    symbols per RS chunk (GF(2^8)).  Each sector may span 1 or 2 chunks
    depending on its length relative to 255 − nsym.

    Returns:
        list of *num_sectors* numpy uint8 arrays (encoded sectors)
    """
    rsc = _get_rsc(nsym)
    return [np.frombuffer(rsc.encode(s), dtype=np.uint8)
            for s in np.split(data, num_sectors)]


def encode_with_bch(data, ecc_bytes, num_sectors):
    """BCH-encode: split *data* into *num_sectors* sectors, add BCH parity
    (*ecc_bytes* per sector) to each.

    Returns:
        flat numpy uint8 array (all encoded sectors concatenated)
    """
    bch = _get_bch_cached(ecc_bytes)
    parts = []
    for sector in np.split(data, num_sectors):
        raw = bytearray(sector)
        ecc = bch.encode(raw)
        parts.append(np.frombuffer(bytes(raw) + bytes(ecc), dtype=np.uint8))
    return np.concatenate(parts)


# ── Corruption ───────────────────────────────────────────

def corrupt_page(sectors, seu_rate, scrub_interval):
    """Simulate SEU bit flips on each sector of a NAND page.

    Args:
        sectors        : list of numpy arrays (one per sector)
        seu_rate       : flip probability per bit per second
        scrub_interval : seconds between scrubbing events

    Returns:
        list of corrupted numpy arrays (same structure as input)
    """
    corrupted = []
    for sector in sectors:
        n_bits  = len(sector) * 8
        lam     = seu_rate * n_bits * scrub_interval
        n_flips = np.random.poisson(lam)

        out = sector.copy()
        if n_flips > 0:
            positions = np.random.choice(n_bits, size=min(n_flips, n_bits), replace=False)
            for pos in positions:
                byte_idx, bit_idx = divmod(pos, 8)
                out[byte_idx] ^= (1 << bit_idx)
        corrupted.append(out)
    return corrupted


def corrupt_page_burst(flat_page, burst_bytes, rng=None):
    """Simulate a contiguous burst error on a flat page.

    A single burst of *burst_bytes* corrupted bytes is placed at a
    uniformly random position within the page.  Every byte inside the
    burst window is XOR-ed with a random nonzero mask (≥ 1 bit flipped).

    Args:
        flat_page   : 1-D numpy uint8 array (the full encoded page)
        burst_bytes : number of contiguous bytes corrupted by the burst
        rng         : numpy random Generator (optional)

    Returns:
        corrupted copy of *flat_page* (same shape)
    """
    if rng is None:
        rng = np.random.default_rng()
    out = flat_page.copy()
    n = len(out)
    if burst_bytes <= 0 or n == 0:
        return out
    burst_bytes = min(burst_bytes, n)
    # Random start position (burst wraps if necessary, but typically fits)
    start = rng.integers(0, n - burst_bytes + 1)
    masks = rng.integers(1, 256, size=burst_bytes, dtype=np.uint8)
    out[start:start + burst_bytes] ^= masks
    return out


def corrupt_page_burst_bits(flat_page, burst_bits, rng=None):
    """Simulate a contiguous burst of *bit* errors on a flat page.

    A single burst of *burst_bits* consecutive bit positions is placed at
    a uniformly random position within the page.  Every bit inside the
    burst window is flipped.

    Args:
        flat_page  : 1-D numpy uint8 array (the full encoded page)
        burst_bits : number of contiguous bits to flip
        rng        : numpy random Generator (optional)

    Returns:
        corrupted copy of *flat_page* (same shape)
    """
    if rng is None:
        rng = np.random.default_rng()
    out = flat_page.copy()
    total_bits = len(out) * 8
    if burst_bits <= 0 or total_bits == 0:
        return out
    burst_bits = min(burst_bits, total_bits)
    start_bit = int(rng.integers(0, total_bits - burst_bits + 1))
    for i in range(burst_bits):
        bit_pos = start_bit + i
        byte_idx, bit_idx = divmod(bit_pos, 8)
        out[byte_idx] ^= (1 << bit_idx)
    return out


# ── Decode ───────────────────────────────────────────────

def decode_with_rs(sectors, nsym):
    """RS-decode each sector.  ``None`` entries (already-failed sectors) are
    propagated as decode failures.

    Args:
        sectors : list of numpy arrays **or** None values
        nsym    : RS parity symbols per chunk (must match encoding)

    Returns:
        (decoded_sectors, num_failures)
        Each decoded sector is a numpy uint8 array, or None on failure.
    """
    rsc = _get_rsc(nsym)
    decoded  = []
    failures = 0
    for sector in sectors:
        if sector is None:
            decoded.append(None)
            failures += 1
            continue
        try:
            data, _, _ = rsc.decode(sector.tobytes())
            decoded.append(np.frombuffer(data, dtype=np.uint8))
        except Exception:
            decoded.append(None)
            failures += 1
    return decoded, failures


def decode_with_bch(sectors, ecc_bytes):
    """BCH-decode each sector.  ``None`` entries are propagated as failures.

    Args:
        sectors   : list of numpy arrays **or** None values
        ecc_bytes : BCH ECC bytes per sector (must match encoding)

    Returns:
        (decoded_sectors, num_failures)
    """
    bch = _get_bch_cached(ecc_bytes)

    # Determine data length from the first valid sector
    valid = next((s for s in sectors if s is not None), None)
    if valid is None:
        return [None] * len(sectors), len(sectors)
    data_len = len(valid) - bch.ecc_bytes

    decoded  = []
    failures = 0
    for sector in sectors:
        if sector is None:
            decoded.append(None)
            failures += 1
            continue
        data = bytearray(sector[:data_len])
        ecc  = bytearray(sector[data_len:])
        try:
            nerr = bch.decode(data, ecc)
        except (ValueError, Exception):
            # bchlib may raise ValueError ("invalid parameters") for
            # heavily corrupted data — treat as uncorrectable.
            nerr = -1
        if nerr < 0:
            decoded.append(None)
            failures += 1
        else:
            bch.correct(data, ecc)
            decoded.append(np.frombuffer(bytes(data), dtype=np.uint8))
    return decoded, failures


def corrupt_page_leo(sectors, seu_rate, scrub_interval, rng=None):
    """Simulate LEO radiation-model errors on each sector of a NAND page.

    Drop-in replacement for :func:`corrupt_page` that uses the validated
    LEO radiation environment model (10:1 SEU-to-burst ratio, validated
    burst-size distribution).

    Args:
        sectors        : list of numpy arrays (one per sector)
        seu_rate       : radiation event rate [events/bit/s]
        scrub_interval : seconds between scrubbing events
        rng            : numpy random Generator (optional)

    Returns:
        list of corrupted numpy arrays (same structure as input)
    """
    from monte_carlo import inject_errors_leo
    if rng is None:
        rng = np.random.default_rng()
    corrupted = []
    for sector in sectors:
        out, _ = inject_errors_leo(sector, seu_rate, scrub_interval, rng,
                                   page_size_bits=len(sector) * 8)
        corrupted.append(out)
    return corrupted


def decode_with_bch_erasure(sectors, ecc_bytes):
    """BCH-decode with erasure flagging for concatenated RS recovery.

    Correctable errors (≤ t) are fixed normally.  When BCH detects
    uncorrectable errors (t+1 … 2t bit flips), the corrupted data
    (stripped of BCH parity) is passed through with an erasure flag
    so that a downstream RS decoder can attempt recovery.

    Args:
        sectors   : list of numpy arrays (one per sector)
        ecc_bytes : BCH ECC bytes per sector (must match encoding)

    Returns:
        (decoded_sectors, bch_corrected, erasure_flags)
        decoded_sectors : list of numpy uint8 arrays (corrected **or** corrupted)
        bch_corrected   : count of sectors where BCH fixed ≥ 1 error
        erasure_flags   : list[bool], True → BCH could not correct (erasure)
    """
    bch = _get_bch_cached(ecc_bytes)

    valid = next((s for s in sectors if s is not None), None)
    if valid is None:
        return [None] * len(sectors), 0, [True] * len(sectors)
    data_len = len(valid) - bch.ecc_bytes

    decoded       = []
    bch_corrected = 0
    erasure_flags = []

    for sector in sectors:
        if sector is None:
            decoded.append(None)
            erasure_flags.append(True)
            continue

        data = bytearray(sector[:data_len])
        ecc  = bytearray(sector[data_len:])
        nerr = bch.decode(data, ecc)

        if nerr < 0:
            # Uncorrectable → erasure: pass corrupted data forward
            decoded.append(np.frombuffer(bytes(data), dtype=np.uint8))
            erasure_flags.append(True)
        else:
            if nerr > 0:
                bch.correct(data, ecc)
                bch_corrected += 1
            decoded.append(np.frombuffer(bytes(data), dtype=np.uint8))
            erasure_flags.append(False)

    return decoded, bch_corrected, erasure_flags