"""`kq_xtx`: Q4_K / Q5_K with a weighted scale-and-minimum search and, where the pinned calibration has the input's
full second moment, GPTQ error feedback.

The built-in `kq_rtn` sets each 32-value sub-block's scale and minimum from the sub-block's own range and rounds every
weight on its own. What the model sees is a Linear's output error x·(w - q) over its real inputs x; for one row that is
(w - q) H (w - q)ᵀ with H = mean x xᵀ. This encoder lowers it in two ways:

* **Weighted grid search.** Each sub-block's (scale, minimum) is chosen among candidate scales around the range fit by
  least squares weighted by the input energy of its 32 columns (diag H), as llama.cpp's make_qkx2_quants does with an
  importance matrix; the 6-bit sub-scales are then fitted to the super-block's f16 scales, and the 4/5-bit values
  re-rounded on the final grid.
* **GPTQ.** When the calibration holds the full H of the Linear's input (`attn_input.xtx` for attention q/k/v and the
  recurrent qkv/z, `ffn_input.xtx` for the shared expert's gate/up, `ffn_down_shexp.xtx` for its down), columns are
  rounded left to right and each column's error is pushed onto the columns not yet rounded through the inverse
  Hessian. A super-block's grid is fitted when GPTQ reaches it, on the weights as earlier feedback left them.

Routed experts use their per-expert diagonal (`exps_input.sumsq` / `exps_down.sumsq` over `exps.count`, as llama.cpp's
imatrix) for the search only; units without statistics (attention o, recurrent out, embeddings, lm_head) get the
search with uniform weights. Everything is float64 NumPy on the CPU; the only BLAS calls are a Cholesky factorization
and the per-block error propagation, both run on one machine for a build and its audit replay.
"""

from __future__ import annotations

import numpy as np

from bittrellis import kquant
from bittrellis.hpc02 import Encoder, register

DAMP = 0.01           # relative diagonal dampening of H, as in GPTQ
CHUNK_ROWS = 2048     # rows per pass (independent given H)
STEPS = np.arange(-4, 5) * 0.1   # candidate scale offsets around the range fit, in units of 1/nmax of the range
NMAX = {"Q4_K": 15, "Q5_K": 31}


# ------------------------------------------------------------------ grid fitting


def _fit_sub(x: np.ndarray, w: np.ndarray, nmax: int):
    """Weighted (scale, min) per sub-block. x, w: [n, 32] -> scale, mn [n] (value = scale*q - mn, q in 0..nmax)."""
    lo = np.minimum(x.min(-1), 0.0)
    hi = x.max(-1)
    rng = hi - lo
    best_err = np.full(len(x), np.inf)
    best_s = np.where(rng > 0, rng / nmax, 0.0)
    best_m = -lo
    sw = w.sum(-1)
    for st in STEPS:
        with np.errstate(divide="ignore", invalid="ignore"):
            iscale = np.where(rng > 0, (nmax + st) / rng, 0.0)
        q = np.clip(np.rint((x - lo[:, None]) * iscale[:, None]), 0, nmax)
        # weighted least squares of x ≈ s*q - m over (s, m)
        sq, sq2 = (w * q).sum(-1), (w * q * q).sum(-1)
        sx, sqx = (w * x).sum(-1), (w * q * x).sum(-1)
        det = sw * sq2 - sq * sq
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(det > 0, (sw * sqx - sq * sx) / det, 0.0)
        m = (s * sq - sx) / np.where(sw > 0, sw, 1.0)
        bad = (s <= 0) | (m < 0)
        s = np.where(bad, best_s, s)
        m = np.where(bad, np.maximum(best_m, 0.0), np.maximum(m, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            q2 = np.where(s[:, None] > 0, np.clip(np.rint((x + m[:, None]) / s[:, None]), 0, nmax), 0)
        err = (w * (s[:, None] * q2 - m[:, None] - x) ** 2).sum(-1)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_s = np.where(better, s, best_s)
        best_m = np.where(better, m, best_m)
    return best_s, best_m


def _grid(x: np.ndarray, w: np.ndarray, nmax: int):
    """Super-block grids for x, w [n, 8, 32]: d, dmin (f16) and 6-bit sc, m [n, 8] (as kquant packs them)."""
    n = len(x)
    s, mn = _fit_sub(x.reshape(-1, 32), w.reshape(-1, 32), nmax)
    s, mn = s.reshape(n, 8), mn.reshape(n, 8)
    d = kquant._f16(s.max(-1) / 63.0)
    dmin = kquant._f16(mn.max(-1) / 63.0)
    df, dmf = d.astype(np.float64), dmin.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(df[:, None] > 0, np.clip(np.rint(s / df[:, None]), 0, 63), 0)
        m = np.where(dmf[:, None] > 0, np.clip(np.rint(mn / dmf[:, None]), 0, 63), 0)
    return d, dmin, sc, m


def _round(x: np.ndarray, step: np.ndarray, off: np.ndarray, nmax: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(step > 0, np.rint((x + off) / step), 0)
    return np.clip(q, 0, nmax)


def _pack(fmt: str, d, dmin, sc, m, q) -> bytes:
    """Same block layout as kquant.quantize_q4k / quantize_q5k. q: [n, 8, 32] uint8."""
    q = q.astype(np.uint8)
    n = len(q)
    head = [d.view(np.uint8).reshape(-1, 2), dmin.view(np.uint8).reshape(-1, 2), kquant._pack_scales_q4k(sc, m)]
    if fmt == "Q4_K":
        qs = (q[:, 0::2, :] | (q[:, 1::2, :] << 4)).reshape(n, 128)
        return np.concatenate(head + [qs], axis=1).tobytes()
    lo, hi = q & 0x0F, q >> 4
    qs = (lo[:, 0::2, :] | (lo[:, 1::2, :] << 4)).reshape(n, 128)
    qh = np.zeros((n, 32), np.uint8)
    for j in range(8):
        qh |= (hi[:, j, :] & 1) << j
    return np.concatenate(head + [qh, qs], axis=1).tobytes()


def quantize_weighted(x: np.ndarray, wdiag: np.ndarray, fmt: str) -> bytes:
    """No feedback: weighted grid search per super-block. x [rows, cols]; wdiag [rows, cols] or [cols] (>= 0)."""
    nmax = NMAX[fmt]
    rows, cols = x.shape
    xb = np.asarray(x, np.float64).reshape(-1, 8, 32)
    wb = np.broadcast_to(np.asarray(wdiag, np.float64), (rows, cols)).reshape(-1, 8, 32)
    d, dmin, sc, m = _grid(xb, wb, nmax)
    step = (d.astype(np.float64)[:, None] * sc)[..., None]
    off = (dmin.astype(np.float64)[:, None] * m)[..., None]
    q = _round(xb, step, off, nmax)
    return _pack(fmt, d, dmin, sc, m, q)


def inverse_hessian_factor(h: np.ndarray, damp: float = DAMP) -> np.ndarray:
    """Upper Cholesky factor U of (H + λI)⁻¹ = UᵀU, float64; dead inputs (zero diagonal) set to 1."""
    h = np.array(h, dtype=np.float64)
    dead = np.diag(h) <= 0
    h[dead, dead] = 1.0
    h[np.diag_indices_from(h)] += damp * float(np.mean(np.diag(h)))
    lo = np.linalg.cholesky(h)
    inv = np.linalg.inv(lo)
    return np.linalg.cholesky(inv.T @ inv).T


def quantize_gptq(x: np.ndarray, h: np.ndarray, fmt: str, u: np.ndarray | None = None) -> bytes:
    """GPTQ over rows x [rows, cols] with input second moment h [cols, cols]; grids fitted per 256-column super-block
    on the error-fed weights, weighted by diag(h)."""
    nmax = NMAX[fmt]
    u = inverse_hessian_factor(h) if u is None else u
    hd = np.diag(np.asarray(h, np.float64)).copy()
    out = []
    for r0 in range(0, x.shape[0], CHUNK_ROWS):
        w = np.array(x[r0:r0 + CHUNK_ROWS], dtype=np.float64)
        rows, cols = w.shape
        nsb = cols // 256
        q_all = np.empty((rows, cols), np.uint8)
        grids = []
        for b in range(nsb):
            c0, c1 = 256 * b, 256 * (b + 1)
            blk = w[:, c0:c1].reshape(rows, 8, 32)
            d, dmin, sc, m = _grid(blk, np.broadcast_to(hd[c0:c1].reshape(1, 8, 32), blk.shape), nmax)
            grids.append((d, dmin, sc, m))
            step = d.astype(np.float64)[:, None] * sc                    # [rows, 8]
            off = dmin.astype(np.float64)[:, None] * m
            err = np.empty((rows, 256))
            u1 = u[c0:c1, c0:c1]
            for j in range(256):
                col = w[:, c0 + j]
                sj, oj = step[:, j // 32], off[:, j // 32]
                q = _round(col, sj, oj, nmax)
                q_all[:, c0 + j] = q
                e = (col - (sj * q - oj)) / u1[j, j]
                err[:, j] = e
                if j < 255:
                    w[:, c0 + j + 1:c1] -= e[:, None] * u1[j, j + 1:]
            if c1 < cols:
                w[:, c1:] -= err @ u[c0:c1, c1:]
        d = np.stack([g[0] for g in grids], 1).reshape(-1)
        dmin = np.stack([g[1] for g in grids], 1).reshape(-1)
        sc = np.stack([g[2] for g in grids], 1).reshape(-1, 8)
        m = np.stack([g[3] for g in grids], 1).reshape(-1, 8)
        out.append(_pack(fmt, d, dmin, sc, m, q_all.reshape(-1, 8, 32)))
    return b"".join(out)


# ------------------------------------------------------------------ statistics per unit


def _stat(ctx, name):
    cal = ctx.calibration
    if cal is None or cal.get(name) is None:
        return None
    return np.asarray(cal.array(name, "<f4"), np.float64)


def _encode(ctx, fmt: str) -> bytes:
    if fmt not in NMAX:
        return kquant.RTN[fmt](ctx.rows)
    u, x = ctx.unit, ctx.rows
    kind, i = u.kind, u.layer
    full = None
    if kind in ("attn.q", "attn.k", "attn.v", "gdn.qkv", "gdn.z"):
        full = _stat(ctx, f"blk.{i}.attn_input.xtx")
    elif kind in ("shexp.gate", "shexp.up"):
        full = _stat(ctx, f"blk.{i}.ffn_input.xtx")
    elif kind == "shexp.down":
        full = _stat(ctx, f"blk.{i}.ffn_down_shexp.xtx")
    if full is not None and full.shape == (x.shape[1], x.shape[1]):
        return quantize_gptq(x, full, fmt)
    if kind in ("exps.gate", "exps.up", "exps.down"):
        name = f"blk.{i}.exps_down.sumsq" if kind == "exps.down" else f"blk.{i}.exps_input.sumsq"
        ss, cnt = _stat(ctx, name), _stat(ctx, f"blk.{i}.exps.count")
        if ss is not None and cnt is not None:
            n_exp = ss.shape[0]
            per = x.shape[0] // n_exp
            imp = np.where(cnt[:, None] > 0, ss / np.maximum(cnt[:, None], 1.0), 1.0)
            imp = np.where(imp > 0, imp, np.maximum(imp.max(1, keepdims=True), 1e-30))
            return b"".join(kquant.chunked(lambda r, imp=imp[e]: quantize_weighted(r, imp, fmt))(x[e * per:(e + 1) * per])
                            for e in range(n_exp))
    return kquant.chunked(lambda r: quantize_weighted(r, np.ones(r.shape[1]), fmt))(x)


register(Encoder("kq_xtx", 1, "regenerable", _encode, formats=("Q4_K", "Q5_K", "Q6_K", "Q8_0")))
