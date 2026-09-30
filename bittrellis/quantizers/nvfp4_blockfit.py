"""NVFP4 with each 16-value block scale chosen by reconstruction error, fitted per projection kind.

The shipped bytes (and `rtn`) give every block the scale that puts its largest value on E2M1's top
level, 6. That is rarely the best scale: E2M1's levels are dense below 2 and sparse above, so most
blocks lose less when the scale moves a few E4M3 steps (their largest value then lands near 4, or
is clipped, and the bulk of the block falls on the fine levels). This encoder keeps the ModelOpt
layout and the max-calibrated tensor scale, tries the E4M3 block scales around the round-to-nearest
one, and keeps the one with the lowest block error. Values round to the nearest E2M1 level (ties
toward the smaller magnitude, as `e2m1_encode`).

The error measure depends on the projection kind; both were chosen by measuring the pinned model
(section-balanced RP-KL, hpc01-e4) with each measure applied everywhere:

* MLP blocks: sum |error|^1.5. Against squared error it lowers MLP-only drift from 0.1218 to 0.1129
  (L1: 0.1193): large weights matter, but a squared measure over-weights them.
* attention and GDN projections: squared error. On these, the 1.5 measure costs math and
  multilingual drift that squared error keeps.

Clipping is allowed. Forbidding scales that clip a block's largest value raised drift from 0.1129 to
0.1297 (squared error, every unit).

It reads only the Linear's own BF16 weight, needs no calibration data, and is deterministic: NumPy
on the CPU, fixed row chunks, float64 loss accumulation. The squared-error fit scales and rounds in
float64; the |e|^1.5 fit rounds through an exact bucket table in float32.
"""

from __future__ import annotations

import numpy as np

from ..precision import NVFP4
from ..quant.formats import E2M1_TABLE, E4M3_TABLE, FP4_E2M1_MAX, FP8_E4M3_MAX, e2m1_encode, e4m3_encode, pack_nibbles
from .base import Produced, QuantContext, Quantizer, f32_rows

# Per kind: (error power, candidate scale offsets from the round-to-nearest E4M3 code). On the pinned
# weights every block's best scale lies inside these windows; the search over all 126 codes agrees.
FITS = {"mlp": (1.5, range(-3, 8)), "attn": (2.0, range(-2, 8)), "gdn": (2.0, range(-2, 8))}
CHUNK_ROWS = 1024

_E4M3_POS = E4M3_TABLE[:127].astype(np.float64)                 # codes 0x00..0x7E, increasing
_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
# y = |x| / scale falls in bucket ceil(4y) - 1; every E2M1 rounding midpoint (0.25, 0.75, 1.25, 1.75,
# 2.5, 3.5, 5) is a bucket edge, so the bucket decides the nearest level exactly (ties to the smaller).
_MIDS4 = np.array([1, 3, 5, 7, 10, 14, 20])
_BUCKET_LEVEL = np.searchsorted(_MIDS4, np.arange(0, 26) + 1, side="left").astype(np.uint8)


def tensor_scale(amax: float) -> np.float32:
    """ModelOpt's max-calibrated weight_scale_2."""
    return np.float32(amax / (FP4_E2M1_MAX * FP8_E4M3_MAX)) if amax > 0 else np.float32(1.0)


def _block_error(r: np.ndarray, power: float) -> np.ndarray:
    """sum over each 16-block of |r|^power, accumulated in float64. `r` is float32 [rows, blocks, 16]."""
    if power == 2.0:
        return np.einsum("ijk,ijk->ij", r, r, dtype=np.float64)
    r = np.abs(r).astype(np.float64)
    return (r ** power).sum(-1)


def _encode_rows_sq(w: np.ndarray, ws2: float, offsets) -> tuple[np.ndarray, np.ndarray]:
    """Squared-error fit, rounding decided in float64 (values scaled in float64, then encoded)."""
    blocks = w.reshape(w.shape[0], -1, 16).astype(np.float64)
    rtn = e4m3_encode((np.abs(blocks).max(-1) / FP4_E2M1_MAX / ws2).astype(np.float32)).astype(np.int64)
    first = np.where(rtn & 0x80, 0, rtn).clip(1, 126)
    best = (blocks ** 2).sum(-1)                                   # scale 0: every value to 0
    best_code = np.zeros(blocks.shape[:2], np.int64)
    best_q = np.zeros(blocks.shape, np.uint8)
    for k in offsets:
        code = np.clip(first + k, 1, 126)
        s = (_E4M3_POS[code] * ws2)[..., None]
        q = e2m1_encode((blocks / s).astype(np.float32))
        loss = ((blocks - E2M1_TABLE[q].astype(np.float64) * s) ** 2).sum(-1)
        take = loss < best
        best[take] = loss[take]
        best_code[take] = code[take]
        best_q[take] = q[take]
    return pack_nibbles(best_q.reshape(w.shape)), best_code.astype(np.uint8)


def encode_rows(w: np.ndarray, ws2: float, power: float = 2.0, offsets=range(-2, 8)) -> tuple[np.ndarray, np.ndarray]:
    """float32 [rows, cols] -> (packed E2M1 codes [rows, cols/2], E4M3 block-scale codes [rows, cols/16])."""
    w = np.asarray(w, dtype=np.float32)
    if power == 2.0:
        return _encode_rows_sq(w, float(ws2), offsets)
    a = np.abs(w).reshape(w.shape[0], -1, 16)
    rtn = e4m3_encode(a.max(-1) / np.float32(FP4_E2M1_MAX) / np.float32(ws2)).astype(np.int64)
    first = np.where(rtn & 0x80, 0, rtn).clip(1, 126)

    best = _block_error(a, power)                                  # scale 0: every value to 0
    best_code = np.zeros(a.shape[:2], np.int64)
    best_idx = np.zeros(a.shape, np.uint8)
    for k in offsets:
        code = np.clip(first + k, 1, 126)
        s = (_E4M3_POS[code] * float(ws2)).astype(np.float32)[..., None]
        b = np.ceil(a * (np.float32(4.0) / s))
        np.subtract(b, 1, out=b)
        np.clip(b, 0, 25, out=b)
        idx = _BUCKET_LEVEL[b.astype(np.uint8)]
        loss = _block_error(a - _LEVELS[idx] * s, power)
        take = loss < best
        best[take] = loss[take]
        best_code[take] = code[take]
        best_idx[take] = idx[take]
    codes = np.where((w < 0).reshape(a.shape), best_idx | 0x8, best_idx).astype(np.uint8)
    return pack_nibbles(codes.reshape(w.shape)), best_code.astype(np.uint8)


def quantize(w: np.ndarray, kind: str = "attn", amax: float | None = None) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """float32 [rows, cols] -> ModelOpt NVFP4 (packed, block scales, weight_scale_2), fitted for `kind`."""
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % 16 or not np.isfinite(w).all():
        raise ValueError("expected finite [rows, cols] with cols divisible by 16")
    power, offsets = FITS[kind]
    ws2 = tensor_scale(float(np.abs(w).max()) if amax is None else float(amax))
    packed = np.empty((w.shape[0], w.shape[1] // 2), np.uint8)
    scales = np.empty((w.shape[0], w.shape[1] // 16), np.uint8)
    for r in range(0, w.shape[0], CHUNK_ROWS):
        packed[r:r + CHUNK_ROWS], scales[r:r + CHUNK_ROWS] = encode_rows(w[r:r + CHUNK_ROWS], float(ws2), power, offsets)
    return packed, scales, ws2


class NVFP4BlockFit(Quantizer):
    name = "nvfp4_blockfit"
    version = 1
    formats = (NVFP4,)
    lineage = "regenerable"
    replay_mode = "independent"
    description = "NVFP4, each block scale chosen by reconstruction error (|e|^1.5 for MLP, e^2 elsewhere)"

    def supports(self, unit, fmt: str) -> bool:
        return fmt == NVFP4 and unit.kind in FITS

    def encode(self, ctx: QuantContext, unit, lin, fmt: str) -> list[Produced]:
        if not self.supports(unit, fmt):
            raise ValueError(f"{self.ref} encodes MLP, attention and GDN projections as NVFP4 only")
        power, offsets = FITS[unit.kind]
        amax = 0.0
        for _, rows in f32_rows(ctx, lin):
            if not np.isfinite(rows).all():
                raise ValueError(f"{lin.prefix}: base weight is not finite")
            amax = max(amax, float(np.abs(rows).max()))
        ws2 = tensor_scale(amax)
        packed = np.empty((lin.rows, lin.cols // 2), np.uint8)
        scales = np.empty((lin.rows, lin.cols // 16), np.uint8)
        for start, rows in f32_rows(ctx, lin, chunk=CHUNK_ROWS):
            p, s = encode_rows(rows, float(ws2), power, offsets)
            packed[start:start + len(rows)], scales[start:start + len(rows)] = p, s
        return [(".weight", "U8", packed.shape, packed),
                (".weight_scale", "F8_E4M3", scales.shape, scales),
                (".weight_scale_2", "F32", (), np.asarray(ws2, "<f4").reshape(()))]
