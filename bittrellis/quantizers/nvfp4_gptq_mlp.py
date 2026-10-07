"""NVFP4 MLP encoder with GPTQ rounding on all three projections, the down projection included.

`nvfp4_gptq` rounds gate_proj and up_proj by GPTQ on the pinned input statistics H = mean x xᵀ and gives
down_proj the plain `nvfp4_blockfit` MLP fit, because the pinned calibration has no statistics for down's input
h = silu(gate·x) ⊙ (up·x). On the pinned weights and statistics the down projection carries about 60% of the MLP
block's remaining output error after gate/up GPTQ, so this encoder estimates down's input statistics from what is
pinned and rounds down by GPTQ as well:

* **down's input statistics.** x is drawn from N(0, H), a Gaussian with the pinned second moment (each
  pre-activation gate·x is a sum over 5,120 inputs, so it is close to Gaussian whatever x looks like), h is
  computed from this MLP unit's own BF16 gate and up weights, and H_d = mean h hᵀ over `samples_for(cols)` draws.
  H_d is shrunk 30% toward its diagonal (sampling noise in a [17408, 17408] second moment) and damped 10%.
  Only the pinned statistics and the base weights of the same unit are read, so the bytes are a function of the
  unit's BF16 weights and the pinned calibration, like every independent encoder's.
* **act-order.** GPTQ rounds whole 16-column blocks in descending order of their mean input energy (diag of H or
  H_d), so the strongest inputs are rounded first and the later, weaker ones absorb their error. NVFP4 block
  scales stay on the original 16-column blocks; only the order in which blocks are rounded changes.
* gate and up use `nvfp4_gptq`'s damping (1%); scales, levels and layout are those of `nvfp4_gptq`.

On the pinned weights and statistics (4,096-channel slices of the MLP; held-out Gaussian and Student-t inputs with
the pinned covariance) the MLP block's output error falls from nvfp4_gptq's by 13-16% on layer 30 (9.1e-3 to
7.6e-3; blockfit 1.7e-2) and by 55-64% on layer 58 (6.8e-3 to 3.0e-3).

Determinism: NumPy on the CPU; the samples come from a PCG64 generator seeded by the unit's module path; matrix
products and factorizations run on one machine for a build and its audit replay, as in `nvfp4_gptq`.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..precision import NVFP4
from . import nvfp4_gptq as gq
from .base import Produced, QuantContext, Quantizer, f32_weight, input_hessian

DAMP_GATE_UP = gq.DAMP      # as nvfp4_gptq
DAMP_DOWN = 0.10            # relative diagonal dampening of the estimated down Hessian
SHRINK_DOWN = 0.30          # share of the estimated down Hessian replaced by its diagonal
SAMPLE_CHUNK = 2048         # draws per accumulation step (bounded memory)
MAX_SAMPLES = 65536


def samples_for(cols: int) -> int:
    """Draws used to estimate down's [cols, cols] input second moment: about four per column, at most 65,536."""
    n = min(MAX_SAMPLES, max(4096, 4 * cols))
    return -(-n // SAMPLE_CHUNK) * SAMPLE_CHUNK


def _seed(path: str) -> np.random.Generator:
    return np.random.default_rng(np.frombuffer(hashlib.sha256(path.encode()).digest()[:16], "<u4"))


def block_order(h: np.ndarray) -> np.ndarray:
    """16-column blocks in descending mean diag(h) (ties keep their position)."""
    d = np.diag(np.asarray(h, dtype=np.float64)).reshape(-1, 16).mean(1)
    return np.argsort(-d, kind="stable")


def upper_inverse_factor(h: np.ndarray, damp: float) -> np.ndarray:
    """Upper U with UᵀU = (h + λI)⁻¹, from one Cholesky and one triangular inverse of the reversed matrix (no
    explicit inverse of h), float32 throughout to bound memory on [17408, 17408]. Dead inputs (zero diagonal) are
    set to 1 as in nvfp4_gptq."""
    h = np.array(h, dtype=np.float32)
    d = np.diag(h).copy()
    dead = d <= 0
    h[dead, dead] = 1.0
    h[np.diag_indices_from(h)] += np.float32(damp * float(np.mean(np.diag(h), dtype=np.float64)))
    m = np.linalg.cholesky(h[::-1, ::-1])           # J h J = M Mᵀ
    del h
    minv = np.linalg.inv(m)
    del m
    return np.ascontiguousarray(minv[::-1, ::-1])    # U = J M⁻¹ J is upper and UᵀU = h⁻¹


def gptq_act_order(w: np.ndarray, h: np.ndarray, damp: float, amax: float | None = None,
                   u: np.ndarray | None = None, order: np.ndarray | None = None):
    """GPTQ with whole-block act-order. Returns ModelOpt NVFP4 (packed, block scales, weight_scale_2)."""
    from ..quant.formats import pack_nibbles
    from . import nvfp4_blockfit as bf

    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % 16 or not np.isfinite(w).all():
        raise ValueError("expected finite [rows, cols] with cols divisible by 16")
    if h.shape != (w.shape[1], w.shape[1]) or not np.isfinite(h).all():
        raise ValueError("expected a finite [cols, cols] input Hessian")
    order = block_order(h) if order is None else order
    perm = (order[:, None] * 16 + np.arange(16)[None, :]).reshape(-1)
    if u is None:
        u = upper_inverse_factor(np.asarray(h)[np.ix_(perm, perm)], damp)
    ws2 = bf.tensor_scale(float(np.abs(w).max()) if amax is None else float(amax))
    rows, cols = w.shape
    packed = np.empty((rows, cols // 2), np.uint8)
    scales = np.empty((rows, cols // 16), np.uint8)
    for r in range(0, rows, gq.CHUNK_ROWS):
        codes_p, sc_p = gq.gptq_rows(w[r:r + gq.CHUNK_ROWS][:, perm], u, float(ws2))
        codes = np.empty_like(codes_p)
        codes[:, perm] = codes_p
        sc = np.empty_like(sc_p)
        sc[:, order] = sc_p
        packed[r:r + gq.CHUNK_ROWS], scales[r:r + gq.CHUNK_ROWS] = pack_nibbles(codes), sc
    return packed, scales, ws2


def down_hessian(gate: np.ndarray, up: np.ndarray, h: np.ndarray, rng: np.random.Generator,
                 samples: int | None = None) -> np.ndarray:
    """Estimated mean h hᵀ of down's input h = silu(gate·x) ⊙ (up·x) for x ~ N(0, h), float32 [inter, inter]."""
    gate = np.asarray(gate, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    n_in = h.shape[0]
    hh = np.array(h, dtype=np.float64)
    hh[np.diag_indices_from(hh)] += 1e-6 * float(np.mean(np.diag(hh))) + 1e-30
    lt = np.linalg.cholesky(hh).T.astype(np.float32)          # x = z @ lt
    del hh
    samples = samples_for(gate.shape[0]) if samples is None else samples
    acc = np.zeros((gate.shape[0], gate.shape[0]), np.float32)
    with np.errstate(over="ignore"):      # exp(-a) -> inf for very negative a gives silu(a) = -0.0, as intended
        for _ in range(0, samples, SAMPLE_CHUNK):
            x = rng.standard_normal((SAMPLE_CHUNK, n_in), dtype=np.float32) @ lt
            a = x @ gate.T
            hid = a / (1 + np.exp(-a))
            hid *= x @ up.T
            acc += hid.T @ hid
    acc /= np.float32(samples)
    return acc


def shrink(hd: np.ndarray, a: float = SHRINK_DOWN) -> np.ndarray:
    d = np.diag(hd).copy()
    hd *= np.float32(1 - a)
    hd[np.diag_indices_from(hd)] += np.float32(a) * d
    return hd


class NVFP4GPTQMLP(Quantizer):
    name = "nvfp4_gptq_mlp"
    version = 1
    formats = (NVFP4,)
    lineage = "regenerable"
    replay_mode = "independent"
    description = ("NVFP4 MLP: GPTQ with block act-order on gate/up (pinned statistics) and on down "
                   "(statistics derived from the pinned ones and the unit's gate/up)")

    def supports(self, unit, fmt: str) -> bool:
        return fmt == NVFP4 and unit.kind == "mlp"

    def available(self, ctx: QuantContext, unit, fmt: str) -> str | None:
        return None if ctx.calibration is not None else "needs the calibration statistics (bittrellis calibration fetch)"

    def encode(self, ctx: QuantContext, unit, lin, fmt: str) -> list[Produced]:
        if not self.supports(unit, fmt):
            raise ValueError(f"{self.ref} encodes MLP blocks as NVFP4 only")
        if ctx.calibration is None:
            raise ValueError(f"{self.ref} needs the calibration statistics (bittrellis calibration fetch)")
        module = lin.prefix.rpartition(".")[0]
        by_leaf = {x.prefix.rpartition(".")[2]: x for x in unit.linears}
        gate_lin, up_lin = by_leaf.get("gate_proj"), by_leaf.get("up_proj")
        if gate_lin is None or up_lin is None:
            raise ValueError(f"{unit.id}: expected gate_proj and up_proj in the MLP unit")
        h = input_hessian(ctx, gate_lin)
        if h is None:
            raise ValueError(f"{gate_lin.prefix}: the pinned calibration has no statistics for this Linear")
        w = f32_weight(ctx, lin)
        if not np.isfinite(w).all():
            raise ValueError(f"{lin.prefix}: base weight is not finite")
        cache = ctx.state.setdefault(self.name, {})
        if lin.prefix.endswith(".down_proj"):
            gate, up = f32_weight(ctx, gate_lin), f32_weight(ctx, up_lin)
            hd = shrink(down_hessian(gate, up, h, _seed(module)))
            del gate, up
            packed, scales, ws2 = gptq_act_order(w, hd, DAMP_DOWN)
        else:
            if cache.get("key") != module:
                order = block_order(h)
                perm = (order[:, None] * 16 + np.arange(16)[None, :]).reshape(-1)
                cache.update(key=module, order=order, u=upper_inverse_factor(np.asarray(h)[np.ix_(perm, perm)], DAMP_GATE_UP))
            packed, scales, ws2 = gptq_act_order(w, h, DAMP_GATE_UP, u=cache["u"], order=cache["order"])
        return [(".weight", "U8", packed.shape, packed),
                (".weight_scale", "F8_E4M3", scales.shape, scales),
                (".weight_scale_2", "F32", (), np.asarray(ws2, "<f4").reshape(()))]
