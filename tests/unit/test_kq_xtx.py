import numpy as np
import pytest
import yaml

gguf = pytest.importorskip("gguf")

from bittrellis import hpc02, kquant  # noqa: E402
from bittrellis.hpc02_encoders import kq_xtx as G  # noqa: E402
from bittrellis.safetensors_io import SafeTensorsDir, ShardWriter  # noqa: E402

QUIET = {"log": lambda *_: None}


def _x(seed, rows=96, cols=512):
    return (np.random.default_rng(seed).standard_t(5, (rows, cols)) * 0.02).astype(np.float32)


def _h(seed, n=512, tokens=4096):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((tokens, n)) @ (rng.standard_normal((n, n)) * 0.15 + np.eye(n))
    a *= np.exp(rng.standard_normal(n) * 0.7)            # uneven channel energy, like real activations
    return (a.T @ a / tokens).astype(np.float32)


def _out_err(x, data, fmt, h):
    e = x.astype(np.float64) - kquant.dequantize(fmt, data, x.shape[1])
    return float(np.einsum("ij,jk,ik->", e, h.astype(np.float64), e))


@pytest.mark.parametrize("fmt", ["Q4_K", "Q5_K"])
def test_bytes_repeat_decode_and_beat_rtn(fmt):
    x, h = _x(1), _h(1)
    g1, g2 = G.quantize_gptq(x, h, fmt), G.quantize_gptq(x, h, fmt)
    assert g1 == g2 and len(g1) == kquant.row_bytes(fmt, x.shape[1]) * x.shape[0]
    w = G.quantize_weighted(x, np.diag(h), fmt)
    rtn = kquant.RTN[fmt](x)
    assert _out_err(x, w, fmt, h) < 0.9 * _out_err(x, rtn, fmt, h)
    assert _out_err(x, g1, fmt, h) < 0.6 * _out_err(x, rtn, fmt, h)


def test_uniform_weights_do_not_lose_to_rtn():
    x = _x(2)
    for fmt in ("Q4_K", "Q5_K"):
        ours = kquant.dequantize(fmt, G.quantize_weighted(x, np.ones(x.shape[1]), fmt), x.shape[1])
        rtn = kquant.dequantize(fmt, kquant.RTN[fmt](x), x.shape[1])
        assert ((x - ours) ** 2).sum() <= ((x - rtn) ** 2).sum()


def test_rows_are_independent_across_chunks(monkeypatch):
    x, h = _x(3, rows=40), _h(3)
    full = G.quantize_gptq(x, h, "Q4_K")
    monkeypatch.setattr(G, "CHUNK_ROWS", 7)
    assert G.quantize_gptq(x, h, "Q4_K") == full


def test_dead_inputs_and_zero_rows():
    h = _h(4)
    h[:, 9] = h[9, :] = 0
    assert np.isfinite(G.inverse_hessian_factor(h)).all()
    z = np.zeros((3, 512), np.float32)
    assert not kquant.dequantize("Q4_K", G.quantize_gptq(z, h, "Q4_K"), 512).any()


def _calibration(root, layers=2, n=256, inter=256, experts=4, seed=0):
    rng = np.random.default_rng(seed)
    w = ShardWriter(root, 1 << 22)
    for i in range(layers):
        for name, k in (("attn_input", n), ("ffn_input", n), ("ffn_down_shexp", inter)):
            a = rng.standard_normal((4 * k, k)) * rng.uniform(0.2, 3.0, k)
            w.add(f"blk.{i}.{name}.xtx", "F32", (k, k), (a.T @ a / len(a)).astype("<f4"))
        w.add(f"blk.{i}.exps_input.sumsq", "F32", (experts, n), rng.uniform(0.1, 5, (experts, n)).astype("<f4"))
        w.add(f"blk.{i}.exps_down.sumsq", "F32", (experts, inter), rng.uniform(0.1, 5, (experts, inter)).astype("<f4"))
        w.add(f"blk.{i}.exps.count", "F32", (experts,), np.array([5, 0, 9, 2][:experts], "<f4"))
    w.close()
    return root


def test_every_statistics_path(tmp_path):
    cal = SafeTensorsDir(_calibration(tmp_path / "cal"))
    U = hpc02.Unit
    cases = [U("L0.gdn.qkv", "blk.0.attn_qkv.weight", 0, "gdn.qkv", 256, 64, 0),          # full H: GPTQ
             U("L1.shexp.down", "blk.1.ffn_down_shexp.weight", 1, "shexp.down", 256, 32, 0),  # shared-expert down H
             U("L0.exps.gate", "blk.0.ffn_gate_exps.weight", 0, "exps.gate", 256, 4 * 16, 0),  # per-expert diagonal
             U("L1.exps.down", "blk.1.ffn_down_exps.weight", 1, "exps.down", 256, 4 * 16, 0),
             U("L0.gdn.out", "blk.0.ssm_out.weight", 0, "gdn.out", 256, 32, 0)]              # no statistics
    for u in cases:
        x = _x(5, rows=u.rows, cols=u.cols)
        a = G._encode(hpc02.EncodeContext(x, u, {}, cal), "Q4_K")
        assert a == G._encode(hpc02.EncodeContext(x, u, {}, cal), "Q4_K")
        assert len(a) == kquant.row_bytes("Q4_K", u.cols) * u.rows
        assert kquant.dequantize("Q4_K", a, u.cols).shape == x.shape
    # without calibration every unit still encodes (weighted search with uniform weights)
    x = _x(6, rows=64, cols=256)
    assert len(G._encode(hpc02.EncodeContext(x, cases[0], {}, None), "Q4_K")) == kquant.row_bytes("Q4_K", 256) * 64
    # formats without a search here are kq_rtn's bytes
    for fmt in ("Q6_K", "Q8_0"):
        assert G._encode(hpc02.EncodeContext(x, cases[0], {}, cal), fmt) == kquant.RTN[fmt](x)


def test_registered():
    assert hpc02.ENCODERS["kq_xtx"].ref == "kq_xtx@v1" and hpc02.ENCODERS["kq_xtx"].lineage == "regenerable"


def test_tiny_build_is_audited(tmp_path):
    from tests.unit.test_hpc02 import write_gguf

    rng = np.random.default_rng(0)
    r = lambda *s: rng.standard_normal(s) * 0.02  # noqa: E731
    t = {"token_embd.weight": (r(64, 256), "bf16"), "output.weight": (r(64, 256), "bf16"), "output_norm.weight": (r(256), "f32")}
    for i in (0, 1):
        b = f"blk.{i}."
        t[b + "attn_norm.weight"] = (r(256), "f32")
        t[b + "ffn_gate_inp.weight"] = (r(4, 256), "f32")
        t[b + "ffn_gate_inp_shexp.weight"] = (r(1, 256), "f32")
        t[b + "ffn_gate_exps.weight"] = (r(4, 32, 256), "bf16")
        t[b + "ffn_up_exps.weight"] = (r(4, 32, 256), "bf16")
        t[b + "ffn_down_exps.weight"] = (r(4, 256, 256), "bf16")
        t[b + "ffn_gate_shexp.weight"] = (r(32, 256), "bf16")
        t[b + "ffn_up_shexp.weight"] = (r(32, 256), "bf16")
        t[b + "ffn_down_shexp.weight"] = (r(256, 256), "bf16")
        t[b + "attn_qkv.weight"] = (r(96, 256), "bf16")
        t[b + "attn_gate.weight"] = (r(64, 256), "bf16")
        t[b + "ssm_out.weight"] = (r(256, 256), "bf16")
        t[b + "ssm_alpha.weight"] = (r(4, 256), "bf16")
    tdir = tmp_path / "template"
    tdir.mkdir()
    write_gguf(tdir / "t.gguf", t)
    cal = _calibration(tmp_path / "cal")
    m = tmp_path / "m.yaml"
    m.write_text(yaml.safe_dump({"schema": "bittrellis/manifest@2", "track": "HPC-02", "name": "t", "default": "Q4_K",
                                 "encoders": {"Q4_K": "kq_xtx"}, "rules": [{"match": "L*.exps.down", "format": "Q5_K"}]}))
    out = tmp_path / "c.gguf"
    rec = hpc02.build(m, tdir, out, calibration_dir=cal, jobs=2, **QUIET)
    again = hpc02.build(m, tdir, tmp_path / "d.gguf", calibration_dir=cal, jobs=1, **QUIET)
    assert rec["candidate_id"] == again["candidate_id"] and out.read_bytes() == (tmp_path / "d.gguf").read_bytes()
    res = hpc02.audit(out, m, tdir, calibration_dir=cal, secret="s", verify=False, **QUIET)
    assert res["ok"], res["errors"]
