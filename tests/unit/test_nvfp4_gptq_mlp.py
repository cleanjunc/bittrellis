import numpy as np
import pytest

from bittrellis.quant.formats import dequantize_nvfp4
from bittrellis.quantizers import nvfp4_blockfit as bf
from bittrellis.quantizers import nvfp4_gptq as gq
from bittrellis.quantizers import nvfp4_gptq_mlp as gm
from bittrellis.synthetic import make_tiny_calibration

QUIET = {"log": lambda *_: None, "verify": False}


def _weights(seed, shape=(48, 256)):
    return (np.random.default_rng(seed).standard_t(5, size=shape) * 0.02).astype(np.float32)


def _hessian(seed, n=256, tokens=1024):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((tokens, n)) @ (rng.standard_normal((n, n)) * 0.2 + np.eye(n))
    x *= np.exp(rng.standard_normal(n))
    return (x.T @ x / tokens).astype(np.float32)


def _out_err(w, result, h):
    e = w.astype(np.float64) - dequantize_nvfp4(*result)
    return float(np.einsum("ij,jk,ik->", e, h.astype(np.float64), e))


def test_upper_inverse_factor():
    h = _hessian(1, 64)
    u = gm.upper_inverse_factor(h, 0.01).astype(np.float64)
    hd = h.astype(np.float64) + 0.01 * np.mean(np.diag(h)) * np.eye(64)
    assert np.allclose(np.triu(u), u)
    assert np.allclose(u.T @ u @ hd, np.eye(64), atol=2e-3)


def test_block_order_keeps_whole_blocks():
    h = _hessian(2, 64)
    order = gm.block_order(h)
    assert sorted(order.tolist()) == list(range(4))
    means = np.diag(h).reshape(4, 16).mean(1)
    assert all(means[order[i]] >= means[order[i + 1]] for i in range(3))


def test_act_order_bytes_repeat_and_beat_blockfit():
    w, h = _weights(3, (64, 256)), _hessian(3)
    a, b = gm.gptq_act_order(w, h, 0.01), gm.gptq_act_order(w, h, 0.01)
    assert all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(a, b, strict=True))
    fit = bf.quantize(w, "mlp")
    assert a[2] == fit[2]
    assert _out_err(w, a, h) < 0.8 * _out_err(w, fit, h)


def test_act_order_is_not_worse_than_plain_gptq():
    w, h = _weights(4, (64, 256)), _hessian(4)
    assert _out_err(w, gm.gptq_act_order(w, h, gq.DAMP), h) <= 1.05 * _out_err(w, gq.quantize(w, h), h)


def test_down_hessian_is_seeded_symmetric_and_psd():
    rng = np.random.default_rng(5)
    gate, up = _weights(5, (64, 32)), _weights(6, (64, 32))
    h = _hessian(7, 32)
    a = gm.down_hessian(gate, up, h, gm._seed("m.layers.1.mlp"), samples=4096)
    b = gm.down_hessian(gate, up, h, gm._seed("m.layers.1.mlp"), samples=4096)
    c = gm.down_hessian(gate, up, h, gm._seed("m.layers.2.mlp"), samples=4096)
    assert a.shape == (64, 64) and a.dtype == np.float32
    assert a.tobytes() == b.tobytes() and a.tobytes() != c.tobytes()
    assert np.allclose(a, a.T, rtol=1e-5, atol=1e-9)
    assert np.linalg.eigvalsh(a.astype(np.float64)).min() > -1e-6 * np.abs(a).max()
    del rng


def test_down_gptq_lowers_the_mlp_output_error():
    """The down projection, rounded on the estimated statistics, lowers the block's output error on fresh inputs."""
    rng = np.random.default_rng(8)
    n_in, inter, n_out = 64, 128, 64
    gate, up = _weights(9, (inter, n_in)) * 4, _weights(10, (inter, n_in)) * 4
    down = _weights(11, (n_out, inter))
    h = _hessian(12, n_in)
    hd = gm.shrink(gm.down_hessian(gate, up, h, gm._seed("m.layers.0.mlp"), samples=8192))
    ours, fit = gm.gptq_act_order(down, hd, gm.DAMP_DOWN), bf.quantize(down, "mlp")
    x = (rng.standard_normal((4096, n_in)) @ np.linalg.cholesky(h.astype(np.float64)).T).astype(np.float32)
    a = x @ gate.T
    hid = a / (1 + np.exp(-a)) * (x @ up.T)
    ref = hid @ down.T
    err = lambda r: float(((hid @ dequantize_nvfp4(*r).T - ref) ** 2).sum())  # noqa: E731
    assert err(ours) < 0.9 * err(fit)


def test_samples_for():
    assert gm.samples_for(16) == 4096
    assert gm.samples_for(17408) == 65536
    assert gm.samples_for(3000) % gm.SAMPLE_CHUNK == 0


@pytest.mark.parametrize("w", [np.ones((2, 17)), np.full((1, 16), np.nan)])
def test_rejects_invalid_weights(w):
    with pytest.raises(ValueError):
        gm.gptq_act_order(w, np.eye(w.shape[-1], dtype=np.float32), 0.01)


def _manifest(q="nvfp4_gptq_mlp"):
    from bittrellis.manifest import Manifest

    return Manifest.from_dict({
        "schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "test-gptq-mlp", "default": "NVFP4",
        "rules": [{"match": "L*.mlp", "format": "NVFP4", "quantizer": q},
                  {"match": "L*.attn.*", "format": "NVFP4", "quantizer": "nvfp4_blockfit"},
                  {"match": "L*.gdn.*", "format": "NVFP4", "quantizer": "nvfp4_blockfit"}],
    })


def test_tiny_build_is_repeatable_and_audited(tiny_all, tmp_path, monkeypatch):
    from bittrellis.build import build
    from bittrellis.safetensors_io import SafeTensorsDir
    from bittrellis.track import load_track
    from bittrellis.validate import audit

    monkeypatch.setenv("BITTRELLIS_REPLAY_CACHE", str(tmp_path / "cache"))
    calib = make_tiny_calibration(tmp_path / "calib", seed=0)
    sources = {**dict(zip(("base", "gittensor_nvfp4", "unsloth_nvfp4"), tiny_all, strict=True)), "calibration": calib}
    first, second = tmp_path / "first", tmp_path / "second"
    record = build(_manifest(), load_track("HPC-01"), sources, first, **QUIET)
    again = build(_manifest(), load_track("HPC-01"), sources, second, **QUIET)
    assert record["files"] == again["files"]
    result = audit(first, _manifest(), sources, verify_sources=False)
    assert result.ok, result.errors
    assert result.lineage["nvfp4_gptq_mlp@v1"]["replayed_tensors"] > 0

    # unlike nvfp4_gptq, down_proj gets its own calibrated bytes
    v1 = tmp_path / "v1"
    build(_manifest("nvfp4_gptq"), load_track("HPC-01"), sources, v1, **QUIET)
    with SafeTensorsDir(first) as a, SafeTensorsDir(v1) as b:
        down = [n for n in a.tensors if n.endswith("mlp.down_proj.weight")]
        assert down and any(bytes(a.raw(n)) != bytes(b.raw(n)) for n in down)

    # other statistics regenerate other bytes, so the audit rejects the checkpoint
    monkeypatch.setenv("BITTRELLIS_REPLAY_CACHE", str(tmp_path / "cache2"))
    other = {**sources, "calibration": make_tiny_calibration(tmp_path / "other", seed=5)}
    assert not audit(first, _manifest(), other, verify_sources=False).ok


def test_refuses_to_build_without_the_statistics(tiny_all, tmp_path):
    from bittrellis.build import build
    from bittrellis.track import load_track

    sources = dict(zip(("base", "gittensor_nvfp4", "unsloth_nvfp4"), tiny_all, strict=True))
    with pytest.raises(ValueError, match="needs the calibration"):
        build(_manifest(), load_track("HPC-01"), sources, tmp_path / "x", **QUIET)


def test_the_fingerprint_probe_is_repeatable_and_differs_from_nvfp4_gptq():
    from bittrellis.fingerprint import by_quantizer, probe, similarity

    first = probe(["nvfp4_gptq_mlp"], seed=3)
    assert first and first == probe(["nvfp4_gptq_mlp"], seed=3)
    ours = by_quantizer(first)["nvfp4_gptq_mlp@v1"]
    v1 = by_quantizer(probe(["nvfp4_gptq"], seed=3))["nvfp4_gptq@v1"]
    sim, compared = similarity(ours, v1)
    assert compared and sim < 0.99
