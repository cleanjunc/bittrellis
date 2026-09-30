import numpy as np
import pytest

from bittrellis.quant.formats import E4M3_TABLE, dequantize_nvfp4, quantize_nvfp4
from bittrellis.quantizers.nvfp4_blockfit import quantize


def _blocks_err(w, result):
    rows = w.shape[0]
    return np.sum((w.astype(np.float64) - dequantize_nvfp4(*result)).reshape(rows, -1, 16) ** 2, axis=-1)


def _weights(seed, shape=(48, 256)):
    return (np.random.default_rng(seed).standard_t(5, size=shape) * 0.02).astype(np.float32)


def test_bytes_repeat():
    w = _weights(1)
    a, b = quantize(w), quantize(w)
    assert all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(a, b, strict=True))


def test_never_worse_than_round_to_nearest_per_block():
    w = _weights(3)
    ours, rtn = quantize(w), quantize_nvfp4(w)
    assert ours[2] == rtn[2]
    # the encoder picks levels in float32; allow float32 rounding noise on the float64 comparison
    assert np.all(_blocks_err(w, ours) <= _blocks_err(w, rtn) * (1 + 2e-6) + 1e-15)


def test_matches_the_full_scale_search():
    w = _weights(6, (32, 256))
    packed, scales, ws2 = quantize(w)
    blocks = w.reshape(32, -1, 16).astype(np.float64)
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])
    best = (blocks ** 2).sum(-1)
    for code in range(1, 127):
        s = E4M3_TABLE[code] * float(ws2)
        near = levels[np.abs(np.abs(blocks / s)[..., None] - levels).argmin(-1)] * np.sign(blocks) * s
        best = np.minimum(best, ((blocks - near) ** 2).sum(-1))
    assert np.all(_blocks_err(w, (packed, scales, ws2)) <= best * (1 + 2e-6) + 1e-15)


def test_lower_error_than_round_to_nearest():
    w = _weights(4, (64, 512))
    assert _blocks_err(w, quantize(w)).sum() < 0.9 * _blocks_err(w, quantize_nvfp4(w)).sum()


def test_chunking_uses_the_tensor_amax():
    w = _weights(5, (40, 64))
    amax = float(np.abs(w).max())
    full = quantize(w, amax=amax)
    parts = [quantize(w[i:i + 8], amax=amax) for i in range(0, 40, 8)]
    for j in (0, 1):
        assert np.array_equal(full[j], np.concatenate([p[j] for p in parts]))
    assert all(p[2] == full[2] for p in parts)


def test_zero_weights():
    result = quantize(np.zeros((3, 32), np.float32))
    assert not dequantize_nvfp4(*result).any()


@pytest.mark.parametrize("w", [np.ones((2, 17)), np.ones(16), np.full((1, 16), np.nan)])
def test_rejects_invalid_weights(w):
    with pytest.raises(ValueError):
        quantize(w)


def test_tiny_build_is_repeatable_and_audited(tiny_all, tmp_path):
    from bittrellis.build import build
    from bittrellis.manifest import Manifest
    from bittrellis.safetensors_io import SafeTensorsDir
    from bittrellis.track import load_track
    from bittrellis.validate import audit

    manifest = Manifest.from_dict({
        "schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "test-blockfit", "default": "NVFP4",
        "rules": [{"match": m, "format": "NVFP4", "quantizer": "nvfp4_blockfit"}
                  for m in ("L*.mlp", "L*.attn.*", "L*.gdn.*")],
    })
    sources = dict(zip(("base", "gittensor_nvfp4", "unsloth_nvfp4"), tiny_all, strict=True))
    first, second = tmp_path / "first", tmp_path / "second"
    kwargs = {"verify": False, "log": lambda *_: None}
    record = build(manifest, load_track("HPC-01"), sources, first, **kwargs)
    again = build(manifest, load_track("HPC-01"), sources, second, **kwargs)
    assert record["files"] == again["files"]
    result = audit(first, manifest, sources, verify_sources=False)
    assert result.ok, result.errors
    assert result.lineage["nvfp4_blockfit@v1"]["replay_mode"] == "independent"

    with SafeTensorsDir(first) as built:
        # the audit always replays the first and last unit; the first is layer 0's recurrent qkv
        name = next(n for n in built.tensors if n.endswith("layers.0.linear_attn.in_proj_qkv.weight"))
        tensor = built.get(name)
    with open(tensor.file, "r+b") as fh:
        fh.seek(tensor.offset)
        byte = fh.read(1)
        fh.seek(tensor.offset)
        fh.write(bytes([byte[0] ^ 1]))
    assert not audit(first, manifest, sources, verify_sources=False).ok


def test_each_kind_uses_its_own_fit():
    from bittrellis.quantizers.nvfp4_blockfit import FITS
    w = _weights(7, (32, 256))
    mlp, attn = quantize(w, "mlp"), quantize(w, "attn")
    assert FITS["mlp"][0] == 1.5 and FITS["attn"][0] == FITS["gdn"][0] == 2.0
    assert quantize(w, "gdn")[1].tobytes() == attn[1].tobytes()
    assert mlp[1].tobytes() != attn[1].tobytes()          # a different measure picks different scales
    # the 1.5 fit is optimal for its own measure among the searched scales, so never worse than RTN there

    def err15(r):
        return np.sum(np.abs(w.astype(np.float64) - dequantize_nvfp4(*r)).reshape(32, -1, 16) ** 1.5, axis=-1)

    assert np.all(err15(mlp) <= err15(quantize_nvfp4(w)) * (1 + 2e-6) + 1e-15)


def test_rejects_units_it_does_not_fit():
    from bittrellis.manifest import Manifest, ManifestError
    from bittrellis.model.qwen38 import Qwen38Arch

    manifest = Manifest.from_dict({
        "schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "head-scope", "default": "NVFP4",
        "modules": {"lm_head": {"format": "NVFP4", "quantizer": "nvfp4_blockfit"}},
    })
    with pytest.raises(ManifestError):
        manifest.expand_assignments(Qwen38Arch().units())
