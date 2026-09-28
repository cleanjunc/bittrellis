"""Quantizer registry.

To add a quantizer: implement `Quantizer` (see base.py for the contract) in a new module here and
add an instance to `_BUILTIN`. Its `ref` (name@vN) becomes part of every candidate id that uses it.
Attested quantizers additionally need a maintainer-approved entry in configs/sources.lock.json.
"""

from __future__ import annotations

from .base import Produced, QuantContext, Quantizer
from .builtin import RTN, Baseline, Runtime, Unsloth
from .nvfp4_blockfit import NVFP4BlockFit

_BUILTIN: list[Quantizer] = [Runtime(), Baseline(), Unsloth(), RTN(), NVFP4BlockFit()]
REGISTRY: dict[str, Quantizer] = {q.name: q for q in _BUILTIN}
DEFAULT_FOR_FORMAT = {"NVFP4": "baseline", "FP8": "rtn", "Q4_K": "runtime"}


class Foreign(Quantizer):
    """A contributed quantizer known only by its ref, in a process that must not execute its code.

    The evaluator's trusted side uses it to compute candidate ids and to audit a checkpoint whose
    samples the contributed code regenerated in isolation (see validate.audit, regenerated_dir).
    """

    lineage = "regenerable"
    replay_mode = "independent"

    def __init__(self, name: str, version: int, formats: tuple[str, ...]):
        self.name, self.version, self.formats = name, version, formats

    def encode(self, ctx, unit, lin, fmt):
        raise RuntimeError(f"{self.ref} is contributed code; it runs only in the sandbox")


def register_foreign(refs: dict[str, tuple[str, ...]]) -> list[Quantizer]:
    """Register stubs for `{"name@vN": formats}` not already known. Returns the stubs added."""
    added = []
    for ref, formats in sorted(refs.items()):
        name, _, v = ref.partition("@v")
        if name in REGISTRY:
            if REGISTRY[name].ref != ref:
                raise ValueError(f"{ref}: this code base has {REGISTRY[name].ref}")
            continue
        added.append(register(Foreign(name, int(v), tuple(formats))))
    return added


def get(name: str) -> Quantizer:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown quantizer {name!r} (known: {sorted(REGISTRY)})") from None


def register(q: Quantizer) -> Quantizer:
    """Register an additional quantizer (tests and experimental plugins)."""
    if q.name in REGISTRY and REGISTRY[q.name] is not q:
        raise ValueError(f"quantizer {q.name!r} already registered")
    if q.lineage not in ("runtime", "regenerable", "attested"):
        raise ValueError(f"{q.name}: bad lineage {q.lineage!r}")
    if q.lineage == "regenerable" and q.replay_mode not in ("independent", "sequential"):
        raise ValueError(f"{q.name}: regenerable quantizers need replay_mode independent|sequential")
    REGISTRY[q.name] = q
    return q


__all__ = ["REGISTRY", "DEFAULT_FOR_FORMAT", "Foreign", "Produced", "QuantContext", "Quantizer", "get", "register", "register_foreign"]
