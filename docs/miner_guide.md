# Miner guide

> You don't touch the engine. You submit a manifest (a format and quantizer for each part of
> Qwen3.8-27B), a pinned RTX 5090 measures it, and a merged tier pays on Gittensor. [Terms](../README.md#key-terms).

```text
manifests/<name>.yaml (+ optional quantizer) ──PR──▶ evaluator
    screen (no GPU) → build → audit → RP-KL → 2 perf runs → tasks → private holdout → ε-frontier
◀── PR comment: score + tier (eval:XL … eval:XS / none / REJECT) ──▶ bot merges the top result ──▶ paid
```

## 1. Set up (no GPU)

```bash
git clone https://github.com/coderbench/bittrellis && cd bittrellis
pip install -e ".[dev]"
pytest -q                          # build → audit → lineage → tamper detection on tiny synthetic models
bittrellis quantizers              # runtime · baseline · unsloth · rtn
bittrellis inventory --out module_inventory.json
```

## 2. Know the competition

The frontier holds V0 (the shipped checkpoint) and every seed. Read
[`results/feasibility/`](../results/feasibility/feasibility_report.md) first:

- **A public fidelity gain is not enough.** Calibrated MLP bytes (V13) cut public RP-KL 11.5% but
  carried 9% of it to the private holdout, so they earn nothing; GDN FP8 (V3) carried 44%. Both fail.
- **What passes today is memory.** Q4_K maps keep quality within noise, save 0.4–1.8 GiB and clear
  the holdout — at up to 49% slower prompt reading.
- **MLP bytes guard math, the recurrent path guards long context** (public corpus). Tool calling is
  the worst category for every checkpoint.
- **The first layer of each kind picks the batched-prefill path:** changing layer 0 can cost 20–50% prefill.
- **Q4_K saves memory, costs prefill.** **Effects interact:** test combinations.

## 3. Write a manifest (or a quantizer)

```yaml
schema: bittrellis/manifest@2
track: HPC-01
name: calibrated-mlp-gdn-fp8-deep-qkv
description: >
  Calibrated MLP bytes, FP8 on the deepest GDN state-writing qkv; layer 0 stays NVFP4 for fast prefill.
authors: [your-handle]
default: NVFP4
rules:
  - match: "L*.mlp"
    layers: "0-55"
    format: NVFP4
    quantizer: unsloth
  - match: "L*.gdn.qkv"
    layers: "48-63"
    format: FP8
```

```bash
bittrellis manifest manifests/calibrated-mlp-gdn-fp8-deep-qkv.yaml --against manifests/
bittrellis search neighbors <manifest> --out proposals/
```

[Syntax](precision_manifest.md). A better encoder for an executed format is
often the biggest win: implement the [quantizer contract](quantizer_contract.md), submit it with one
manifest using it, and wait for maintainer review and `eval-approved`.

Open encoder work (epoch hpc01-e6):

- **Calibrated MLP encoders.** `bittrellis calibration fetch` downloads the pinned statistics (6.7 GB);
  `input_hessian(ctx, lin)` gives each layer's MLP gate/up input statistics for GPTQ-style rounding
  ([contract](quantizer_contract.md#calibration-statistics-epoch-hpc01-e6)). `nvfp4_gptq_mlp` also rounds
  `down_proj` by GPTQ, on input statistics it derives from the pinned gate/up statistics and the unit's own
  gate/up weights.
- **FP8 bytes.** Every FP8 tensor today is round-to-nearest (`rtn`); smarter per-row scales need no calibration.
- **Sequential encoders.** The audit replays `replay_mode: sequential` (error carried from earlier layers);
  no encoder uses it yet.

Check a gain against text you did not tune on before you submit: `bittrellis corpus build --split
public-validation --out data/corpus/validation.json` builds the other half of the public sources, then
`bittrellis reference --corpus data/corpus/validation.json --out data/reference/validation` and
`bittrellis evaluate-public <checkpoint> --corpus data/corpus/validation.json --reference data/reference/validation --out <dir>`.
It is not the holdout (anyone can rebuild it), but a gain that vanishes there will not carry over.

## 4. Measure locally (optional)

RTX 5090 host, ~120 GB disk, ~60 GB RAM. Your numbers help you iterate but are never scored.

```bash
scripts/setup_sparkinfer.sh                 # pinned runtime + fixed-partition scorer
scripts/setup_models.sh                     # base, shipped, unsloth; hash-verified
bittrellis reference                        # BF16 reference, once
bittrellis build manifests/mine.yaml --out models/candidates/mine
bittrellis evaluate models/candidates/mine --out artifacts/mine
bittrellis compare results/feasibility/artifacts/V0-baseline-rebuild artifacts/mine
bittrellis frontier --with-seeds artifacts/mine
```

## 5. Open a PR

One manifest per PR in `manifests/`, hypothesis in `description`. Don't touch the
[protected evaluator paths](../CONTRIBUTING.md).

## How it's scored

1. **Screen** (CPU): identical recipes, over-memory predictions and new quantizers reproducing an
   existing encoder's bytes are not measured; near-copies (≤ 2% of weights differ from another author's earlier PR)
   earn only what they add; an author's 4th+ waiting PR queues. [guards.md](guards.md)
2. **Audit:** legal formats and quantizers, lineage proven, frozen tensors untouched.
3. **Gates:** runtime correctness · RP-KL ≤ 0.30 and top-1 ≥ 0.80 · every BF16-retrievable needle at
   8K/16K/32K · no significant task loss vs V0 on all 784 questions, paired per question · private holdout PASS. Stops early: a
   quality-gate fail skips speed runs; dominated or below XS after them skips tasks and holdout.
4. **ε-frontier:** RP-KL ↓ · decode ↑ · 4K prefill ↑ · peak GPU memory ↓, noise-aware, against V0,
   seeds, accepted results and earlier open PRs by other authors. [frontier.md](frontier.md)
5. **FG-2 → tier:** dominated, invalid or duplicate: 0; else `eval:XL` (≥ 0.50%) … `eval:XS` (≥ 0.005%), counting only gains beyond noise; no holdout PASS, no tier.
6. **Merge:** the top open result gets `bt:merge-first` and the evaluator merges it on that pass;
   Gittensor pays the tier on the merge. [rewards.md](rewards.md)

## What earns nothing

- relabeling metadata or stored formats without changing execution
- Q4_K "quantizers" · changing frozen weights · `SPARKINFER_*` tricks (they are cleared)
- fitting the public corpus (holdout catches it)
- copying a PR, even with rewritten rules or a renamed encoder
- disk-size savings with no runtime effect · resubmitting a seed or an accepted candidate
