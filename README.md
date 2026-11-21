# vec-preflight-cn — Virtual Embryo Challenge (NeurIPS 2026) 提交通检工具 & 中文实战指南

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Bilingual Community Toolkit (中文 + English)** for the [Virtual Embryo Challenge (NeurIPS 2026)](https://virtualembryo.ai/challenge) hosted by the Qiu Lab at Stanford University.
>
> Designed to save teams from burning their scarce **P3 final submission quota (max 2 submissions per board)** on silent format, normalization, gene-panel, spatial-key, or Agent Team evidence mistakes — and to compute the **official `floor = 50 / ceiling = 100` rescaled skill scores** offline alongside [`aristoteleo/veckit`](https://github.com/aristoteleo/veckit).

📖 **完整中文入门与避坑指南**: [`docs/GUIDE_ZH.md`](docs/GUIDE_ZH.md)

---

## Why this toolkit exists / 解决什么痛点

1. **P3 final attempts are irreversible & capped at 2 per board.** Official `veckit` raises errors on hard failures, but silent traps — such as submitting **raw integer counts** instead of `log1p`-normalized expression (remember: the scorer **never** renormalizes submissions, and `sample_heart_9.25.h5ad` in `veckit/data` itself carries raw values up to `63.0` while `sample_heart_9.5.h5ad` is `log1p`-scaled up to `8.43`!), putting coordinates in `obsm["spatial"]` instead of `obsm["spatial_3D"]`, gene column reordering, or broadcasting a single mean cell — can silently ruin a submission.
2. **Raw `veckit` metrics are hard to compare across tasks.** `veckit` outputs raw metrics (`mmd_u`, `energy_distance`, `variogram`, `d2_shape`, etc.) without applying the official leaderboard rescaling (`floor = 50`, split-half `ceiling = 100`). `vec-preflight score` computes the floor baseline (`copy_last` / `wt_identity`), a disjoint split-half ceiling on your local pseudo-target, and the exact official `skill(m) = min(d_floor / (d_floor + d(m)), 1.0)` table.
3. **Track 2 (Agent Team) entries are held without verifiable evidence.** Per Official Rules §8 & §14, Agent Team submissions require at least **two distinct evidence kinds** from `{trajectory, prompts, harness}` before entering the scoring queue. `vec-preflight check-evidence` checks structure, JSON/JSONL integrity, and accidental API-key/token leaks before upload.
4. **First complete Chinese walkthrough grounded in `veckit` source & official P3 rules.** Covers T1/T2/T3 definitions, exact metric formulas, P3 validation-unlock rules, and the top 8 real pitfalls.

---

## Quickstart / 快速上手

```bash
git clone https://github.com/XinOrigin/vec-preflight-cn.git
cd vec-preflight-cn
pip install -e .
# Optional: install veckit if you want `vec-preflight score`
pip install veckit
```

### 1. Pre-submission `.h5ad` validation / 提交前 `.h5ad` 格式与尺度体检

```bash
# Task 1 (scRNA-seq, 32,285 genes)
vec-preflight check --task T1 --input pred_t1.h5ad --reference E9.5.h5ad

# Task 2 (3D MERFISH, 500 genes, requires obsm['spatial_3D'])
vec-preflight check --task T2 --input pred_t2.h5ad --reference heart_E9.5.h5ad

# Task 3 (Perturbation KO, 500 genes, requires obsm['spatial_3D'])
vec-preflight check --task T3 --input pred_t3.h5ad --reference wt.h5ad
```

Example output catching a raw-count matrix (`sample_heart_9.25.h5ad`, `max=63.0`) against `log1p`-scaled `sample_heart_9.5.h5ad`:

```text
[vec-preflight] T2 PASS — sample_heart_9.25.h5ad
  [PASS] h5ad_readable: Loaded AnnData with 150 cells x 500 genes.
  [PASS] matrix_2d: prediction.X is 2D float32 [150, 500].
  [PASS] matrix_finite: All entries in prediction.X are finite.
  [PASS] matrix_nonnegative: prediction.X is nonnegative (min=0, max=63).
  [WARN] normalization_scale: prediction.X looks like raw counts (max=63.00, 97.6% integer nonzeros). The VEC scorer NEVER renormalizes submissions...
  [PASS] gene_panel_alignment: Exact gene panel and order match (500 genes).
  [WARN] library_size_vs_reference: Mean library size ratio vs reference is 0.184 (pred=35.78, ref=194.53)...
  [PASS] spatial_3D_valid: obsm['spatial_3D'] valid with shape [150, 3]...
```

### 2. Offline official `floor=50 / ceiling=100` skill score / 离线计算官方 0–100 标度分

```bash
vec-preflight score \
  --task T1 \
  --input pred.h5ad \
  --target local_pseudo_target.h5ad \
  --reference preceding_stage.h5ad \
  # Optional: pass a held-out biological replicate instead of split-half ceiling:
  # --ceiling-target replicate_target.h5ad
```

*(Pass `--veckit-dir /path/to/veckit` if running against a local git clone of `aristoteleo/veckit` without `pip install veckit`. On tiny toy subsamples `<500` cells where split-half variance makes a metric ceiling worse than the floor, `vec-preflight` automatically flags `anchor_status=inverted` and excludes that metric from rescaled headline skill rather than falsely inflating the floor to 100.)*

> **Important reminder:** Local scoring uses a pseudo-target from your released training data; it verifies pipeline correctness and relative gains over `copy_last`/`wt_identity`, not your final score on the hidden held-out test set.

### 3. Track 2 (Agent Team) evidence pre-check / Agent 赛道证据包检查

```bash
vec-preflight check-evidence --dir ./my_agent_evidence
```

Checks:
- At least 2 of `{trajectory, prompts, harness}` detected and non-empty
- Valid UTF-8 / JSON / JSONL syntax
- Secret-leak scan (OpenAI / Anthropic / GitHub / AWS key patterns)

---

## Running tests / 运行测试

```bash
pip install -e ".[dev]"
pytest -v
```

---

## License

MIT — see [`LICENSE`](LICENSE). Independent community resource for the Virtual Embryo Challenge; not affiliated with Stanford Qiu Lab.
