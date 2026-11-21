"""Tests for vec-preflight-cn covering validator, skill-score math, evidence checks, and real veckit samples."""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from vec_preflight.cli import main as cli_main
from vec_preflight.cli import run_rescaled_score
from vec_preflight.evidence import validate_evidence_bundle
from vec_preflight.skill_score import compute_metric_skill, rescale_panel_to_skill
from vec_preflight.validator import validate_h5ad

VECKIT_DIR = Path(__file__).resolve().parents[2] / "veckit"


def _make_h5ad(
    path: Path,
    X: np.ndarray,
    genes: list[str] | None = None,
    *,
    spatial_key: str | None = None,
    coords: np.ndarray | None = None,
) -> Path:
    n_obs, n_vars = X.shape
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.var_names = genes if genes is not None else [f"G{i}" for i in range(n_vars)]
    adata.obs["celltype"] = [f"ct_{i % 3}" for i in range(n_obs)]
    if spatial_key is not None and coords is not None:
        adata.obsm[spatial_key] = coords.astype(np.float32)
    adata.write_h5ad(path)
    return path


def test_validator_passes_clean_t1_and_t2(tmp_path: Path):
    rng = np.random.default_rng(42)
    ref_x = np.log1p(rng.poisson(2.0, size=(30, 12)).astype(np.float32))
    pred_x = np.log1p(rng.poisson(2.2, size=(30, 12)).astype(np.float32))
    coords = rng.normal(size=(30, 3)).astype(np.float32)

    ref_p = _make_h5ad(tmp_path / "ref.h5ad", ref_x, spatial_key="spatial_3D", coords=coords)
    pred_p = _make_h5ad(tmp_path / "pred.h5ad", pred_x, spatial_key="spatial_3D", coords=coords + 0.1)

    r1 = validate_h5ad(pred_p, "T1", ref_p)
    assert r1.ok, r1.to_dict()
    assert r1.summary["fail_count"] == 0

    r2 = validate_h5ad(pred_p, "T2", ref_p)
    assert r2.ok, r2.to_dict()
    assert any(c.name == "spatial_3D_valid" and c.status == "PASS" for c in r2.checks)


def test_validator_rejects_negative_nan_collapsed_and_wrong_spatial_key(tmp_path: Path):
    rng = np.random.default_rng(7)
    bad_x = rng.normal(size=(20, 8)).astype(np.float32)  # has negatives
    bad_x[0, 0] = np.nan
    coords = rng.normal(size=(20, 3)).astype(np.float32)
    p = _make_h5ad(tmp_path / "bad.h5ad", bad_x, spatial_key="spatial", coords=coords)

    rep = validate_h5ad(p, "T2")
    assert not rep.ok
    failed = {c.name for c in rep.checks if c.status == "FAIL"}
    assert {"matrix_finite", "matrix_nonnegative", "spatial_3D_present"} <= failed

    # Collapsed population (identical cells) must fail
    const_x = np.ones((15, 8), dtype=np.float32)
    p_const = _make_h5ad(tmp_path / "const.h5ad", const_x)
    rep_const = validate_h5ad(p_const, "T1")
    assert not rep_const.ok
    assert any(c.name == "population_variance" and c.status == "FAIL" for c in rep_const.checks)


def test_validator_detects_raw_counts_and_gene_order_mismatch(tmp_path: Path):
    rng = np.random.default_rng(1)
    genes = [f"G{i}" for i in range(10)]
    ref_x = np.log1p(rng.integers(0, 50, size=(25, 10)).astype(np.float32))
    raw_x = rng.integers(0, 80, size=(25, 10)).astype(np.float32)
    raw_x[0, 0] = 63.0

    ref_p = _make_h5ad(tmp_path / "ref.h5ad", ref_x, genes=genes)
    raw_p = _make_h5ad(tmp_path / "raw.h5ad", raw_x, genes=list(reversed(genes)))

    rep_strict = validate_h5ad(raw_p, "T1", ref_p, allow_reorder=False)
    assert not rep_strict.ok
    assert any(c.name == "gene_panel_alignment" and c.status == "FAIL" for c in rep_strict.checks)
    assert any(c.name == "normalization_scale" and c.status == "WARN" for c in rep_strict.checks)

    rep_reord = validate_h5ad(raw_p, "T1", ref_p, allow_reorder=True)
    assert rep_reord.ok
    assert any(c.name == "gene_panel_alignment" and c.status == "WARN" for c in rep_reord.checks)


def test_skill_score_anchors_and_edge_cases():
    # Higher-is-better: floor=0.0, ceiling=0.8 -> value=0.0 gives 0.5, value=0.8 gives 1.0, >ceiling clips to 1.0
    assert compute_metric_skill(0.0, 0.0, 0.8, direction="higher") == pytest.approx(0.5)
    assert compute_metric_skill(0.8, 0.0, 0.8, direction="higher") == pytest.approx(1.0)
    assert compute_metric_skill(0.95, 0.0, 0.8, direction="higher") == pytest.approx(1.0)
    assert compute_metric_skill(None, 0.0, 0.8, direction="higher") == 0.0
    assert compute_metric_skill(float("nan"), 0.0, 0.8, direction="higher") == 0.0

    # Lower-is-better with take_abs (signed log ratios): -0.2 and +0.2 are equally distant from 0
    assert compute_metric_skill(-0.2, 0.2, 0.0, direction="lower", take_abs=True) == pytest.approx(0.5)
    assert compute_metric_skill(0.0, 0.2, 0.0, direction="lower", take_abs=True) == pytest.approx(1.0)

    # Panel rescaling: floor itself must score 50.00 across all T1 groups
    floor = {"de_score": 0.0, "de_direction": 0.0, "mmd_u": 0.10, "energy_distance": 0.40, "variogram": 0.005}
    ceil = {"de_score": 0.6, "de_direction": 0.8, "mmd_u": 0.01, "energy_distance": 0.05, "variogram": 0.001}
    res_floor = rescale_panel_to_skill("T1", floor, floor, ceil)
    assert res_floor["headline_skill_100"] == pytest.approx(50.0)
    assert res_floor["beats_floor"] is False
    assert res_floor["warnings"] == []
    res_ceil = rescale_panel_to_skill("T1", ceil, floor, ceil)
    assert res_ceil["headline_skill_100"] == pytest.approx(100.0)
    assert res_ceil["beats_floor"] is True

    # Inverted ceiling (e.g. tiny subsample where split-half mmd_u > floor mmd_u) is flagged and excluded
    ceil_inv = {**ceil, "mmd_u": 0.25}
    res_inv = rescale_panel_to_skill("T1", floor, floor, ceil_inv)
    assert res_inv["per_metric"]["mmd_u"]["anchor_status"] == "inverted"
    assert res_inv["per_metric"]["mmd_u"]["skill_100"] is None
    assert res_inv["headline_skill_100"] == pytest.approx(50.0)
    assert len(res_inv["warnings"]) == 1


def test_evidence_bundle_validation(tmp_path: Path):
    # Only 1 kind -> FAIL
    d1 = tmp_path / "ev_one"
    d1.mkdir()
    (d1 / "prompts.txt").write_text("Initial user prompt\n", encoding="utf-8")
    r1 = validate_evidence_bundle(d1)
    assert not r1.ok

    # 2 valid kinds -> PASS with warning about optional 3rd
    (d1 / "trajectory.jsonl").write_text('{"step": 1, "action": "train"}\n', encoding="utf-8")
    r2 = validate_evidence_bundle(d1)
    assert r2.ok
    assert len(r2.warnings) == 1

    # Secret leak -> FAIL
    (d1 / "harness_run.py").write_text('KEY = "ghp_1234567890abcdefghij1234567890"\n', encoding="utf-8")
    r3 = validate_evidence_bundle(d1)
    assert not r3.ok
    assert any("github_pat" in msg for msg in r3.issues)


@pytest.mark.skipif(not (VECKIT_DIR / "data" / "sample_8.5.h5ad").exists(), reason="local veckit samples absent")
def test_real_veckit_samples_preflight_and_rescaled_score(capsys):
    data = VECKIT_DIR / "data"
    # T1 clean sample
    r_t1 = validate_h5ad(data / "sample_8.5.h5ad", "T1", data / "sample_9.5.h5ad")
    assert r_t1.ok

    # T2 raw-count sample_heart_9.25 vs log1p sample_heart_9.5 must trigger normalization WARN
    r_t2 = validate_h5ad(data / "sample_heart_9.25.h5ad", "T2", data / "sample_heart_9.5.h5ad")
    assert r_t2.ok
    assert any(c.name == "normalization_scale" and c.status == "WARN" for c in r_t2.checks)

    # End-to-end rescaled score of copy_last (sample_8.5) must read exactly 50.00 headline skill
    # and surface the inverted split-half ceiling warning on the tiny toy sample
    scored = run_rescaled_score(
        "T1",
        data / "sample_8.5.h5ad",
        data / "sample_9.5.h5ad",
        data / "sample_8.5.h5ad",
        seed=0,
        veckit_dir=VECKIT_DIR,
    )
    assert scored["skill"]["headline_skill_100"] == pytest.approx(50.0, abs=0.05)
    assert scored["skill"]["beats_floor"] is False
    assert len(scored["skill"]["warnings"]) == 1

    # CLI JSON smoke test
    rc = cli_main(["check", "--task", "T3", "--input", str(data / "sample_wt.h5ad"), "--reference", str(data / "sample_mab21l2_ko.h5ad"), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
