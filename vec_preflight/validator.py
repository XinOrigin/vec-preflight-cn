"""Pre-submission .h5ad validator for the Virtual Embryo Challenge (T1 / T2 / T3).

Checks every hard schema requirement enforced by ``aristoteleo/veckit`` plus
high-frequency silent failure modes (raw un-normalized counts, wrong spatial
obsm key, reordered gene panel, collapsed single-cell broadcast) before a team
spends one of its limited P3 final submissions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse


TASKS_REQUIRING_SPATIAL_3D = {"T2", "T3"}


@dataclass
class CheckItem:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightReport:
    task: str
    input_path: str
    reference_path: str | None
    ok: bool
    checks: list[CheckItem] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "input_path": self.input_path,
            "reference_path": self.reference_path,
            "ok": self.ok,
            "summary": self.summary,
            "checks": [asdict(c) for c in self.checks],
        }


def _to_dense_float32(X: Any) -> np.ndarray:
    arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
    return arr.astype(np.float32, copy=False)


def _nonzero_integer_fraction(X: np.ndarray) -> float:
    nz = X[X > 0]
    if nz.size == 0:
        return 0.0
    sample = nz[:500_000]
    return float(np.mean(np.isclose(sample, np.round(sample), atol=1e-5)))


def validate_h5ad(
    input_path: str | Path,
    task: str,
    reference_path: str | Path | None = None,
    *,
    allow_reorder: bool = False,
) -> PreflightReport:
    """Validate a candidate .h5ad submission offline against VEC T1/T2/T3 contracts."""
    task = task.upper()
    if task not in {"T1", "T2", "T3"}:
        raise ValueError(f"Unsupported task {task!r}; expected T1, T2, or T3")

    in_path = Path(input_path)
    ref_path = Path(reference_path) if reference_path is not None else None
    checks: list[CheckItem] = []

    if not in_path.exists():
        checks.append(CheckItem("file_exists", "FAIL", f"Input file not found: {in_path}"))
        return PreflightReport(task, str(in_path), str(ref_path) if ref_path else None, False, checks)

    try:
        adata = ad.read_h5ad(in_path)
    except Exception as exc:
        checks.append(CheckItem("h5ad_readable", "FAIL", f"Failed to read .h5ad file: {exc}"))
        return PreflightReport(task, str(in_path), str(ref_path) if ref_path else None, False, checks)

    checks.append(
        CheckItem(
            "h5ad_readable",
            "PASS",
            f"Loaded AnnData with {adata.n_obs} cells x {adata.n_vars} genes.",
            {"n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars)},
        )
    )

    if adata.n_obs == 0 or adata.n_vars == 0:
        checks.append(CheckItem("non_empty", "FAIL", f"Matrix must be non-empty, got shape {adata.shape}."))
        return PreflightReport(task, str(in_path), str(ref_path) if ref_path else None, False, checks)

    # 1. Expression matrix shape, finiteness, and non-negativity
    X = _to_dense_float32(adata.X)
    if X.ndim != 2:
        checks.append(CheckItem("matrix_2d", "FAIL", f"prediction.X must be 2D, got ndim={X.ndim}."))
    else:
        checks.append(CheckItem("matrix_2d", "PASS", f"prediction.X is 2D float32 {list(X.shape)}."))

    if not np.isfinite(X).all():
        n_bad = int(np.size(X) - np.isfinite(X).sum())
        checks.append(
            CheckItem(
                "matrix_finite",
                "FAIL",
                f"prediction.X contains {n_bad} NaN/Inf values; veckit rejects non-finite entries.",
                {"non_finite_count": n_bad},
            )
        )
    else:
        checks.append(CheckItem("matrix_finite", "PASS", "All entries in prediction.X are finite."))

    x_min = float(np.nanmin(X)) if X.size else 0.0
    x_max = float(np.nanmax(X)) if X.size else 0.0
    if x_min < 0.0:
        n_neg = int((X < 0).sum())
        checks.append(
            CheckItem(
                "matrix_nonnegative",
                "FAIL",
                f"prediction.X has {n_neg} negative values (min={x_min:.4g}); "
                "metrics expect nonnegative log-normalized expression (not z-scored).",
                {"min": x_min, "negative_count": n_neg},
            )
        )
    else:
        checks.append(
            CheckItem(
                "matrix_nonnegative",
                "PASS",
                f"prediction.X is nonnegative (min={x_min:.4g}, max={x_max:.4g}).",
                {"min": x_min, "max": x_max},
            )
        )

    # 2. Scale / raw-count heuristic (veckit never renormalizes submissions)
    int_frac = _nonzero_integer_fraction(X)
    lib_mean = float(np.mean(X.sum(axis=1)))
    if x_max > 20.0 and int_frac > 0.95:
        checks.append(
            CheckItem(
                "normalization_scale",
                "WARN",
                f"prediction.X looks like raw counts (max={x_max:.2f}, {int_frac*100:.1f}% integer nonzeros). "
                "The VEC scorer NEVER renormalizes submissions; apply log1p normalization matching the reference stage "
                "or pb_rel_err / library_size_ratio will fail.",
                {"max": x_max, "nonzero_integer_fraction": round(int_frac, 4), "mean_library_size": round(lib_mean, 3)},
            )
        )
    else:
        checks.append(
            CheckItem(
                "normalization_scale",
                "PASS",
                f"Expression scale looks continuous/log-normalized (max={x_max:.3f}, mean row sum={lib_mean:.2f}).",
                {"max": x_max, "nonzero_integer_fraction": round(int_frac, 4), "mean_library_size": round(lib_mean, 3)},
            )
        )

    # 3. Population collapse check (catches single-cell broadcast / mean collapse)
    gene_std = X.std(axis=0)
    mean_std = float(np.mean(gene_std))
    active_genes = int((gene_std > 1e-6).sum())
    if adata.n_obs > 1 and active_genes == 0:
        checks.append(
            CheckItem(
                "population_variance",
                "FAIL",
                "All predicted cells are identical (zero variance across cells); "
                "variance_ratio will read 0 and collapse distributional metrics.",
                {"mean_gene_std": mean_std, "variable_genes": active_genes},
            )
        )
    elif adata.n_obs > 1 and mean_std < 1e-4:
        checks.append(
            CheckItem(
                "population_variance",
                "WARN",
                f"Very low cell-to-cell variance (mean gene std={mean_std:.2e}); check for mean-collapse.",
                {"mean_gene_std": mean_std, "variable_genes": active_genes},
            )
        )
    else:
        checks.append(
            CheckItem(
                "population_variance",
                "PASS",
                f"Non-degenerate cell population ({active_genes}/{adata.n_vars} genes vary across cells).",
                {"mean_gene_std": round(mean_std, 6), "variable_genes": active_genes},
            )
        )

    # 4. Duplicate var_names check
    pred_genes = [str(g) for g in adata.var_names]
    if len(set(pred_genes)) != len(pred_genes):
        checks.append(
            CheckItem(
                "unique_var_names",
                "FAIL",
                f"var_names contains {len(pred_genes) - len(set(pred_genes))} duplicate gene names.",
            )
        )
    else:
        checks.append(CheckItem("unique_var_names", "PASS", f"All {len(pred_genes)} var_names are unique."))

    # 5. Reference gene-panel alignment and scale comparison
    ref_adata = None
    if ref_path is not None:
        if not ref_path.exists():
            checks.append(CheckItem("reference_exists", "FAIL", f"Reference file not found: {ref_path}"))
        else:
            ref_adata = ad.read_h5ad(ref_path)
            ref_genes = [str(g) for g in ref_adata.var_names]
            if pred_genes == ref_genes:
                checks.append(
                    CheckItem(
                        "gene_panel_alignment",
                        "PASS",
                        f"Exact gene panel and order match ({len(ref_genes)} genes).",
                        {"expected_genes": len(ref_genes)},
                    )
                )
            elif set(pred_genes) == set(ref_genes) and len(pred_genes) == len(ref_genes):
                status = "WARN" if allow_reorder else "FAIL"
                checks.append(
                    CheckItem(
                        "gene_panel_alignment",
                        status,
                        "Same gene set as reference, but column order differs. "
                        "Reorder columns with `adata[:, ref.var_names].copy()` before submission.",
                        {"expected_genes": len(ref_genes), "order_mismatch": True},
                    )
                )
            else:
                pred_set, ref_set = set(pred_genes), set(ref_genes)
                missing = [g for g in ref_genes if g not in pred_set][:10]
                extra = [g for g in pred_genes if g not in ref_set][:10]
                checks.append(
                    CheckItem(
                        "gene_panel_alignment",
                        "FAIL",
                        f"Gene panel mismatch: expected {len(ref_genes)}, got {len(pred_genes)}. "
                        f"missing_head={missing}, extra_head={extra}.",
                        {"expected_genes": len(ref_genes), "got_genes": len(pred_genes), "missing_head": missing, "extra_head": extra},
                    )
                )

            ref_X = _to_dense_float32(ref_adata.X)
            ref_lib_mean = float(np.mean(ref_X.sum(axis=1)))
            if ref_lib_mean > 0:
                ratio = lib_mean / ref_lib_mean
                if ratio < 0.2 or ratio > 5.0:
                    checks.append(
                        CheckItem(
                            "library_size_vs_reference",
                            "WARN",
                            f"Mean library size ratio vs reference is {ratio:.3f} "
                            f"(pred={lib_mean:.2f}, ref={ref_lib_mean:.2f}); verify normalization matches reference.",
                            {"ratio": round(ratio, 4), "pred_mean": round(lib_mean, 3), "ref_mean": round(ref_lib_mean, 3)},
                        )
                    )
                else:
                    checks.append(
                        CheckItem(
                            "library_size_vs_reference",
                            "PASS",
                            f"Mean library size ratio vs reference is {ratio:.3f} (within [0.2, 5.0]).",
                            {"ratio": round(ratio, 4)},
                        )
                    )

    # 6. Spatial 3D coordinates check for T2 / T3
    if task in TASKS_REQUIRING_SPATIAL_3D:
        if "spatial_3D" not in adata.obsm:
            alt_keys = [k for k in ("spatial", "spatial_2D", "X_spatial") if k in adata.obsm]
            hint = f" Found {alt_keys} instead — rename/provide 3D coords in obsm['spatial_3D']." if alt_keys else ""
            checks.append(
                CheckItem(
                    "spatial_3D_present",
                    "FAIL",
                    f"{task} requires obsm['spatial_3D'], which is missing.{hint}",
                    {"available_obsm": list(adata.obsm.keys())},
                )
            )
        else:
            C = np.asarray(adata.obsm["spatial_3D"], dtype=np.float32)
            if C.ndim != 2 or C.shape[0] != adata.n_obs or C.shape[1] < 3:
                checks.append(
                    CheckItem(
                        "spatial_3D_shape",
                        "FAIL",
                        f"obsm['spatial_3D'] must have shape ({adata.n_obs}, >=3), got {list(C.shape)}.",
                    )
                )
            elif not np.isfinite(C[:, :3]).all():
                checks.append(
                    CheckItem(
                        "spatial_3D_finite",
                        "FAIL",
                        "obsm['spatial_3D'][:, :3] contains NaN or Inf values.",
                    )
                )
            else:
                spread = C[:, :3].std(axis=0)
                if float(np.max(spread)) < 1e-6:
                    checks.append(
                        CheckItem(
                            "spatial_3D_spread",
                            "FAIL",
                            "obsm['spatial_3D'] coordinates collapsed to a single point (zero spread).",
                            {"axis_std": [round(float(s), 6) for s in spread]},
                        )
                    )
                else:
                    checks.append(
                        CheckItem(
                            "spatial_3D_valid",
                            "PASS",
                            f"obsm['spatial_3D'] valid with shape {list(C.shape)} and axis std {[round(float(s), 4) for s in spread]}.",
                            {"shape": list(C.shape), "axis_std": [round(float(s), 4) for s in spread]},
                        )
                    )

    # 7. Cell-type metadata advisory
    if "celltype" in adata.obs:
        n_ct = int(adata.obs["celltype"].astype(str).nunique())
        checks.append(
            CheckItem(
                "obs_celltype",
                "PASS",
                f"obs['celltype'] present ({n_ct} unique labels). Note: official VEC scorer uses a frozen truth probe, not submitted labels.",
                {"unique_celltypes": n_ct},
            )
        )
    else:
        checks.append(
            CheckItem(
                "obs_celltype",
                "PASS",
                "obs['celltype'] omitted (allowed; official VEC scorer predicts cell types via a frozen truth-trained probe).",
            )
        )

    ok = all(c.status != "FAIL" for c in checks)
    summary = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "pass_count": sum(c.status == "PASS" for c in checks),
        "warn_count": sum(c.status == "WARN" for c in checks),
        "fail_count": sum(c.status == "FAIL" for c in checks),
    }
    return PreflightReport(
        task=task,
        input_path=str(in_path),
        reference_path=str(ref_path) if ref_path else None,
        ok=ok,
        checks=checks,
        summary=summary,
    )
