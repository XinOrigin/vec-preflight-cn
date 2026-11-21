"""Official Virtual Embryo Challenge floor/ceiling skill-score rescaling.

Implements the exact rescaling published at
``https://virtualembryo.ai/challenge/evaluation?section=scoring``:

    d(m)      = (m - m_ceiling) if lower_is_better else (m_ceiling - m)
    d_floor   = |m_ceiling - m_floor|
    skill(m)  = min(d_floor / (d_floor + d(m)), 1.0)   (x 100 on the 0..100 board)

Signed log-ratio terms (``scale_log_ratio``, ``count_log_ratio``) are evaluated
on ``abs(m)`` as lower-is-better terms. Missing or non-finite values score 0.0.
At the floor, ``skill = 0.5`` (50 points); at or beyond the split-half ceiling,
``skill = 1.0`` (100 points).
"""
from __future__ import annotations

import math
from typing import Any

# Direction & absolute-value semantics matching T1/T2/T3 metrics_v2.py
METRIC_SPECS: dict[str, tuple[str, bool]] = {
    # metric_name: (direction: "higher" | "lower", take_abs_first)
    "de_score": ("higher", False),
    "de_direction": ("higher", False),
    "severity_slope": ("higher", False),
    "energy_distance": ("lower", False),
    "mmd_u": ("lower", False),
    "variogram": ("lower", False),
    "d2_shape": ("lower", False),
    "sliced_wasserstein": ("lower", False),
    "occupancy_dice": ("higher", False),
    "scale_log_ratio": ("lower", True),
    "count_log_ratio": ("lower", True),
    "neighborhood_mmd": ("lower", False),
}

# Question-group weights for Task 1 (published on virtualembryo.ai/challenge/evaluation)
T1_QUESTION_GROUPS: list[tuple[str, float, list[str]]] = [
    ("de_gene_recovery", 0.25, ["de_score"]),
    ("change_direction", 0.25, ["de_direction"]),
    ("cell_state_distribution", 0.30, ["mmd_u", "energy_distance"]),
    ("gene_gene_covariation", 0.20, ["variogram"]),
]

# Equal-question groupings for T2/T3 local diagnostic summaries
T2_QUESTION_GROUPS: list[tuple[str, float, list[str]]] = [
    ("expression_de", 0.25, ["de_score", "de_direction"]),
    ("expression_distribution", 0.25, ["mmd_u", "energy_distance", "variogram"]),
    ("spatial_shape", 0.25, ["d2_shape", "sliced_wasserstein", "occupancy_dice"]),
    ("growth_and_local", 0.25, ["scale_log_ratio", "count_log_ratio", "neighborhood_mmd"]),
]

T3_QUESTION_GROUPS: list[tuple[str, float, list[str]]] = [
    ("perturbation_response", 0.40, ["de_score", "de_direction", "severity_slope"]),
    ("mutant_distribution", 0.30, ["mmd_u", "variogram"]),
    ("spatial_shape", 0.30, ["d2_shape", "sliced_wasserstein", "occupancy_dice", "scale_log_ratio", "count_log_ratio", "neighborhood_mmd"]),
]

TASK_GROUPS = {"T1": T1_QUESTION_GROUPS, "T2": T2_QUESTION_GROUPS, "T3": T3_QUESTION_GROUPS}


def classify_anchor(
    floor_val: float | None,
    ceiling_val: float | None,
    *,
    direction: str,
    take_abs: bool = False,
) -> str:
    """Classify the floor-vs-ceiling anchor pair as 'ok', 'inverted', 'degenerate', or 'missing'."""
    if floor_val is None or ceiling_val is None:
        return "missing"
    if not (math.isfinite(floor_val) and math.isfinite(ceiling_val)):
        return "missing"
    mf = abs(floor_val) if take_abs else float(floor_val)
    mc = abs(ceiling_val) if take_abs else float(ceiling_val)
    if abs(mc - mf) <= 1e-12:
        return "degenerate"
    if direction == "lower" and mc > mf:
        return "inverted"
    if direction == "higher" and mc < mf:
        return "inverted"
    return "ok"


def compute_metric_skill(
    value: float | None,
    floor_val: float | None,
    ceiling_val: float | None,
    *,
    direction: str,
    take_abs: bool = False,
) -> float:
    """Compute a single metric's rescaled skill in [0, 1] (floor=0.5, ceiling=1.0)."""
    if value is None or floor_val is None or ceiling_val is None:
        return 0.0
    if not (math.isfinite(value) and math.isfinite(floor_val) and math.isfinite(ceiling_val)):
        return 0.0

    m = abs(value) if take_abs else float(value)
    mf = abs(floor_val) if take_abs else float(floor_val)
    mc = abs(ceiling_val) if take_abs else float(ceiling_val)

    d_floor = abs(mc - mf)
    if d_floor <= 1e-12:
        # Degenerate anchor where floor == ceiling on a tiny toy split.
        better_or_eq = (m <= mc) if direction == "lower" else (m >= mc)
        return 1.0 if better_or_eq else 0.0

    d_m = (m - mc) if direction == "lower" else (mc - m)
    if d_m <= 0.0:
        return 1.0
    denom = d_floor + d_m
    if denom <= 0.0:
        return 1.0
    return min(d_floor / denom, 1.0)


def rescale_panel_to_skill(
    task: str,
    pred_metrics: dict[str, Any],
    floor_metrics: dict[str, Any],
    ceiling_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Rescale raw veckit metrics against floor and split-half/replicate ceiling anchors (0..100 scale)."""
    task = task.upper()
    if task not in TASK_GROUPS:
        raise ValueError(f"Unsupported task {task!r}")

    per_metric: dict[str, dict[str, Any]] = {}
    inverted_metrics: list[str] = []
    for name, (direction, take_abs) in METRIC_SPECS.items():
        if name not in pred_metrics and name not in floor_metrics:
            continue
        status = classify_anchor(
            floor_metrics.get(name),
            ceiling_metrics.get(name),
            direction=direction,
            take_abs=take_abs,
        )
        if status == "inverted":
            inverted_metrics.append(name)
            skill_100: float | None = None
        else:
            sk = compute_metric_skill(
                pred_metrics.get(name),
                floor_metrics.get(name),
                ceiling_metrics.get(name),
                direction=direction,
                take_abs=take_abs,
            )
            skill_100 = round(sk * 100.0, 2)
        per_metric[name] = {
            "raw": pred_metrics.get(name),
            "floor": floor_metrics.get(name),
            "ceiling": ceiling_metrics.get(name),
            "anchor_status": status,
            "skill_100": skill_100,
        }

    warnings: list[str] = []
    if inverted_metrics:
        warnings.append(
            f"Inverted ceiling anchor on {', '.join(inverted_metrics)} (ceiling worse than floor). "
            "Excluded from rescaled headline skill to avoid inflating floor=50 to 100. "
            "This happens on small subsamples (<500 cells) where split-half variance exceeds stage shift; "
            "evaluate on full-size embryos or pass --ceiling-target <replicate.h5ad>."
        )

    group_scores: dict[str, float | None] = {}
    headline = 0.0
    total_weight = 0.0
    for group_name, weight, members in TASK_GROUPS[task]:
        present = [
            per_metric[m]["skill_100"]
            for m in members
            if m in per_metric and per_metric[m]["skill_100"] is not None
        ]
        if not present:
            if any(m in per_metric for m in members):
                group_scores[group_name] = None
            continue
        g_score = sum(present) / len(present)
        group_scores[group_name] = round(g_score, 2)
        headline += weight * g_score
        total_weight += weight

    headline_score = round(headline / total_weight, 2) if total_weight > 0 else 0.0
    return {
        "task": task,
        "headline_skill_100": headline_score,
        "beats_floor": headline_score > 50.0,
        "group_scores_100": group_scores,
        "per_metric": per_metric,
        "warnings": warnings,
    }
