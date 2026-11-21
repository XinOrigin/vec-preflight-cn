"""Command-line interface for vec-preflight-cn."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

from .evidence import validate_evidence_bundle
from .skill_score import rescale_panel_to_skill
from .validator import validate_h5ad


def _load_veckit_scorer(veckit_dir: Path | None = None):
    if veckit_dir is not None:
        p = str(veckit_dir.resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import score_h5ad  # type: ignore
        return score_h5ad
    except ImportError as exc:
        raise RuntimeError(
            "veckit is required for `score`. Install it with `pip install veckit` "
            "or pass `--veckit-dir /path/to/veckit`."
        ) from exc


def _run_veckit(
    score_mod: Any,
    task: str,
    pred_path: Path,
    target_path: Path,
    ref_path: Path,
    *,
    seed: int = 0,
    allow_reorder: bool = False,
    setting: str = "heart",
) -> dict[str, Any]:
    ns = argparse.Namespace(
        task=task,
        input=pred_path,
        target=target_path,
        reference=ref_path if task in {"T1", "T2"} else None,
        wt=ref_path if task == "T3" else None,
        seed=seed,
        allow_reorder=allow_reorder,
        setting=setting,
    )
    if task == "T1":
        _, scores = score_mod.score_t1(ns)
    elif task == "T2":
        _, scores = score_mod.score_t2(ns)
    else:
        _, scores = score_mod.score_t3(ns)
    return scores


def run_rescaled_score(
    task: str,
    input_path: Path,
    target_path: Path,
    reference_path: Path,
    *,
    ceiling_target_path: Path | None = None,
    seed: int = 0,
    allow_reorder: bool = False,
    setting: str = "heart",
    veckit_dir: Path | None = None,
) -> dict[str, Any]:
    """Run veckit on pred, floor (copy_last / wt_identity), and split-half or replicate ceiling."""
    task = task.upper()
    score_mod = _load_veckit_scorer(veckit_dir)

    pred_metrics = _run_veckit(
        score_mod, task, input_path, target_path, reference_path,
        seed=seed, allow_reorder=allow_reorder, setting=setting,
    )
    floor_metrics = _run_veckit(
        score_mod, task, reference_path, target_path, reference_path,
        seed=seed, allow_reorder=allow_reorder, setting=setting,
    )

    if ceiling_target_path is not None:
        ceiling_mode = "replicate"
        ceiling_metrics = _run_veckit(
            score_mod, task, ceiling_target_path, target_path, reference_path,
            seed=seed, allow_reorder=allow_reorder, setting=setting,
        )
    else:
        ceiling_mode = "split_half"
        # Split-half attainable ceiling: disjoint half A of target scored against half B
        target_ad = ad.read_h5ad(target_path)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(target_ad.n_obs)
        mid = target_ad.n_obs // 2
        if mid < 2:
            raise ValueError(f"Target file needs at least 4 cells to compute split-half ceiling, got {target_ad.n_obs}")

        with tempfile.TemporaryDirectory() as tmp:
            half_a = Path(tmp) / "half_a.h5ad"
            half_b = Path(tmp) / "half_b.h5ad"
            target_ad[perm[:mid]].copy().write_h5ad(half_a)
            target_ad[perm[mid:]].copy().write_h5ad(half_b)
            ceiling_metrics = _run_veckit(
                score_mod, task, half_a, half_b, reference_path,
                seed=seed, allow_reorder=allow_reorder, setting=setting,
            )

    skill = rescale_panel_to_skill(task, pred_metrics, floor_metrics, ceiling_metrics)
    return {
        "task": task,
        "ceiling_mode": ceiling_mode,
        "skill": skill,
        "raw": {
            "prediction": pred_metrics,
            "floor": floor_metrics,
            "ceiling": ceiling_metrics,
        },
    }


def _print_check_human(report) -> None:
    banner = "PASS" if report.ok else "FAIL"
    print(f"[vec-preflight] {report.task} {banner} — {report.input_path}")
    for c in report.checks:
        print(f"  [{c.status:4s}] {c.name}: {c.message}")
    s = report.summary
    if s:
        print(
            f"Summary: {s['n_obs']} cells x {s['n_vars']} genes | "
            f"{s['pass_count']} PASS, {s['warn_count']} WARN, {s['fail_count']} FAIL"
        )


def _print_score_human(payload: dict[str, Any]) -> None:
    sk = payload["skill"]
    print(f"[vec-preflight score] {payload['task']} ({payload.get('ceiling_mode', 'split_half')}) "
          f"— Headline skill: {sk['headline_skill_100']:.2f} / 100 "
          f"(floor=50.00, beats_floor={sk['beats_floor']})")
    groups_str = ", ".join(
        f"{k}={'N/A(inv)' if v is None else f'{v:.2f}'}" for k, v in sk["group_scores_100"].items()
    )
    print("  Groups:", groups_str)
    print(f"  {'Metric':22s} {'Pred':>10s} {'Floor':>10s} {'Ceiling':>10s} {'Skill(0-100)':>13s} {'Anchor':>10s}")
    for name, row in sk["per_metric"].items():
        fmt = lambda v: "None" if v is None else f"{v:.5g}"
        sk_str = "N/A" if row["skill_100"] is None else f"{row['skill_100']:.2f}"
        print(
            f"  {name:22s} {fmt(row['raw']):>10s} {fmt(row['floor']):>10s} "
            f"{fmt(row['ceiling']):>10s} {sk_str:>13s} {row.get('anchor_status', 'ok'):>10s}"
        )
    for w in sk.get("warnings", []):
        print(f"  [WARN] {w}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vec-preflight",
        description="Virtual Embryo Challenge (NeurIPS 2026) pre-submission validator, "
                    "official floor/ceiling skill-score wrapper, and Agent Team evidence checker.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="Validate a candidate .h5ad submission before uploading.")
    p_check.add_argument("--task", required=True, choices=["T1", "T2", "T3"])
    p_check.add_argument("--input", required=True, type=Path, help="Candidate .h5ad file")
    p_check.add_argument("--reference", type=Path, help="Reference stage .h5ad (checks exact gene panel & scale)")
    p_check.add_argument("--allow-reorder", action="store_true", help="Treat gene order mismatch as WARN instead of FAIL")
    p_check.add_argument("--json", action="store_true", help="Print JSON output")

    p_score = sub.add_parser("score", help="Score via veckit and rescale onto the official 0..100 (floor=50) skill board.")
    p_score.add_argument("--task", required=True, choices=["T1", "T2", "T3"])
    p_score.add_argument("--input", required=True, type=Path)
    p_score.add_argument("--target", required=True, type=Path)
    p_score.add_argument("--reference", required=True, type=Path, help="Preceding stage (T1/T2) or matched WT (T3)")
    p_score.add_argument("--ceiling-target", type=Path, help="Optional biological replicate .h5ad for replicate ceiling instead of split-half")
    p_score.add_argument("--seed", type=int, default=0)
    p_score.add_argument("--allow-reorder", action="store_true")
    p_score.add_argument("--setting", default="heart", choices=["heart", "embryo"])
    p_score.add_argument("--veckit-dir", type=Path, help="Optional local path to cloned aristoteleo/veckit")
    p_score.add_argument("--json", action="store_true")

    p_ev = sub.add_parser("check-evidence", help="Validate an Agent Team (Track 2) evidence directory before upload.")
    p_ev.add_argument("--dir", required=True, type=Path, help="Directory holding trajectory, prompts, and/or harness files")
    p_ev.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "check":
        report = validate_h5ad(
            args.input, args.task, args.reference, allow_reorder=args.allow_reorder
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        else:
            _print_check_human(report)
        return 0 if report.ok else 1

    if args.cmd == "score":
        payload = run_rescaled_score(
            args.task,
            args.input,
            args.target,
            args.reference,
            ceiling_target_path=args.ceiling_target,
            seed=args.seed,
            allow_reorder=args.allow_reorder,
            setting=args.setting,
            veckit_dir=args.veckit_dir,
        )
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        else:
            _print_score_human(payload)
        return 0

    if args.cmd == "check-evidence":
        rep = validate_evidence_bundle(args.dir)
        if args.json:
            print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
        else:
            status = "PASS" if rep.ok else "FAIL"
            print(f"[vec-preflight check-evidence] {status} — {rep.directory}")
            for kind, files in rep.detected_kinds.items():
                print(f"  {kind:12s}: {files if files else '(none)'}")
            for w in rep.warnings:
                print(f"  [WARN] {w}")
            for err in rep.issues:
                print(f"  [FAIL] {err}")
        return 0 if rep.ok else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
