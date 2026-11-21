"""Agent Team evidence bundle validator for Virtual Embryo Challenge Track 2.

Per Official Rules §8 and §14, Agent Team submissions are held until at least
TWO distinct evidence kinds from {trajectory, prompts, harness} are attached,
and prize eligibility requires verifiable agent-authored artifacts.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EVIDENCE_PATTERNS: dict[str, tuple[str, ...]] = {
    "trajectory": ("trajectory", "transcript", "session", "trace", "log"),
    "prompts": ("prompt", "prompts", "instruction"),
    "harness": ("harness", "runner", "agent", "workflow", "config"),
}

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("github_pat", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
]


@dataclass
class EvidenceReport:
    directory: str
    ok: bool
    detected_kinds: dict[str, list[str]] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify_file(path: Path) -> str | None:
    low = path.name.lower()
    for kind, tokens in EVIDENCE_PATTERNS.items():
        if any(tok in low for tok in tokens):
            return kind
    return None


def validate_evidence_bundle(directory: str | Path) -> EvidenceReport:
    """Inspect a local directory of Agent Team evidence files before uploading."""
    root = Path(directory)
    detected: dict[str, list[str]] = {"trajectory": [], "prompts": [], "harness": []}
    issues: list[str] = []
    warnings: list[str] = []

    if not root.exists() or not root.is_dir():
        issues.append(f"Evidence directory not found: {root}")
        return EvidenceReport(str(root), False, detected, issues, warnings)

    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    if not files:
        issues.append("Evidence directory is empty; attach at least two distinct evidence kinds (trajectory, prompts, harness).")
        return EvidenceReport(str(root), False, detected, issues, warnings)

    for f in files:
        rel = str(f.relative_to(root))
        size = f.stat().st_size
        if size == 0:
            issues.append(f"Empty evidence file: {rel}")
            continue

        kind = _classify_file(f)
        if kind is not None:
            detected[kind].append(rel)

        # Validate UTF-8 readability and JSON/JSONL structure where applicable
        if f.suffix.lower() in {".json", ".jsonl", ".txt", ".md", ".py", ".sh", ".yaml", ".yml", ".toml"}:
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                issues.append(f"File is not valid UTF-8 ({rel}): {exc}")
                continue

            if f.suffix.lower() == ".json":
                try:
                    json.loads(text)
                except json.JSONDecodeError as exc:
                    issues.append(f"Invalid JSON in {rel}: {exc}")
            elif f.suffix.lower() == ".jsonl":
                for lineno, line in enumerate(text.splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        issues.append(f"Invalid JSONL at {rel}:{lineno}: {exc}")
                        break

            for secret_name, pat in SECRET_PATTERNS:
                if pat.search(text):
                    issues.append(f"Potential secret ({secret_name}) found in {rel}; scrub before uploading.")

    present_kinds = [k for k, v in detected.items() if v]
    if len(present_kinds) < 2:
        issues.append(
            f"Agent Team rules require at least 2 distinct evidence kinds from "
            f"{{trajectory, prompts, harness}}, found {len(present_kinds)} ({present_kinds})."
        )
    elif len(present_kinds) == 2:
        missing = [k for k in detected if not detected[k]]
        warnings.append(
            f"2 of 3 evidence kinds present ({present_kinds}); adding {missing[0]} strengthens prize auditability."
        )

    return EvidenceReport(
        directory=str(root),
        ok=len(issues) == 0,
        detected_kinds=detected,
        issues=issues,
        warnings=warnings,
    )
