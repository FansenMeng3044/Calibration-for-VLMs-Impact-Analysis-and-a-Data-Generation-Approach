#!/usr/bin/env python3
"""Audit the final ATV migration validation report.

This script is intentionally simple and conservative. It reads the machine
checklist produced by ``validate_atv_migration.py`` and emits a compact
PASS/FAIL audit that is easy to attach to a paper-grade evidence package.
It does not recompute metrics; it checks whether the strict evidence gates are
present, passing, and traceable back to the migration claims.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Row = Dict[str, str]


def read_csv_rows(path: Path) -> List[Row]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: Sequence[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "check_id",
        "source",
        "artifact",
        "requirement",
        "required_for_strict",
        "status",
        "evidence_count",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def is_yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "yes", "true", "y"}


def norm_status(value: str) -> str:
    text = str(value or "").strip().upper()
    return text if text else "MISSING"


def manifest_audit_rows(manifest_path: Path, manifest_rows: Sequence[Row]) -> List[Row]:
    rows: List[Row] = []
    if not manifest_path.is_file():
        rows.append(
            {
                "check_id": "manifest_present",
                "source": "manifest",
                "artifact": "validation_manifest.csv",
                "requirement": "strict evidence checklist exists",
                "required_for_strict": "yes",
                "status": "FAIL",
                "evidence_count": "0",
                "note": "validation_manifest.csv is missing",
            }
        )
        return rows

    if not manifest_rows:
        rows.append(
            {
                "check_id": "manifest_nonempty",
                "source": "manifest",
                "artifact": "validation_manifest.csv",
                "requirement": "strict evidence checklist is nonempty",
                "required_for_strict": "yes",
                "status": "FAIL",
                "evidence_count": "0",
                "note": "manifest has no data rows",
            }
        )
        return rows

    for idx, row in enumerate(manifest_rows, start=1):
        required = is_yes(row.get("required_for_strict", ""))
        status = norm_status(row.get("status", ""))
        audit_status = "PASS" if (not required or status == "PASS") else "FAIL"
        rows.append(
            {
                "check_id": "manifest_row_%03d" % idx,
                "source": "manifest",
                "artifact": row.get("artifact", ""),
                "requirement": row.get("requirement", ""),
                "required_for_strict": "yes" if required else "no",
                "status": audit_status,
                "evidence_count": row.get("evidence_count", ""),
                "note": row.get("note", ""),
            }
        )
    return rows


def traceability_audit_rows(
    trace_path: Path,
    trace_rows: Sequence[Row],
    manifest_rows: Sequence[Row],
) -> List[Row]:
    rows: List[Row] = []
    strict_requirements = {
        str(row.get("requirement", "")).strip()
        for row in manifest_rows
        if is_yes(row.get("required_for_strict", ""))
    }

    if not trace_path.is_file():
        rows.append(
            {
                "check_id": "traceability_present",
                "source": "traceability",
                "artifact": "validation_traceability.csv",
                "requirement": "objective-to-evidence traceability exists",
                "required_for_strict": "yes",
                "status": "FAIL",
                "evidence_count": "0",
                "note": "validation_traceability.csv is missing",
            }
        )
        return rows

    if not trace_rows:
        rows.append(
            {
                "check_id": "traceability_nonempty",
                "source": "traceability",
                "artifact": "validation_traceability.csv",
                "requirement": "objective-to-evidence traceability is nonempty",
                "required_for_strict": "yes",
                "status": "FAIL",
                "evidence_count": "0",
                "note": "traceability CSV has no data rows",
            }
        )
        return rows

    traced_requirements = {
        str(row.get("manifest_requirement", "")).strip()
        for row in trace_rows
        if str(row.get("manifest_requirement", "")).strip()
    }
    missing_traced = sorted(req for req in strict_requirements if req and req not in traced_requirements)
    rows.append(
        {
            "check_id": "traceability_covers_strict_manifest",
            "source": "traceability",
            "artifact": "validation_traceability.csv",
            "requirement": "every strict manifest requirement has a traceability row",
            "required_for_strict": "yes",
            "status": "PASS" if not missing_traced else "FAIL",
            "evidence_count": str(len(traced_requirements)),
            "note": "missing=%s" % ",".join(missing_traced[:20]) if missing_traced else "all strict requirements traced",
        }
    )

    for idx, row in enumerate(trace_rows, start=1):
        status = norm_status(row.get("current_status", ""))
        rows.append(
            {
                "check_id": "traceability_row_%03d" % idx,
                "source": "traceability",
                "artifact": row.get("evidence_artifact", ""),
                "requirement": row.get("objective_requirement", ""),
                "required_for_strict": "yes",
                "status": "PASS" if status == "PASS" else "FAIL",
                "evidence_count": row.get("evidence_count", ""),
                "note": "manifest_requirement=%s; manifest_status=%s; %s"
                % (
                    row.get("manifest_requirement", ""),
                    status,
                    row.get("manifest_note", ""),
                ),
            }
        )
    return rows


def summarize(rows: Sequence[Row]) -> Tuple[int, int, int]:
    strict_rows = [row for row in rows if is_yes(row.get("required_for_strict", ""))]
    strict_fail = [row for row in strict_rows if norm_status(row.get("status", "")) != "PASS"]
    return len(strict_rows), len(strict_rows) - len(strict_fail), len(strict_fail)


def md_table(rows: Sequence[Row]) -> List[str]:
    lines = [
        "| Source | Artifact | Requirement | Status | Count | Note |",
        "|---|---|---|---|---:|---|",
    ]
    for row in rows:
        note = row.get("note", "").replace("|", "\\|")
        requirement = row.get("requirement", "").replace("|", "\\|")
        lines.append(
            "| %s | `%s` | %s | %s | %s | %s |"
            % (
                row.get("source", ""),
                row.get("artifact", ""),
                requirement,
                row.get("status", ""),
                row.get("evidence_count", ""),
                note,
            )
        )
    return lines


def write_markdown(path: Path, rows: Sequence[Row]) -> None:
    strict_total, strict_pass, strict_fail = summarize(rows)
    failed = [row for row in rows if is_yes(row.get("required_for_strict", "")) and norm_status(row.get("status", "")) != "PASS"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# ATV Validation Strict Audit\n\n")
        f.write("- Strict verdict: %s.\n" % ("PASS" if strict_fail == 0 else "FAIL"))
        f.write("- Strict gates: %d total, %d pass, %d fail.\n" % (strict_total, strict_pass, strict_fail))
        f.write("- Machine-readable audit: `strict_audit_summary.csv`.\n\n")
        if failed:
            f.write("## Failed Strict Gates\n\n")
            f.write("\n".join(md_table(failed)))
            f.write("\n\n")
            f.write("Do not claim the ATV migration is fully validated until these gates pass.\n")
        else:
            f.write("All strict manifest and traceability gates pass.\n")


def build_audit(report_dir: Path) -> List[Row]:
    manifest_path = report_dir / "validation_manifest.csv"
    trace_path = report_dir / "validation_traceability.csv"
    manifest_rows = read_csv_rows(manifest_path)
    trace_rows = read_csv_rows(trace_path)
    return manifest_audit_rows(manifest_path, manifest_rows) + traceability_audit_rows(
        trace_path,
        trace_rows,
        manifest_rows,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit strict ATV migration validation gates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--report_dir", required=True, type=Path)
    p.add_argument("--out_csv", type=Path, default=None)
    p.add_argument("--out_md", type=Path, default=None)
    p.add_argument("--no_fail", action="store_true", help="Write audit files but return success even when strict gates fail.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir
    out_csv = args.out_csv or (report_dir / "strict_audit_summary.csv")
    out_md = args.out_md or (report_dir / "strict_audit_summary.md")

    rows = build_audit(report_dir)
    write_csv(out_csv, rows)
    write_markdown(out_md, rows)

    strict_total, strict_pass, strict_fail = summarize(rows)
    print(
        "[AUDIT] strict gates: total=%d pass=%d fail=%d report=%s"
        % (strict_total, strict_pass, strict_fail, report_dir)
    )
    print("[AUDIT] wrote: %s" % out_csv)
    print("[AUDIT] wrote: %s" % out_md)

    if strict_fail and not args.no_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
