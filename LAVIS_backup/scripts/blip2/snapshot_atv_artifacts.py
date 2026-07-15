#!/usr/bin/env python3
"""Create a reproducibility snapshot for ATV validation artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_REPORT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".png",
    ".txt",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def parse_include(text: str) -> Tuple[str, Path]:
    if "=" in text:
        role, path = text.split("=", 1)
        return role.strip() or "included", Path(path)
    return "included", Path(text)


def add_file(
    rows: List[Dict[str, object]],
    seen: set,
    role: str,
    path: Path,
    hash_max_bytes: int,
) -> None:
    key = str(path.resolve()) if path.exists() else str(path)
    if key in seen:
        return
    seen.add(key)

    if not path.exists():
        rows.append(
            {
                "role": role,
                "path": str(path),
                "exists": "no",
                "bytes": "",
                "mtime_utc": "",
                "sha256": "",
                "note": "missing",
            }
        )
        return

    if path.is_dir():
        rows.append(
            {
                "role": role,
                "path": str(path),
                "exists": "yes",
                "bytes": "",
                "mtime_utc": mtime_utc(path),
                "sha256": "",
                "note": "directory",
            }
        )
        return

    size = path.stat().st_size
    if size <= hash_max_bytes:
        digest = sha256_file(path)
        note = "hashed"
    else:
        digest = ""
        note = "sha256 skipped: file exceeds hash_max_bytes=%d" % hash_max_bytes

    rows.append(
        {
            "role": role,
            "path": str(path),
            "exists": "yes",
            "bytes": size,
            "mtime_utc": mtime_utc(path),
            "sha256": digest,
            "note": note,
        }
    )


def report_files(report_dir: Path) -> Iterable[Path]:
    if not report_dir.exists():
        return []
    return sorted(
        p for p in report_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in DEFAULT_REPORT_SUFFIXES
    )


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["role", "path", "exists", "bytes", "mtime_utc", "sha256", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [r for r in rows if r["exists"] != "yes"]
    hashed = [r for r in rows if r.get("sha256")]
    skipped = [r for r in rows if str(r.get("note", "")).startswith("sha256 skipped")]
    with path.open("w", encoding="utf-8") as f:
        f.write("# ATV Validation Artifact Snapshot\n\n")
        f.write("- Total entries: %d.\n" % len(rows))
        f.write("- Hashed files: %d.\n" % len(hashed))
        f.write("- Large files without hashes: %d.\n" % len(skipped))
        f.write("- Missing entries: %d.\n\n" % len(missing))
        f.write("| Role | Exists | Bytes | SHA256 | Path | Note |\n")
        f.write("|---|---|---:|---|---|---|\n")
        for row in rows:
            digest = str(row.get("sha256", ""))
            digest_s = digest[:12] + "..." if digest else ""
            f.write(
                "| %s | %s | %s | %s | %s | %s |\n"
                % (
                    row.get("role", ""),
                    row.get("exists", ""),
                    row.get("bytes", ""),
                    digest_s,
                    str(row.get("path", "")).replace("|", "\\|"),
                    str(row.get("note", "")).replace("|", "\\|"),
                )
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Snapshot ATV validation report artifacts and checkpoint provenance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--report_dir", required=True, type=Path)
    p.add_argument(
        "--include",
        action="append",
        default=[],
        help="Extra path to include, optionally role=/path/to/file. Repeatable.",
    )
    p.add_argument(
        "--hash_max_bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="Hash files up to this many bytes; larger files record size only.",
    )
    p.add_argument("--out_csv", type=Path, default=None)
    p.add_argument("--out_md", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_csv = args.out_csv or args.report_dir / "artifact_snapshot.csv"
    out_md = args.out_md or args.report_dir / "artifact_snapshot.md"
    rows: List[Dict[str, object]] = []
    seen = set()

    for path in report_files(args.report_dir):
        role = "report/%s" % path.suffix.lower().lstrip(".")
        add_file(rows, seen, role, path, args.hash_max_bytes)

    for item in args.include:
        role, path = parse_include(item)
        add_file(rows, seen, role, path, args.hash_max_bytes)

    write_csv(out_csv, rows)
    write_markdown(out_md, rows)
    missing = [r for r in rows if r["exists"] != "yes"]
    print("[OK] wrote artifact snapshot CSV: %s" % out_csv)
    print("[OK] wrote artifact snapshot MD: %s" % out_md)
    if missing:
        print("[WARN] missing artifact entries: %d" % len(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
