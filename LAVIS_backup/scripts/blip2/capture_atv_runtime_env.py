#!/usr/bin/env python3
"""Capture runtime provenance for ATV migration validation.

This script does not load the BLIP2 model or run pruning. It records enough
environment and source provenance to make a validation report reproducible:
Python/Torch/CUDA state, git commits/statuses, selected environment variables,
and hashes of the source files that implement the migration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from importlib import metadata as importlib_metadata
except Exception:  # pragma: no cover - Python 3.7 fallback only.
    importlib_metadata = None  # type: ignore


SOURCE_RELATIVE_PATHS = [
    "evaluate_blip.py",
    "lavis/configs/models/blip2/blip2_pretrain_flant5xl.yaml",
    "lavis/models/blip2_models/blip2.py",
    "lavis/compression/pruners/wanda_pruner.py",
    "lavis/models/blip2_models/blip2_t5.py",
    "scripts/blip2/capture_atv_runtime_env.py",
    "scripts/blip2/audit_atv_validation_report.py",
    "scripts/blip2/check_ckpt_sparsity.py",
    "scripts/blip2/compare_ckpts.py",
    "scripts/blip2/export_blip2_full_dense_state_dict.py",
    "scripts/blip2/materialize_cc3m_calib_cfg.py",
    "scripts/blip2/preflight_atv_validation.py",
    "scripts/blip2/snapshot_atv_artifacts.py",
    "scripts/blip2/validate_atv_migration.py",
    "scripts/blip2/run_atv_cc3m_prune_then_eval.sh",
    "scripts/blip2/run_atv_eval_matrix_fourbench.sh",
    "scripts/blip2/run_atv_full_verify.sh",
    "scripts/blip2/run_atv_multiseed_validation.sh",
]

ENV_KEYS = [
    "CUDA_VISIBLE_DEVICES",
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "LAVIS_ATV_DIAGNOSTIC_DIR",
    "OMP_NUM_THREADS",
    "PYTHONPATH",
    "REPORT_DIR",
    "TORCH_HOME",
    "TRANSFORMERS_CACHE",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(cmd: Sequence[str], cwd: Optional[Path] = None, timeout: int = 30) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": list(cmd),
            "cwd": str(cwd) if cwd else "",
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[-8000:],
            "stderr": proc.stderr.strip()[-8000:],
        }
    except Exception as exc:
        return {
            "command": list(cmd),
            "cwd": str(cwd) if cwd else "",
            "returncode": None,
            "stdout": "",
            "stderr": "%s: %s" % (type(exc).__name__, exc),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(role: str, path: Path, hash_max_bytes: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "bytes": None,
        "sha256": "",
        "note": "",
    }
    if not path.exists():
        row["note"] = "missing"
        return row
    if path.is_dir():
        row["note"] = "directory"
        return row
    size = path.stat().st_size
    row["bytes"] = size
    if size <= hash_max_bytes:
        row["sha256"] = sha256_file(path)
        row["note"] = "hashed"
    else:
        row["note"] = "sha256 skipped: file exceeds hash_max_bytes=%d" % hash_max_bytes
    return row


def git_info(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        info["error"] = "path does not exist"
        return info
    git = shutil.which("git")
    if not git:
        info["error"] = "git executable not found"
        return info

    safe_git = [git, "-c", "safe.directory=%s" % path]
    commands = {
        "top_level": safe_git + ["-C", str(path), "rev-parse", "--show-toplevel"],
        "commit": safe_git + ["-C", str(path), "rev-parse", "HEAD"],
        "branch": safe_git + ["-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
        "status_short": safe_git + ["-C", str(path), "status", "--short"],
        "diff_stat": safe_git + ["-C", str(path), "diff", "--stat"],
    }
    for key, cmd in commands.items():
        result = run_command(cmd, timeout=20)
        info[key] = result["stdout"] if result["returncode"] == 0 else ""
        if result["returncode"] != 0:
            info[key + "_error"] = result["stderr"]
    return info


def package_version(name: str) -> str:
    if importlib_metadata is None:
        return "metadata unavailable"
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "not installed"


def collect_torch(skip_import: bool) -> Dict[str, Any]:
    if skip_import:
        return {"skipped": True}
    try:
        import torch  # type: ignore
    except Exception as exc:
        return {"available": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    cuda_devices: List[Dict[str, Any]] = []
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            cuda_devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                }
            )
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": cuda_devices,
    }


def collect_nvidia_smi() -> Dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "note": "nvidia-smi not found on PATH"}
    return {"available": True, "result": run_command([exe], timeout=20)}


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# ATV Runtime Environment\n\n")
        f.write("- Captured UTC: `%s`.\n" % payload.get("captured_utc", ""))
        f.write("- Python executable: `%s`.\n" % payload["python"].get("executable", ""))
        f.write("- Python version: `%s`.\n" % payload["python"].get("version", ""))
        f.write("- Platform: `%s`.\n" % payload["platform"].get("platform", ""))
        torch_info = payload.get("torch", {})
        f.write("- Torch: `%s`.\n" % torch_info.get("version", torch_info.get("error", "unknown")))
        f.write("- CUDA available: `%s`.\n\n" % torch_info.get("cuda_available", "unknown"))

        f.write("## Git\n\n")
        f.write("| Repo | Commit | Branch | Dirty Lines |\n")
        f.write("|---|---|---|---:|\n")
        for role, info in payload.get("git", {}).items():
            status = str(info.get("status_short", ""))
            dirty_lines = len([x for x in status.splitlines() if x.strip()])
            commit = str(info.get("commit", ""))[:12]
            f.write("| %s | `%s` | `%s` | %d |\n" % (role, commit, info.get("branch", ""), dirty_lines))

        f.write("\n## Source Files\n\n")
        f.write("| Role | Exists | Bytes | SHA256 | Path |\n")
        f.write("|---|---|---:|---|---|\n")
        for row in payload.get("source_files", []):
            digest = str(row.get("sha256", ""))
            digest_s = digest[:12] + "..." if digest else ""
            f.write(
                "| %s | %s | %s | `%s` | %s |\n"
                % (
                    row.get("role", ""),
                    row.get("exists", ""),
                    row.get("bytes", ""),
                    digest_s,
                    str(row.get("path", "")).replace("|", "\\|"),
                )
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture runtime/source provenance for ATV migration validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lavis_root", required=True, type=Path)
    p.add_argument("--original_atv_root", required=True, type=Path)
    p.add_argument("--report_dir", required=True, type=Path)
    p.add_argument("--base", default="", help="Experiment base directory recorded for provenance.")
    p.add_argument("--stamp", default="", help="Run stamp recorded for provenance.")
    p.add_argument("--seeds", default="", help="Seed list recorded for provenance.")
    p.add_argument("--models", default="", help="Model list recorded for provenance.")
    p.add_argument("--run_prune", default="", help="Whether pruning was requested.")
    p.add_argument("--run_eval", default="", help="Whether evaluation was requested.")
    p.add_argument("--hash_max_bytes", type=int, default=128 * 1024 * 1024)
    p.add_argument("--skip_torch_import", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    source_files = [
        file_record("original_atv_pruner", args.original_atv_root / "qwen" / "activation_aware_pruner.py", args.hash_max_bytes)
    ]
    for rel in SOURCE_RELATIVE_PATHS:
        source_files.append(file_record(rel, args.lavis_root / rel, args.hash_max_bytes))

    payload: Dict[str, Any] = {
        "captured_utc": utc_now(),
        "argv": sys.argv,
        "experiment": {
            "base": args.base,
            "report_dir": str(args.report_dir),
            "stamp": args.stamp,
            "seeds": args.seeds,
            "models": args.models,
            "run_prune": args.run_prune,
            "run_eval": args.run_eval,
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
        },
        "environment": {key: os.environ.get(key, "") for key in ENV_KEYS},
        "packages": {
            name: package_version(name)
            for name in ["torch", "transformers", "timm", "omegaconf", "salesforce-lavis"]
        },
        "torch": collect_torch(args.skip_torch_import),
        "nvidia_smi": collect_nvidia_smi(),
        "git": {
            "lavis_root": git_info(args.lavis_root),
            "original_atv_root": git_info(args.original_atv_root),
        },
        "source_files": source_files,
    }

    out_json = args.report_dir / "runtime_environment.json"
    out_md = args.report_dir / "runtime_environment.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_md, payload)
    print("[OK] wrote runtime environment JSON: %s" % out_json)
    print("[OK] wrote runtime environment MD: %s" % out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
