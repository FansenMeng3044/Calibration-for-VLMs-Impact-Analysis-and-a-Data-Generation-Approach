#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 RUN_ROOT WORKER_ID PHYSICAL_GPU SSH_PORT_LABEL" >&2
  exit 2
fi

run_root="$1"
worker_id="$2"
physical_gpu="$3"
port_label="$4"
code_root="/private/workspace/hycui/mfs/cosmos_atv"
python_bin="/private/workspace/hycui/envs/cosmos3-edge/bin/python"
tasks_root="${run_root}/controller/tasks"
worker_dir="${run_root}/controller/workers/${worker_id}"
gate_dir="${tasks_root}/000_gate_partial_save_reload_eval"

mkdir -p "${worker_dir}"
printf '%s\n' "$$" >"${worker_dir}/worker.pid"

task_readiness() {
  "${python_bin}" - "$1" "${tasks_root}" "${gate_dir}" <<'PY'
import json
import pathlib
import sys

task_path = pathlib.Path(sys.argv[1])
tasks_root = pathlib.Path(sys.argv[2])
gate_dir = pathlib.Path(sys.argv[3])
config = json.loads(task_path.read_text(encoding="utf-8"))
dependency = config.get("dependency")
dependencies = list(config.get("dependencies") or [])
if dependency:
    dependencies.append(dependency)
for dependency in dependencies:
    dep_dir = tasks_root / dependency
    if (dep_dir / ".failed").exists() or (dep_dir / ".blocked").exists():
        print(f"dependency terminal failure: {dependency}")
        raise SystemExit(4)
    if not (dep_dir / ".done").exists():
        raise SystemExit(3)
if config.get("requires_gate"):
    if (gate_dir / ".failed").exists() or (gate_dir / ".blocked").exists():
        print("checkpoint save/reload gate failed")
        raise SystemExit(4)
    if not (gate_dir / ".done").exists():
        raise SystemExit(3)
raise SystemExit(0)
PY
}

while true; do
  printf '{"worker_id":"%s","port":"%s","physical_gpu":"%s","pid":%s,"heartbeat":"%s"}\n' \
    "${worker_id}" "${port_label}" "${physical_gpu}" "$$" "$(date -Is)" \
    >"${worker_dir}/heartbeat.json"

  total=0
  terminal=0
  claimed_task=""
  while IFS= read -r task_json; do
    total=$((total + 1))
    task_dir="$(dirname "${task_json}")"
    if [[ -e "${task_dir}/.done" || -e "${task_dir}/.failed" || -e "${task_dir}/.blocked" ]]; then
      terminal=$((terminal + 1))
      continue
    fi
    if [[ -d "${task_dir}/claim" ]]; then
      continue
    fi

    readiness_output=""
    set +e
    readiness_output="$(task_readiness "${task_json}" 2>&1)"
    readiness_status=$?
    set -e
    if [[ ${readiness_status} -eq 4 ]]; then
      printf '%s\n' "${readiness_output}" >"${task_dir}/.blocked"
      terminal=$((terminal + 1))
      continue
    fi
    if [[ ${readiness_status} -ne 0 ]]; then
      continue
    fi

    if mkdir "${task_dir}/claim" 2>/dev/null; then
      printf '{"worker_id":"%s","port":"%s","physical_gpu":"%s","pid":%s,"claimed":"%s"}\n' \
        "${worker_id}" "${port_label}" "${physical_gpu}" "$$" "$(date -Is)" \
        >"${task_dir}/claim/owner.json"
      claimed_task="${task_json}"
      break
    fi
  done < <(find "${tasks_root}" -mindepth 2 -maxdepth 2 -name task.json -print | sort)

  if [[ -n "${claimed_task}" ]]; then
    task_dir="$(dirname "${claimed_task}")"
    printf '%s\n' "${claimed_task}" >"${worker_dir}/current_task"
    printf '%s claimed %s on physical GPU %s\n' "$(date -Is)" "${claimed_task}" "${physical_gpu}" \
      >>"${worker_dir}/worker.log"
    set +e
    "${python_bin}" "${code_root}/run_full_matrix_task.py" \
      --task-json "${claimed_task}" \
      --physical-gpu "${physical_gpu}" \
      --worker-id "${worker_id}" \
      >>"${task_dir}/run.log" 2>&1
    task_status=$?
    set -e
    if [[ ${task_status} -ne 0 ]]; then
      printf '%s\n' "$(date -Is)" >"${task_dir}/.failed"
      printf '%s failed %s exit=%s\n' "$(date -Is)" "${claimed_task}" "${task_status}" \
        >>"${worker_dir}/worker.log"
    else
      printf '%s completed %s\n' "$(date -Is)" "${claimed_task}" \
        >>"${worker_dir}/worker.log"
    fi
    : >"${worker_dir}/current_task"
    continue
  fi

  if [[ ${total} -gt 0 && ${terminal} -eq ${total} ]]; then
    printf '%s\n' "$(date -Is)" >"${worker_dir}/all_tasks_terminal"
    exit 0
  fi
  sleep 10
done
