#!/usr/bin/env bash
# Run distributed 183-athletes retargeting on Kubernetes (64 indexed worker pods).
#
# Usage:
#   ./deploy/scripts/run-retarget-athletes.sh [none|static-optimization|moco-track]
#   ACTIVATION_METHOD=static_optimization ./deploy/scripts/run-retarget-athletes.sh
#
# Environment overrides:
#   RETARGET_NUM_SHARDS   shard count (default: 64)
#   WORKER_TIMEOUT        kubectl wait timeout (default: 48h)
#   KUSTOMIZE_DIR         override worker job kustomization path
#   SKIP_DELETE           set to 1 to skip deleting stale jobs before apply

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JOBS_ROOT="${REPO_ROOT}/deploy/jobs/retarget-athletes"

resolve_method() {
  local raw="${1:-${ACTIVATION_METHOD:-none}}"
  case "${raw}" in
    none)
      METHOD_FOLDER="none"
      ACTIVATION_METHOD="none"
      WORKER_JOB_NAME="pg-big-retarget-none"
      ;;
    static_optimization|static-optimization)
      METHOD_FOLDER="static-optimization"
      ACTIVATION_METHOD="static_optimization"
      WORKER_JOB_NAME="pg-big-retarget-static-optimization"
      ;;
    moco_track|moco-track)
      METHOD_FOLDER="moco-track"
      ACTIVATION_METHOD="moco_track"
      WORKER_JOB_NAME="pg-big-retarget-moco-track"
      ;;
    *)
      echo "Unknown retarget method: ${raw}" >&2
      echo "Use: none, static-optimization, or moco-track" >&2
      exit 1
      ;;
  esac
}

METHOD_ARG="${1:-}"
resolve_method "${METHOD_ARG}"

KUSTOMIZE_DIR="${KUSTOMIZE_DIR:-${JOBS_ROOT}/${METHOD_FOLDER}}"
WORKER_TIMEOUT="${WORKER_TIMEOUT:-48h}"
export RETARGET_NUM_SHARDS="${RETARGET_NUM_SHARDS:-64}"
RUN_LOG_ID="${RUN_LOG_ID:-retarget_${METHOD_FOLDER}_$(date -u +%Y%m%dT%H%M%SZ)}"
export RUN_LOG_ID

echo "=== retarget athletes (distributed) ==="
echo "method=${METHOD_FOLDER} activation_method=${ACTIVATION_METHOD}"
echo "num_shards=${RETARGET_NUM_SHARDS}"
echo "run_log_id=${RUN_LOG_ID}"
echo "worker kustomize: ${KUSTOMIZE_DIR}"

if [[ "${SKIP_DELETE:-0}" != "1" ]]; then
  echo "Deleting stale job (if any)..."
  kubectl delete job "${WORKER_JOB_NAME}" --ignore-not-found
fi

echo "Applying worker Indexed Job..."
manifest="$(kubectl kustomize "${KUSTOMIZE_DIR}")"
manifest="$(printf '%s\n' "${manifest}" | sed \
  "s|RUN_LOG_ID_PLACEHOLDER|${RUN_LOG_ID}|g; \
  /name: RETARGET_NUM_SHARDS/{n;s/value: .*/value: \"${RETARGET_NUM_SHARDS}\"/;}")"
printf '%s\n' "${manifest}" | kubectl apply -f -

echo "Waiting for worker Job to complete (timeout=${WORKER_TIMEOUT})..."
kubectl wait --for=condition=complete "job/${WORKER_JOB_NAME}" --timeout="${WORKER_TIMEOUT}"

echo "=== retarget complete ==="
kubectl get job "${WORKER_JOB_NAME}"
