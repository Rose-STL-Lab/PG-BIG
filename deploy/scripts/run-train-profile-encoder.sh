#!/usr/bin/env bash
# Submit profile encoder training Job on Kubernetes.
#
# Usage: ./deploy/scripts/run-train-profile-encoder.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KUSTOMIZE_DIR="${KUSTOMIZE_DIR:-${REPO_ROOT}/deploy/jobs/train-profile-encoder}"
JOB_NAME="${JOB_NAME:-pg-big-train-profile-encoder}"
TIMEOUT="${TIMEOUT:-72h}"
RUN_LOG_ID="${RUN_LOG_ID:-train_profile_encoder_$(date -u +%Y%m%dT%H%M%SZ)}"

echo "=== train profile encoder ==="
echo "job=${JOB_NAME} run_log_id=${RUN_LOG_ID}"

if [[ "${SKIP_DELETE:-0}" != "1" ]]; then
  kubectl delete job "${JOB_NAME}" --ignore-not-found
fi

manifest="$(kubectl kustomize "${KUSTOMIZE_DIR}")"
manifest="$(printf '%s\n' "${manifest}" | sed "s|RUN_LOG_ID_PLACEHOLDER|${RUN_LOG_ID}|g")"
printf '%s\n' "${manifest}" | kubectl apply -f -

kubectl wait --for=condition=complete "job/${JOB_NAME}" --timeout="${TIMEOUT}"
kubectl get job "${JOB_NAME}"
