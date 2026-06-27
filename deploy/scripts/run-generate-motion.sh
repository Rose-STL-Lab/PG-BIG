#!/usr/bin/env bash
# Submit motion generation inference Job on Kubernetes.
#
# Usage: ./deploy/scripts/run-generate-motion.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KUSTOMIZE_DIR="${KUSTOMIZE_DIR:-${REPO_ROOT}/deploy/jobs/generate-motion}"
JOB_NAME="${JOB_NAME:-pg-big-generate-motion}"
TIMEOUT="${TIMEOUT:-2h}"
RUN_LOG_ID="${RUN_LOG_ID:-generate_motion_$(date -u +%Y%m%dT%H%M%SZ)}"

echo "=== generate motion ==="
echo "job=${JOB_NAME} run_log_id=${RUN_LOG_ID}"

if [[ "${SKIP_DELETE:-0}" != "1" ]]; then
  kubectl delete job "${JOB_NAME}" --ignore-not-found
fi

manifest="$(kubectl kustomize "${KUSTOMIZE_DIR}")"
manifest="$(printf '%s\n' "${manifest}" | sed "s|RUN_LOG_ID_PLACEHOLDER|${RUN_LOG_ID}|g")"
printf '%s\n' "${manifest}" | kubectl apply -f -

kubectl wait --for=condition=complete "job/${JOB_NAME}" --timeout="${TIMEOUT}"
kubectl get job "${JOB_NAME}"
