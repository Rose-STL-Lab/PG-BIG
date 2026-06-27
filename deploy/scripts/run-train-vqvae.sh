#!/usr/bin/env bash
# Submit VQ-VAE training Job on Kubernetes.
#
# Usage: ./deploy/scripts/run-train-vqvae.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KUSTOMIZE_DIR="${KUSTOMIZE_DIR:-${REPO_ROOT}/deploy/jobs/train-vqvae}"
JOB_NAME="${JOB_NAME:-pg-big-train-vqvae}"
TIMEOUT="${TIMEOUT:-72h}"
RUN_LOG_ID="${RUN_LOG_ID:-train_vqvae_$(date -u +%Y%m%dT%H%M%SZ)}"

echo "=== train VQ-VAE ==="
echo "job=${JOB_NAME} run_log_id=${RUN_LOG_ID}"

if [[ "${SKIP_DELETE:-0}" != "1" ]]; then
  kubectl delete job "${JOB_NAME}" --ignore-not-found
fi

manifest="$(kubectl kustomize "${KUSTOMIZE_DIR}")"
manifest="$(printf '%s\n' "${manifest}" | sed "s|RUN_LOG_ID_PLACEHOLDER|${RUN_LOG_ID}|g")"
printf '%s\n' "${manifest}" | kubectl apply -f -

kubectl wait --for=condition=complete "job/${JOB_NAME}" --timeout="${TIMEOUT}"
kubectl get job "${JOB_NAME}"
