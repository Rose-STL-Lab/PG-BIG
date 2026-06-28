# Kubernetes deployment

PG-BIG runs locally via `scripts/*.py` and on Kubernetes via Job manifests in `deploy/jobs/`.

## Layout

```
deploy/
  pvc.yaml
  scripts/
    run-retarget-athletes.sh
    run-train-vqvae.sh
    run-train-profile-encoder.sh
    run-train-subject-prior.sh
    run-train-surrogate.sh
    run-generate-motion.sh
  components/
    cluster-config/            # image + PVC (all jobs + dev pod)
  jobs/
    retarget-athletes/         # 90-way indexed (none | static-optimization | moco-track)
    train-vqvae/
    train-profile-encoder/
    train-subject-prior/
    train-surrogate/
    generate-motion/
  dev/                         # long-running interactive pod
```

## Pipeline order

| Step | Local | Kubernetes |
|------|-------|------------|
| 1. Retarget | `python scripts/retarget_athletes.py` | `./deploy/scripts/run-retarget-athletes.sh none` |
| 2. VQ-VAE | `python scripts/train_vqvae.py --config configs/train_vqvae.json` | `./deploy/scripts/run-train-vqvae.sh` |
| 3. Profile encoder | `python scripts/train_profile_encoder.py` | `./deploy/scripts/run-train-profile-encoder.sh` |
| 4. Subject prior | `python scripts/train_subject_prior.py` | `./deploy/scripts/run-train-subject-prior.sh` |
| 5. Surrogate | `python scripts/train_surrogate.py` | `./deploy/scripts/run-train-surrogate.sh` |
| 6. Generate motion | `python scripts/generate_motion.py` | `./deploy/scripts/run-generate-motion.sh` |

Retarget also supports `static-optimization` and `moco-track` activation methods (all write to `nimble_b3d/`).

**Not on Kubernetes:** `run_guidance.py` (WIP), `visualize_motion.py`, `render_frames_poly.py`, `retarget_baselines.py` — use the dev pod or local runs.

## Quick start

### 1. Configure cluster settings

Edit `deploy/components/cluster-config/`:

| File | Set |
|------|-----|
| `kustomization.yaml` | `images.newName`, `images.newTag` |
| `pvc-patch.yaml` | `claimName` (default: `pg-big`) |

### 2. Create storage

```bash
kubectl apply -f deploy/pvc.yaml
```

PVC layout:

- Repo at `/mnt/PG-BIG`
- Data at `/mnt/PG-BIG/datasets/183_athletes/`
- Checkpoints at `/mnt/PG-BIG/outputs/`

### 3. Run Jobs

```bash
# Data prep (pick activation method)
./deploy/scripts/run-retarget-athletes.sh none

# Training (sequential — later steps need earlier checkpoints)
./deploy/scripts/run-train-vqvae.sh
./deploy/scripts/run-train-profile-encoder.sh
./deploy/scripts/run-train-subject-prior.sh
./deploy/scripts/run-train-surrogate.sh

# Inference
./deploy/scripts/run-generate-motion.sh
```

Or apply manifests directly:

```bash
kubectl apply -k deploy/jobs/train-vqvae
kubectl wait --for=condition=complete job/pg-big-train-vqvae --timeout=72h
```

Preview a manifest:

```bash
kubectl kustomize deploy/jobs/train-vqvae
```

## Resource profiles

| Job | GPUs | CPU | Memory | Notes |
|-----|------|-----|--------|-------|
| retarget-athletes/* (×90 pods) | — | 1 each | 2Gi each | Indexed Job |
| train-vqvae | 2 | 16 | 64Gi | DeepSpeed if 2+ GPUs visible |
| train-profile-encoder | 1 | 8 | 32Gi | Needs VQ-VAE checkpoint |
| train-subject-prior | 1 | 8 | 32Gi | Needs VQ-VAE + encoder checkpoints |
| train-surrogate | 1 | 4 | 16Gi | Needs retargeted data |
| generate-motion | 1 | 4 | 16Gi | Needs all three model checkpoints |

GPU training jobs mount a `/dev/shm` emptyDir for PyTorch dataloaders.

## Checkpoint paths (defaults in job YAML)

Training jobs expect artifacts under `/mnt/PG-BIG/outputs/`:

| Stage | Default checkpoint |
|-------|-------------------|
| VQ-VAE | `outputs/VQVAE/VQVAE/300000.pth` |
| Profile encoder | `outputs/profile_encoder/profile_encoder_meta/profile_encoder.pth` |
| Subject prior decoder | `outputs/subject_prior/profile_decoder_best.pth` |

Override via env vars on the Job manifest (`VQVAE_CHECKPOINT`, `ENCODER_CHECKPOINT`, `DECODER_CHECKPOINT`) before apply.

## Retarget worker details

Each variant uses an **Indexed Job** with **90 pods** (one CPU, 2Gi memory each):

| Method | Job name | Output directory |
|--------|----------|------------------|
| `none` | `pg-big-retarget-none` | `datasets/183_athletes/nimble_b3d/` |
| `static_optimization` | `pg-big-retarget-static-optimization` | `datasets/183_athletes/nimble_b3d/` |
| `moco_track` | `pg-big-retarget-moco-track` | `datasets/183_athletes/nimble_b3d/` |

Shards use `JOB_COMPLETION_INDEX`. Each pod writes `retarget_manifest.NNNN.jsonl` when complete.

## Dev pod

```bash
kubectl apply -k deploy/dev
kubectl exec -it pg-big-dev -- bash -l
```

Inside: `conda activate pg-big`, then run scripts from `/mnt/PG-BIG`.
