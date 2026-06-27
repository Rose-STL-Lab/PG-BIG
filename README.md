# PG-BIG: Personalize Guidance for Biomechanically Informed GenAI

PG-BIG is a framework for personalized guidance in biomechanically informed generative AI, focusing on motion modeling using VQ-VAE, profile encoders, subject priors, and surrogate muscle models.

## Setup

```bash
git clone https://github.com/your-org/PG-BIG.git
cd PG-BIG
conda env create -f env/environment.yaml
conda activate pg-big
```

Run all commands from the **repository root** with `PYTHONPATH=.` (scripts add the repo root automatically).

## Data layout

Place the 183-athletes Figshare data under `datasets/183_athletes/`:

```
datasets/183_athletes/
├── Kinematic_Data/          # Raw C3D per subject
├── Participants Info/       # Subject metadata spreadsheets
└── retargeted/              # Output: one {subject_id}.b3d per athlete
```

Override the data root with the `ATHLETES_DATA_ROOT` environment variable if needed.

## Pipeline

### 1. Retarget C3D markers to Rajagopal skeleton

```bash
python scripts/retarget_athletes.py
```

### 2. Train VQ-VAE

```bash
python scripts/train_vqvae.py --config configs/train_vqvae.json
```

With DeepSpeed (2+ GPUs):

```bash
deepspeed --num_gpus=<N> scripts/train_vqvae.py --config configs/train_vqvae.json
```

Supported datasets: `183_athletes`, `addbiomechanics`.

### 3. Train profile encoder

```bash
python scripts/train_profile_encoder.py --config configs/train_profile_encoder.json
```

### 4. Train subject prior

```bash
python scripts/train_subject_prior.py --config configs/train_subject_prior.json
```

### 5. Train surrogate model

```bash
python scripts/train_surrogate.py --config configs/train_surrogate.json
```

### 6. Generate motion

```bash
python scripts/generate_motion.py --help
```

### 7. Visualize motion

```bash
python scripts/visualize_motion.py --b3d-path datasets/183_athletes/retargeted/927.b3d
```

## Project layout

| Path | Role |
|------|------|
| `scripts/` | CLI entry points (thin wrappers) |
| `common/` | Paths, logging, runtime, motion math |
| `datasets/` | PyTorch dataset loaders + on-disk data |
| `nimble/` | Nimblephysics retargeting and visualization |
| `vqvae/` | VQ-VAE model and training |
| `profile/` | Profile encoder and subject prior |
| `surrogate/` | Muscle activation surrogate |
| `eval/` | Motion-text evaluation (SMPL-based) |
| `configs/` | JSON training defaults |
| `visualization/` | 3D skeleton plotting |
| `env/` | Conda environment and Docker image |
| `deploy/` | Kubernetes manifests |

## Migration note

On-disk data moved from `dataset/183_athletes/` to `datasets/183_athletes/`. Update any external references accordingly.

## Kubernetes

See [deploy/README.md](deploy/README.md) for the full pipeline:

```bash
./deploy/scripts/run-retarget-athletes.sh none
./deploy/scripts/run-train-vqvae.sh
./deploy/scripts/run-train-profile-encoder.sh
./deploy/scripts/run-train-subject-prior.sh
./deploy/scripts/run-train-surrogate.sh
./deploy/scripts/run-generate-motion.sh
```
