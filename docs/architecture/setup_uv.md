# Setup uv

Le dépôt utilise `uv` pour standardiser l'environnement local et machine GPU.

## Cible recommandée

- Python projet: `3.12`
- macOS Apple Silicon ou Intel pour les sprints 1D en CPU
- Linux NVIDIA pour les phases 2D, baselines plus lourdes et acquisition distribuée

## Bootstrap rapide

### macOS / CPU

```bash
TORCH_BACKEND=cpu ./scripts/bootstrap_uv.sh
```

### Linux / NVIDIA CUDA 12.4

```bash
TORCH_BACKEND=cu124 ./scripts/bootstrap_uv.sh
```

### Linux / auto-détection PyTorch

```bash
TORCH_BACKEND=auto ./scripts/bootstrap_uv.sh
```

## Commandes de travail

### Tests

```bash
uv run pytest
```

### Génération offline 1D

```bash
uv run python scripts/generate_offline_data.py \
  data.dataset_version=local \
  data.output_dir=data/generated_local \
  solver.grid_size=64 \
  data.num_steps=8
```

### Entraînement AE 1D

```bash
uv run python scripts/train_autoencoder.py \
  train.dataset_root=data/generated_local/burgers_1d_offline/local \
  train.output_dir=artifacts/runs/ae_local \
  train.epochs=5 \
  train.batch_size=8
```

### Entraînement dynamique 1D

```bash
uv run python scripts/train_dynamics.py \
  train.dataset_root=data/generated_local/burgers_1d_offline/local \
  train.ae_checkpoint=artifacts/runs/ae_local/best.pt \
  train.output_dir=artifacts/runs/dynamics_local \
  train.epochs=5 \
  train.batch_size=8
```

### Campagne séquentielle Sprint 4

```bash
uv run python scripts/run_sprint4_experiments.py --prepare-data --epochs 4
```

Pour `KS 1D`, utiliser par exemple:

```bash
uv run python scripts/run_sprint4_experiments.py \
  --prepare-data \
  --data-config ks_1d \
  --solver-config ks_1d \
  --dataset-root data/generated_sprint4/ks_1d_offline/sprint4_ks \
  --output-root artifacts/runs/sprint4_seq_ks \
  --data-version sprint4_ks \
  --epochs 4
```

### Campagne longue chaînée

Lancement direct au premier plan:

```bash
uv run python scripts/run_long_campaign.py --prepare-data
```

Lancement en arrière-plan avec log et PID:

```bash
./scripts/launch_long_campaign.sh --prepare-data
```

Par défaut, la campagne longue enchaîne:

- baselines Burgers 1D longues;
- baselines KS 1D longues;
- world model Burgers 1D long avec acquisition online par active sampling d'états.

Les defaults actuels sont:

- `baseline_epochs=120`
- `ae_epochs=120`
- `dynamics_epochs=120`
- `fine_tune_epochs=100`
- `online_iters=3`
- `ensemble_size=3`

## Notes plateforme

- Le dépôt reste compatible avec `uv run` sur macOS CPU pour Burgers/KS 1D.
- Pour Linux NVIDIA, `uv sync --torch-backend cu124` permet d'installer une roue PyTorch CUDA sans modifier `pyproject.toml`.
- Le script `bootstrap_uv.sh` fait un `uv sync` standard puis réinstalle `torch` avec le backend demandé quand `TORCH_BACKEND != cpu`.
- Les benchmarks 2D et la suite Ray/Slurm devront être validés sur une machine Linux dédiée; le Mac est ciblé pour les premiers sprints 1D.
