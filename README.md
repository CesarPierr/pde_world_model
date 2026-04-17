# PDE World Model

A publication-grade latent world model for PDE benchmarks with **active acquisition in latent space**. The system learns compressed representations of PDE states, predicts their dynamics, and intelligently selects which new simulations to run — including a novel **Conditional Flow Matching generative sampler** that focuses acquisition on regions where the model struggles most.

## Key Features

- **Autoencoder**: CNN encoder/decoder for 1D PDE states (PDE-agnostic)
- **Latent dynamics**: FiLM-conditioned residual transition model with physics context
- **Ensemble committee**: Multiple dynamics models for epistemic uncertainty estimation
- **Active acquisition**: 9 strategies including loss-weighted generative sampling
- **Auto hardware**: Automatic CUDA / MPS / CPU detection
- **W&B logging**: Integrated experiment tracking with Weights & Biases

## Supported PDEs

| PDE | Dimension | Status |
|-----|-----------|--------|
| Burgers equation | 1D | ✅ Implemented |
| Kuramoto-Sivashinsky | 1D | ✅ Implemented |
| Diffusion-reaction | 2D | 📋 Planned |
| Navier-Stokes | 2D | 📋 Planned |

---

## Installation

### Prerequisites

- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/) package manager
- (Optional) NVIDIA GPU with CUDA-compatible driver

### macOS / CPU

```bash
git clone https://github.com/CesarPierr/pde_world_model.git
cd pde_world_model
./scripts/bootstrap_uv.sh
```

### Linux / NVIDIA GPU

```bash
git clone https://github.com/CesarPierr/pde_world_model.git
cd pde_world_model
TORCH_BACKEND=auto ./scripts/bootstrap_uv.sh
```

To force a specific backend, pass `TORCH_BACKEND=<uv-supported-backend>` (for example `cpu`).

### Verify installation

```bash
uv run pytest          # run all tests (33 tests)
uv run python -c "from pdewm.utils.device import resolve_device; print(resolve_device('auto'))"
```

---

## Quick Start

### 1. Generate offline data

```bash
# Burgers 1D (default)
uv run python scripts/generate_offline_data.py

# Kuramoto-Sivashinsky 1D
uv run python scripts/generate_offline_data.py --data-config ks_1d --solver-config ks_1d
```

### 2. Train the autoencoder

```bash
uv run python scripts/train_autoencoder.py \
  train.dataset_root=data/generated/burgers_1d_offline/v0 \
  train.output_dir=artifacts/runs/ae_burgers \
  train.epochs=120
```

### 3. Train the latent dynamics model

```bash
uv run python scripts/train_dynamics.py \
  train.dataset_root=data/generated/burgers_1d_offline/v0 \
  train.ae_checkpoint=artifacts/runs/ae_burgers/last.pt \
  train.output_dir=artifacts/runs/dynamics_burgers \
  train.epochs=120 \
  train.regime=joint_ema
```

### 4. Train baselines

```bash
# Individual baseline
uv run python scripts/train_baseline.py --model-config fno_1d \
  train.dataset_root=data/generated/burgers_1d_offline/v0 \
  train.output_dir=artifacts/runs/fno_burgers

# All baselines sequentially
uv run python scripts/run_sprint4_experiments.py --wandb
```

---

## Benchmarks

### Standard Benchmark (CPU / light GPU)

Compares 6 heuristic acquisition strategies under a fixed solver budget:

```bash
uv run python scripts/run_worldmodel_benchmark.py \
  --output-root artifacts/runs/benchmark \
  --wandb --prepare-data
```

### Challenging Benchmark (GPU server)

Designed for RTX 2070 Super / 16 CPU cores — larger models, more data, higher resolution:

```bash
uv run python scripts/run_worldmodel_challenging_benchmark.py \
  --pdes burgers_1d \
  --seeds 7 42 123 \
  --wandb --prepare-data
```

Key differences vs standard: grid 256 (vs 64), 64 train trajectories (vs 16), ensemble of 5 (vs 3), 300 training epochs, 512 online transition budget.

### Generative Benchmark (GPU server)

Compares all **9 strategies** — 6 heuristic + 3 generative (flow matching):

```bash
uv run python scripts/run_worldmodel_generative_benchmark.py \
  --output-root artifacts/runs/generative_benchmark \
  --wandb --prepare-data --prepare-ae
```

The 3 new generative strategies use a **Conditional Flow Matching** model trained in a compressed 512-dim latent space with loss-weighted sampling:

| Strategy | Description |
|----------|-------------|
| `generative_loss_weighted` | Flow matching with sampling probability ∝ transition loss |
| `generative_uniform` | Flow matching with uniform weights (ablation) |
| `generative_combined` | Merged pool of flow + heuristic candidates |

---

## Project Structure

```
pde_world_model/
├── configs/                        # YAML configs (OmegaConf)
│   ├── data/                       #   dataset generation configs
│   ├── model/                      #   model architecture configs
│   ├── solver/                     #   PDE solver configs
│   ├── train/                      #   training hyperparameters
│   └── logging/                    #   wandb config
├── src/pdewm/                      # Source code
│   ├── acquisition/                #   active acquisition strategies
│   │   ├── generative.py           #     flow matching generative sampler
│   │   ├── heuristic.py            #     committee-based heuristic sampler
│   │   └── online.py               #     online acquisition loop
│   ├── data/                       #   dataset loading and generation
│   ├── models/                     #   neural network architectures
│   │   ├── representations/        #     autoencoder
│   │   ├── dynamics/               #     latent transition model
│   │   └── auxiliaries/            #     baselines (FNO, UNet, CNN, POD)
│   ├── solvers/                    #   PDE solvers (Burgers, KS)
│   ├── training/                   #   training loops
│   └── utils/                      #   device detection, config, wandb
├── scripts/                        # Entry points
│   ├── generate_offline_data.py    #   PDE dataset generation
│   ├── train_autoencoder.py        #   AE training
│   ├── train_dynamics.py           #   dynamics model training
│   ├── train_baseline.py           #   baseline training
│   ├── run_worldmodel_benchmark.py #   standard benchmark (6 strategies)
│   ├── run_worldmodel_generative_benchmark.py  # generative benchmark (9 strategies)
│   └── run_worldmodel_challenging_benchmark.py # GPU-scale benchmark
├── tests/                          # Test suite (33 tests)
├── IMPLEMENTATION_TRACKER.md       # Sprint-level progress tracking
└── agend.md                        # Operational memory / agenda
```

## Configuration

All training scripts accept OmegaConf overrides as positional arguments:

```bash
uv run python scripts/train_dynamics.py \
  train.epochs=200 \
  train.batch_size=32 \
  train.regime=joint_ema \
  model.hidden_channels=128
```

### Training Regimes

| Regime | AE weights | AE loss | Description |
|--------|-----------|---------|-------------|
| `frozen` | Locked | No | AE is a fixed feature extractor |
| `joint_no_ema` | Updated | Yes | AE co-trained with dynamics |
| `joint_ema` | Updated (EMA) | Yes | AE updated via exponential moving average |

### Device Auto-Detection

All configs default to `device: auto`, which resolves to:
1. `cuda` if NVIDIA GPU available
2. `mps` if Apple Silicon available
3. `cpu` otherwise

Override manually: `train.device=cpu` or `train.device=cuda:1`.

---

## Tracking

- **W&B**: Pass `--wandb` to any benchmark script
- **Local summaries**: Each run produces `summary.json` + `summary.md` in its output directory
- **Acquisition curves**: Auto-generated PNG plots comparing strategies

## Development

```bash
uv run pytest                  # run all tests
uv run ruff check src/ tests/  # lint
uv run ruff format src/ tests/ # format
```

## References

- Flow Matching: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
- DAS-PINN: Tang et al., "DAS: A deep adaptive sampling method for solving PDEs"
- KRnet: Tang et al., "Deep density estimation via KRnet"
