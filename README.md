# PDE World Model

World model latent conditionnel multi-PDE, aligné sur les notes de cadrage du dépôt et structuré pour une montée en charge progressive:

1. POC 1D sur Burgers et Kuramoto-Sivashinsky.
2. Montée 2D sur diffusion-reaction puis Navier-Stokes.
3. Extension multi-PDE avec acquisition online en espace latent.

## Principes directeurs

- encodeur/décodeur non conditionnés par la PDE;
- dynamique latente conditionnée par le contexte physique;
- datasets versionnés et reproductibles;
- benchmark rigoureux avec baselines comparables;
- orchestration pensée dès le départ pour l'offline puis Ray/Slurm.

## Structure

Le dépôt suit la structure recommandée dans `world_model_pde_publication_grade_plan.md` et `spec_ray_slurm_wandb_architecture.md`:

- `configs/`: configurations Hydra/YAML par responsabilité;
- `src/pdewm/`: code source;
- `scripts/`: points d'entrée de génération, entraînement et évaluation;
- `tests/`: unitaires, intégration, régression, performance;
- `slurm/`: gabarits cluster;
- `docs/`: documentation d'architecture et registre d'expériences;
- `artifacts/`: schémas et exemples.

## Pilotage du développement

Deux fichiers racine servent de mémoire persistante et de suivi d'implémentation:

- `IMPLEMENTATION_TRACKER.md`: état détaillé des livrables par sprint et par module;
- `agend.md`: journal de bord, prochaines actions et mémoire opérationnelle d'une conversation à l'autre.

## Démarrage visé

Le dépôt est initialisé pour une implémentation incrémentale, commit par commit:

1. socle du projet et tracking;
2. solveurs et génération de données 1D;
3. autoencodeur 1D;
4. dynamique latente 1D;
5. baselines;
6. acquisition online;
7. extension 2D et orchestration distribuée.

## Environnement `uv`

Le dépôt cible un environnement `uv` avec Python `3.12`.

- macOS / CPU: `TORCH_BACKEND=cpu ./scripts/bootstrap_uv.sh`
- Linux / NVIDIA CUDA 12.4: `TORCH_BACKEND=cu124 ./scripts/bootstrap_uv.sh`
- tests: `uv run pytest`

La documentation détaillée est dans [docs/architecture/setup_uv.md](/Users/pierre/pde_world_model/docs/architecture/setup_uv.md:1).
