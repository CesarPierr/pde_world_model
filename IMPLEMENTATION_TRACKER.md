# Implementation Tracker

Ce fichier suit l'état réel d'implémentation du dépôt par rapport aux notes de cadrage. Il doit être mis à jour à chaque lot significatif.

## Légende

- `done`: implémenté et vérifié localement
- `in_progress`: en cours d'implémentation
- `planned`: prévu mais non codé
- `blocked`: dépendance ou arbitrage manquant

## Macro-roadmap

| Bloc | Cible | Statut | Notes |
| --- | --- | --- | --- |
| Repo scaffold | structure, packaging, conventions, mémoire persistante | in_progress | en cours dans le bootstrap initial |
| Sprint 1 | solveurs 1D + génération dataset + QA | planned | Burgers 1D puis KS 1D |
| Sprint 2 | AE 1D + losses reconstruction | planned | L1/L2/gradient/spectral |
| Sprint 3 | dynamique latente 1D single-PDE | planned | transition résiduelle conditionnée |
| Sprint 4 | baselines 1D | planned | CNN AR, U-Net AR, FNO, POD+MLP |
| Sprint 5 | acquisition online 1D | planned | memory bank, uncertainty, diversity, risk |
| Sprint 6 | multi-paramètre / multi-PDE 1D | planned | shared latent + invariance |
| Sprint 7+ | extension 2D et orchestration distribuée | planned | diffusion-reaction puis NS |

## Exigences de cadrage suivies

| Exigence | Source | Statut | Notes |
| --- | --- | --- | --- |
| structure du repo par responsabilité scientifique | `spec_ray_slurm_wandb_architecture.md` | in_progress | structure créée |
| suivi persistant de l'avancement | demande utilisateur | in_progress | `IMPLEMENTATION_TRACKER.md` + `agend.md` |
| progression commit par commit | demande utilisateur | in_progress | stratégie adoptée dès ce cycle |
| POC 1D avant multi-PDE/2D | `world_model_pde_publication_grade_plan.md` | planned | ordre d'exécution conservé |
| AE non conditionné par PDE | `spec_model_and_losses.md` | planned | à respecter dans l'implémentation |
| dynamique conditionnée par contexte physique | `spec_model_and_losses.md` | planned | à respecter |
| acquisition initiale en contextual batch bandit | `spec_online_acquisition_and_rl.md` | planned | pas encore codé |

## Journal synthétique

### 2026-04-15
- dépôt audité: uniquement des documents markdown de cadrage;
- architecture cible extraite depuis les cinq specs;
- bootstrap du repo lancé avec mémoire persistante et structure standardisée.

## Prochaines actions fermes

1. implémenter l'interface solveur commune et les contextes PDE.
2. livrer Burgers 1D et KS 1D avec validations de stabilité et reproductibilité.
3. ajouter writer dataset/manifests et script de génération offline.
4. démarrer l'AE 1D avec losses de reconstruction.

