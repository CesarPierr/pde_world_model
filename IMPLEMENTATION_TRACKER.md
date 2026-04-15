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
| Repo scaffold | structure, packaging, conventions, mémoire persistante | done | squelette, packaging, tracker et agenda en place |
| Sprint 1 | solveurs 1D + génération dataset + QA | done | Burgers 1D et KS 1D, writer Zarr, manifests, script offline, tests unitaires et smoke tests |
| Sprint 2 | AE 1D + losses reconstruction | done | AE 1D résiduel, losses L1/L2/gradient/spectrale, trainer, tests et smoke train |
| Sprint 3 | dynamique latente 1D single-PDE | planned | transition résiduelle conditionnée |
| Sprint 4 | baselines 1D | planned | CNN AR, U-Net AR, FNO, POD+MLP |
| Sprint 5 | acquisition online 1D | planned | memory bank, uncertainty, diversity, risk |
| Sprint 6 | multi-paramètre / multi-PDE 1D | planned | shared latent + invariance |
| Sprint 7+ | extension 2D et orchestration distribuée | planned | diffusion-reaction puis NS |

## Exigences de cadrage suivies

| Exigence | Source | Statut | Notes |
| --- | --- | --- | --- |
| structure du repo par responsabilité scientifique | `spec_ray_slurm_wandb_architecture.md` | done | structure créée et prête à être remplie par module |
| suivi persistant de l'avancement | demande utilisateur | done | `IMPLEMENTATION_TRACKER.md` + `agend.md` |
| progression commit par commit | demande utilisateur | done | bootstrap déjà commité, sprint 1 prêt au commit séparé |
| POC 1D avant multi-PDE/2D | `world_model_pde_publication_grade_plan.md` | done | ordre d'exécution conservé dans le code et le suivi |
| AE non conditionné par PDE | `spec_model_and_losses.md` | done | interface `encode/decode` sans dépendance à la PDE ni skip externe |
| dynamique conditionnée par contexte physique | `spec_model_and_losses.md` | planned | à respecter |
| acquisition initiale en contextual batch bandit | `spec_online_acquisition_and_rl.md` | planned | pas encore codé |
| solveurs 1D déterministes avec QA | `spec_benchmarks_and_baselines.md` | done | tests solveurs + génération offline validée |
| config centralisée de type Hydra ou équivalent | `spec_ray_slurm_wandb_architecture.md` | done | loader YAML `OmegaConf` avec overrides `key=value`, choisi pour éviter l'incompatibilité Hydra/Python 3.14 de l'environnement |
| losses reconstruction L1/L2/gradient/spectrale | `spec_model_and_losses.md` | done | implémentées et testées |
| surfit court batch AE | `world_model_pde_publication_grade_plan.md` | done | test d'overfit sur petit batch + smoke train sur dataset offline |

## Journal synthétique

### 2026-04-15
- dépôt audité: uniquement des documents markdown de cadrage;
- architecture cible extraite depuis les cinq specs;
- bootstrap du repo lancé avec mémoire persistante et structure standardisée;
- sprint 1 implémenté: contextes PDE, solveurs Burgers/KS 1D, schémas dataset, writer Zarr, script de génération offline;
- validations effectuées: compilation Python, `pytest tests/unit`, smoke tests de génération offline sur Burgers et KS;
- sprint 2 implémenté: AE 1D résiduel, losses de reconstruction, dataset torch, trainer et script d'entraînement;
- validations Sprint 2: `pytest tests/unit tests/integration/test_autoencoder_overfit.py` et smoke train réel sur mini dataset Burgers offline.

## Prochaines actions fermes

1. implémenter la dynamique latente 1D conditionnée par le contexte physique.
2. ajouter le conditionnement `pde_id + paramètres + dt` avec une variante principale FiLM.
3. brancher les losses `latent one-step`, `phys one-step` et rollout court.
4. écrire le test de surfit sur une petite trajectoire et un smoke train single-PDE.
