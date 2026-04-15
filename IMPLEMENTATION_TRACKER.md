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
| Environnement uv | Python géré, lockfile, bootstrap CPU/CUDA, doc d'usage | done | `uv` + Python 3.12 + script `bootstrap_uv.sh` + `uv.lock` |
| Sprint 1 | solveurs 1D + génération dataset + QA | done | Burgers 1D et KS 1D, writer Zarr, manifests, script offline, tests unitaires et smoke tests |
| Sprint 2 | AE 1D + losses reconstruction | done | AE 1D résiduel, losses L1/L2/gradient/spectrale, trainer, tests et smoke train |
| Sprint 3 | dynamique latente 1D single-PDE | done | dataset de fenêtres, encodeur contexte physique, transition FiLM, trainer, tests et smoke train réel |
| Sprint 4 | baselines 1D | done | CNN AR, U-Net AR, FNO 1D et POD+MLP, trainer commun, runner séquentiel, campagne Burgers 1D lancée |
| Sprint 5 | acquisition online 1D | in_progress | boucle heuristique memory-bank + uncertainty + novelty + risk + réentraînement de comité, smoke test validé |
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
| dynamique conditionnée par contexte physique | `spec_model_and_losses.md` | done | encodeur `pde_id + paramètres + dt`, transition FiLM résiduelle |
| acquisition initiale en contextual batch bandit | `spec_online_acquisition_and_rl.md` | in_progress | version heuristique score-based livrée; policy apprise et crash predictor dédié restent à faire |
| solveurs 1D déterministes avec QA | `spec_benchmarks_and_baselines.md` | done | tests solveurs + génération offline validée |
| config centralisée de type Hydra ou équivalent | `spec_ray_slurm_wandb_architecture.md` | done | loader YAML `OmegaConf` avec overrides `key=value`, choisi pour éviter l'incompatibilité Hydra/Python 3.14 de l'environnement |
| losses reconstruction L1/L2/gradient/spectrale | `spec_model_and_losses.md` | done | implémentées et testées |
| surfit court batch AE | `world_model_pde_publication_grade_plan.md` | done | test d'overfit sur petit batch + smoke train sur dataset offline |
| setup reproductible local et future machine GPU | demande utilisateur | done | workflow `uv` CPU/CUDA documenté et verrouillé via `uv.lock` |
| baselines 1D comparables sous protocole commun | `spec_benchmarks_and_baselines.md` | done | CNN AR, U-Net AR, FNO 1D, POD+MLP, même dataset et même horizon de rollout |

## Journal synthétique

### 2026-04-15
- dépôt audité: uniquement des documents markdown de cadrage;
- architecture cible extraite depuis les cinq specs;
- bootstrap du repo lancé avec mémoire persistante et structure standardisée;
- setup `uv` ajouté: Python projet 3.12, bootstrap script macOS CPU / Linux CUDA, lockfile généré, docs d'installation;
- sprint 1 implémenté: contextes PDE, solveurs Burgers/KS 1D, schémas dataset, writer Zarr, script de génération offline;
- validations effectuées: compilation Python, `pytest tests/unit`, smoke tests de génération offline sur Burgers et KS;
- sprint 2 implémenté: AE 1D résiduel, losses de reconstruction, dataset torch, trainer et script d'entraînement;
- validations Sprint 2: `pytest tests/unit tests/integration/test_autoencoder_overfit.py` et smoke train réel sur mini dataset Burgers offline;
- validations `uv`: `uv run pytest -q`, génération offline Burgers via `uv run`, puis entraînement AE réel avec amélioration val d'environ 54%.
- sprint 3 implémenté: `TransitionWindowDataset`, encodeur de contexte physique, dynamique latente 1D conditionnée par FiLM, script `train_dynamics.py`;
- validations Sprint 3: `uv run pytest -q` vert, smoke train réel `AE gelé -> dynamics` avec meilleure `val_loss` ~ `0.108` sur Burgers 1D.
- sprint 4 implémenté: modèles `cnn_ar_1d`, `unet_ar_1d`, `fno_1d`, `pod_mlp_1d`, trainer baseline commun et script séquentiel `run_sprint4_experiments.py`;
- campagne Sprint 4 Burgers 1D lancée séquentiellement sur Mac CPU, résultats initiaux:
  - `fno_1d` meilleur: test one-step ~ `2.8e-06`, rollout ~ `7.0e-06`;
  - `unet_ar_1d` second: test one-step ~ `9.1e-05`, rollout ~ `2.25e-04`;
  - `cnn_ar_1d` derrière: test one-step ~ `6.9e-04`, rollout ~ `1.67e-03`;
  - `pod_mlp_1d` nettement plus faible: test one-step ~ `7.48e-03`, rollout ~ `7.75e-03`.
- sprint 5 initial implémenté: acquisition heuristique en espace d'états/latent, enrichissement dataset versionné, réentraînement séquentiel d'un comité de dynamique;
- validations Sprint 5: smoke test court `run_worldmodel_active_sampling.py` sur Burgers 1D avec 1 itération online, 2 membres d'ensemble, 12 nouveaux samples acquis et réentraînement réussi.

## Prochaines actions fermes

1. laisser tourner la campagne longue chaînée et revenir sur les logs/résumés.
2. ajouter une comparaison explicite `baselines longues vs world model long vs world model + active sampling`.
3. durcir Sprint 5 avec un crash predictor dédié et une policy apprise si le gain heuristique est confirmé.
4. étendre la même logique à `ks_1d`.
