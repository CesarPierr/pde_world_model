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
| Sprint 5 | acquisition online 1D | done | protocole corrigé implémenté: budget en transitions solveur, ablation `frozen/joint_no_ema/joint_ema`, acquisitions `offline/random/uncertainty/diversity/uncertainty+diversity/ours`, courbes automatiques |
| Sprint 5b | acquisition générative (flow matching) | done | Conditional Flow Matching en espace latent compressé (512-dim), ConvAttention velocity net, loss-weighted training, 3 nouvelles stratégies (`generative_loss_weighted/uniform/combined`), benchmark dédié |
| Infra | auto-device + wandb fix | done | `resolve_device("auto")` CUDA > MPS > CPU, wandb enabled par défaut, correction du parsing CLI `--wandb` |
| Infra | challenging benchmark GPU | done | `run_worldmodel_challenging_benchmark.py` pour RTX 2070 Super: grid 256, AE latent 64, ensemble 5, budget 512, multi-PDE + multi-seed |
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
| acquisition initiale en contextual batch bandit | `spec_online_acquisition_and_rl.md` | done | version heuristique score-based livrée + sampler génératif flow matching (DAS-PINN/KRnet-inspired); policy apprise et crash predictor dédié restent à faire |
| solveurs 1D déterministes avec QA | `spec_benchmarks_and_baselines.md` | done | tests solveurs + génération offline validée |
| config centralisée de type Hydra ou équivalent | `spec_ray_slurm_wandb_architecture.md` | done | loader YAML `OmegaConf` avec overrides `key=value`, choisi pour éviter l'incompatibilité Hydra/Python 3.14 de l'environnement |
| losses reconstruction L1/L2/gradient/spectrale | `spec_model_and_losses.md` | done | implémentées et testées |
| surfit court batch AE | `world_model_pde_publication_grade_plan.md` | done | test d'overfit sur petit batch + smoke train sur dataset offline |
| setup reproductible local et future machine GPU | demande utilisateur | done | workflow `uv` CPU/CUDA documenté et verrouillé via `uv.lock` |
| baselines 1D comparables sous protocole commun | `spec_benchmarks_and_baselines.md` | done | CNN AR, U-Net AR, FNO 1D, POD+MLP, même dataset et même horizon de rollout |
| suivi d'expériences et métriques avec wandb | `spec_ray_slurm_wandb_architecture.md` + demande utilisateur | done | instrumentation `wandb` optionnelle branchée dans `train_autoencoder.py`, `train_baseline.py`, `train_dynamics.py` et propagée dans les runners séquentiels |
| métriques communes sur trajectoires d'évaluation | demande utilisateur + protocole corrigé | done | split unique d'éval, métriques `one-step` et `rollout`, `RMSE/NRMSE`, quantiles `p10/p25/p50/p75/p90/p95/p99`, rollouts `best/median/worst` et stats `loss_min/loss_max/loss_std` dans les summaries/W&B |
| budget d'acquisition fixé en transitions solveur | demande utilisateur + protocole corrigé | done | unité `1 transition = (state,next_state)`, budget online distinct du dataset offline, comptage exact et pertes de transitions journalisées |
| joint training AE+dynamics avec EMA | demande utilisateur + `spec_model_and_losses.md` | done | régimes `frozen`, `joint_no_ema`, `joint_ema`, checkpoints `student/EMA`, reprise de training supportée |

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
- logging `wandb` ajouté: dépendance projet, wrapper optionnel, logs epoch/final summary, groupage via runners longs et smoke run local validé en mode `offline`.
- protocole world model corrigé implémenté:
  - `train_dynamics.py` supporte `frozen`, `joint_no_ema`, `joint_ema`;
  - les summaries world model et baseline incluent maintenant des métriques trajectoire partagées;
  - nouveau runner `run_worldmodel_benchmark.py` avec ablation des régimes, sélection du régime, benchmark des stratégies d'acquisition à budget en transitions solveur et génération automatique des courbes;
  - smoke benchmark validé sur `artifacts/runs/protocol_smoke`.
- auto-détection GPU/MPS implémentée: `src/pdewm/utils/device.py` avec `resolve_device("auto")` (CUDA > MPS > CPU); tous les configs YAML passent de `device: cpu` à `device: auto`; tous les scripts d'orchestration mis à jour.
- correction wandb: `wandb.yaml` passe à `enabled: true` par défaut; les runners envoient `logging.wandb.enabled=false` explicitement quand `--wandb` n'est pas passé; la cause racine (flag `--wandb` parsé comme partie du chemin `--output-root`) est résolue.
- `run_worldmodel_challenging_benchmark.py` ajouté: benchmark GPU (RTX 2070 Super, 16 cores) avec grid 256, 64 trajectoires train, 48 steps, AE base=32/latent=64/mults=[1,2,4,8], dynamics hidden=64, ensemble 5, budget online 512, support multi-PDE (`--pdes`) et multi-seed (`--seeds`).
- acquisition générative par flow matching implémentée:
  - `src/pdewm/acquisition/generative.py`: Conditional Flow Matching en espace latent compressé (512-dim fixe via `LatentCompressor`/`LatentDecompressor`);
  - velocity network: `ConvAttentionVelocityNet` — 1D conv + FiLM time conditioning + self-attention spatiale (pas un gros MLP);
  - training loss-weighted: probabilité d'échantillonnage ∝ `transition_loss^temperature`;
  - 3 nouvelles stratégies: `generative_loss_weighted`, `generative_uniform` (ablation T=0), `generative_combined` (flow + heuristique fusionnés);
  - `scripts/run_worldmodel_generative_benchmark.py`: benchmark 9 stratégies (6 heuristiques + 3 génératives);
  - `tests/test_generative.py`: 9 tests unitaires (compressor, decompressor, velocity net, sampler fit/sample, temperature=0);
  - tous les 33 tests passent.
- améliorations benchmark génératif (W&B + orchestration + métriques):
  - logique `--wandb-group` harmonisée: préfixe de campagne + sous-groupes sémantiques (`ae`, `abl/<regime>`, `acq/<strategy>/<regime>`), appliquée aux scripts benchmark;
  - mode d'entraînement parallèle du comité ajouté dans `run_worldmodel_generative_benchmark.py` (`--ensemble-train-mode`, `--ensemble-max-parallel`);
  - `--ensemble-size` explicite préservé même avec le profil `realistic_1d` (ne repasse plus automatiquement à 5 si surchargé);
  - résumé benchmark enrichi avec quantiles NRMSE (`p25/p50/p75`) pour one-step et rollout;
  - export explicite des quantiles finaux vers W&B (`trajectory_eval/final/*`) + `ae_eval/final/*`;
  - courbe rollout `eval_nrmse_curves.png` en échelle log sur l'axe Y.

## Prochaines actions fermes

1. exécuter `run_worldmodel_generative_benchmark.py` sur machine GPU avec wandb pour comparer les 9 stratégies d'acquisition.
2. exécuter `run_worldmodel_challenging_benchmark.py` sur machine GPU multi-seeds pour résultats de publication.
3. analyser l'impact du temperature sur le flow matching (ablation T=0.5, 1.0, 2.0).
4. étendre les benchmarks à `ks_1d` (Kuramoto-Sivashinsky).
5. durcir Sprint 5 avec un crash predictor dédié et une policy apprise si le gain heuristique/génératif est confirmé.
6. ouvrir Sprint 6: multi-PDE shared latent + invariance.
