# agend

Mémoire persistante de pilotage. Ce fichier résume où reprendre le travail sans relire tout l'historique de conversation.

## Vision de livraison

- priorité absolue: verrouiller un POC 1D scientifiquement propre avant d'ouvrir le front 2D;
- progression imposée par les specs: data -> AE -> dynamique -> baselines -> acquisition -> multi-PDE;
- tous les livrables doivent rester traçables, testés et faciles à revenir en arrière via commits lisibles.

## État courant

- le dépôt contenait uniquement les notes de cadrage;
- l'ordre d'implémentation retenu suit `world_model_pde_publication_grade_plan.md`, section `10.1`;
- les benchmarks de départ retenus sont Burgers 1D et KS 1D;
- les fichiers de suivi persistants ont été créés pour éviter de repartir de zéro;
- le sprint 1 est livré: solveurs 1D, writer dataset, manifests et génération offline validés localement;
- la gestion de configuration est assurée par un loader `OmegaConf` compatible Hydra, car Hydra casse à l'exécution sous Python 3.14 dans cet environnement;
- le sprint 2 est livré: AE 1D, losses L1/L2/gradient/spectrale, trainer, tests et smoke train;
- le setup `uv` est désormais la voie standard: Python 3.12, `uv.lock`, bootstrap `cpu`/`cuda`, tests et smoke runs déjà validés;
- le sprint 3 est livré: dynamique latente 1D conditionnée par contexte physique, avec trainer et smoke train réel sur Burgers;
- le sprint 4 est livré: baselines 1D directes + runner séquentiel, avec première campagne Burgers déjà exécutée.
- le sprint 5 heuristique est livré: active sampling d'états, versioning dataset online, réentraînement du comité, smoke test court validé;
- le protocole corrigé est maintenant codé:
  - régimes `frozen`, `joint_no_ema`, `joint_ema` dans `train_dynamics.py`;
  - `EMA` sur l'AE pour la cible latente et l'acquisition en mode `joint_ema`;
  - métriques trajectoire communes val/test avec `RMSE`, `NRMSE` et quantiles;
  - nouveau runner `scripts/run_worldmodel_benchmark.py` à budget online en transitions solveur;
- auto-détection CUDA/MPS: `resolve_device("auto")` dans `src/pdewm/utils/device.py`, tous les configs/scripts mis à jour;
- wandb corrigé: `enabled: true` par défaut, runners explicitement `false` quand `--wandb` n'est pas passé;
- benchmark challenging GPU ajouté: `run_worldmodel_challenging_benchmark.py` (RTX 2070 Super, grid 256, ensemble 5, multi-PDE, multi-seed);
- acquisition générative par flow matching implémentée:
  - `src/pdewm/acquisition/generative.py`: Conditional Flow Matching en espace latent compressé 512-dim, ConvAttention velocity net;
  - 3 nouvelles stratégies: `generative_loss_weighted`, `generative_uniform`, `generative_combined`;
  - `scripts/run_worldmodel_generative_benchmark.py`: benchmark 9 stratégies;
  - 33 tests passent (24 originaux + 9 nouveaux).
- updates benchmark génératif récentes:
  - regroupement W&B par type d'expérience via `--wandb-group` (préfixe de campagne + sous-groupe sémantique);
  - mode parallèle pour entraînement des membres d'ensemble: `--ensemble-train-mode parallel --ensemble-max-parallel N`;
  - surcharge `--ensemble-size` respectée même sous `--benchmark-profile realistic_1d`;
  - quantiles NRMSE (`p25/p50/p75`) exposés dans le summary benchmark et envoyés explicitement à W&B;
  - plot rollout en échelle logarithmique pour meilleure séparation visuelle.

## Règles de conduite du projet

- ne pas sauter directement au multi-PDE ou à Ray sans avoir un pipeline 1D stable;
- respecter l'invariance PDE du latent dans le design principal;
- garder les datasets et les expériences versionnés;
- avancer par commits courts et réversibles.

## Reprise recommandée au prochain échange

1. vérifier l'état git et le contenu de `IMPLEMENTATION_TRACKER.md`;
2. exécuter `run_worldmodel_generative_benchmark.py` sur machine GPU avec wandb pour comparer les 9 stratégies;
3. exécuter `run_worldmodel_challenging_benchmark.py` sur machine GPU multi-seeds;
4. analyser l'impact du temperature flow matching (ablation T=0.5, 1.0, 2.0);
5. étendre les benchmarks à `ks_1d`;
6. mettre à jour ce fichier et le tracker après chaque bloc livré;
7. conserver la discipline commit par commit avec vérification locale avant chaque commit.
8. pour GPU 8GB (RTX 2070 Super), préférer `--ensemble-size 3` et `--ensemble-max-parallel 1..2` selon VRAM disponible.
