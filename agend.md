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
- le sprint 5 heuristique est désormais entamé: active sampling d'états, versioning dataset online, réentraînement du comité, smoke test court validé;
- le protocole corrigé est maintenant codé:
  - régimes `frozen`, `joint_no_ema`, `joint_ema` dans `train_dynamics.py`;
  - `EMA` sur l'AE pour la cible latente et l'acquisition en mode `joint_ema`;
  - métriques trajectoire communes val/test avec `RMSE`, `NRMSE` et quantiles;
  - nouveau runner `scripts/run_worldmodel_benchmark.py` à budget online en transitions solveur;
- un runner long chaîné existe maintenant pour lancer les campagnes longues d'un seul bloc, avec script de lancement en arrière-plan.
- campagne longue réellement lancée le `2026-04-15` en session PTY persistante `91896`, log courant: `artifacts/launches/long_campaign_tty_20260415_210509.log`.
- le logging `wandb` est maintenant branché sur les trainers et runners séquentiels; la campagne longue déjà partie avant ce patch n'en bénéficie pas.
- smoke benchmark protocolaire validé dans `artifacts/runs/protocol_smoke`.

## Règles de conduite du projet

- ne pas sauter directement au multi-PDE ou à Ray sans avoir un pipeline 1D stable;
- respecter l'invariance PDE du latent dans le design principal;
- garder les datasets et les expériences versionnés;
- avancer par commits courts et réversibles.

## Reprise recommandée au prochain échange

1. vérifier l'état git et le contenu de `IMPLEMENTATION_TRACKER.md`;
2. considérer la campagne longue `91896` comme exploration historique sous ancien protocole;
3. pour toute nouvelle campagne de référence, utiliser `scripts/run_worldmodel_benchmark.py` ou `scripts/run_long_campaign.py` mis à jour;
4. relancer un benchmark long multi-seeds avec budget en transitions solveur et suivre les courbes générées;
5. mettre à jour ce fichier et le tracker après chaque bloc livré;
6. conserver la discipline commit par commit avec vérification locale avant chaque commit.
