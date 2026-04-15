# Spec 4 — Architecture logicielle, Ray + Slurm + W&B, orchestration cluster

## 1. Objectif de ce document

Ce document décrit l’architecture logicielle et système du projet.

Les objectifs sont :
- simplicité,
- modularité,
- exécution distribuée,
- reprise sur incident,
- reproductibilité scientifique,
- facilité de modification.

---

## 2. Principes d’architecture

Le code doit être organisé autour des responsabilités scientifiques.

### Principes directeurs
- modules petits et cohérents,
- interfaces explicites,
- configuration centralisée,
- séparation claire entre science, orchestration et scripts,
- composants testables indépendamment.

---

## 3. Structure de repo recommandée

```text
repo/
  README.md
  pyproject.toml
  configs/
    data/
    solver/
    model/
    train/
    acquisition/
    evaluation/
    cluster/
    experiment/
  src/
    pdewm/
      data/
      solvers/
      models/
        representations/
        dynamics/
        auxiliaries/
      acquisition/
      baselines/
      training/
      evaluation/
      orchestration/
      logging/
      utils/
  scripts/
    generate_data.py
    train_repr.py
    train_dynamics.py
    train_baselines.py
    run_online_loop.py
    evaluate.py
    launch_ray_cluster.py
  tests/
    unit/
    integration/
    regression/
    performance/
  slurm/
    data_generation.sbatch
    train_model.sbatch
    ray_head.sbatch
    ray_workers.sbatch
    online_loop.sbatch
  docs/
    architecture/
    experiment_registry/
```

---

## 4. Responsabilités des modules

## 4.1. `data/`
Responsable de :
- lecture/écriture datasets,
- split management,
- normalisation,
- manifestes de versions,
- replay buffers.

## 4.2. `solvers/`
Responsable de :
- solveurs PDE,
- wrappers unifiés,
- validation de stabilité,
- collecte métadonnées,
- batching éventuel.

## 4.3. `models/representations/`
Responsable de :
- encodeurs,
- décodeurs,
- losses AE,
- visualisations reconstruction.

## 4.4. `models/dynamics/`
Responsable de :
- encodeur contexte physique,
- transition conditionnelle,
- rollout latent,
- ensembles.

## 4.5. `models/auxiliaries/`
Responsable de :
- crash predictor,
- adversarial PDE classifier,
- density model latent,
- calibrateurs d’incertitude.

## 4.6. `acquisition/`
Responsable de :
- candidate generation,
- scoring,
- selection diverse,
- politiques batch bandit,
- suivi du budget solveur.

## 4.7. `baselines/`
Responsable de :
- FNO,
- F-FNO,
- CNN/U-Net AR,
- POD,
- AE baselines.

## 4.8. `training/`
Responsable de :
- trainer AE,
- trainer dynamics,
- trainer baselines,
- trainer acquisition policy,
- checkpoints,
- callbacks.

## 4.9. `evaluation/`
Responsable de :
- métriques,
- suites benchmark,
- figures,
- tableaux,
- exports publication.

## 4.10. `orchestration/`
Responsable de :
- acteurs Ray,
- files d’attente,
- coordination entraîneur/solveur,
- reprise sur incident,
- synchronisation datasets et checkpoints.

---

## 5. Convention de configuration

## 5.1. Configuration centralisée
Utiliser Hydra ou équivalent.

Chaque run doit être entièrement décrit par :
- configuration de données,
- configuration solveur,
- configuration modèle,
- configuration entraînement,
- configuration acquisition,
- configuration cluster.

## 5.2. Principes
- configs immuables une fois le run lancé,
- sauvegarde automatique avec le run,
- overrides explicites depuis CLI,
- validation stricte des champs.

## 5.3. Niveaux de config

### Niveau projet
Defaults globaux.

### Niveau benchmark
Burgers, KS, NS, etc.

### Niveau modèle
AE, dynamics, baselines.

### Niveau expérience
Combinaison finale qui référence les sous-configs.

---

## 6. Logging et W&B

## 6.1. Ce qui doit être loggué

### Configs
- config complète,
- git hash,
- seed,
- dataset version,
- solver version.

### Métriques
- losses,
- reconstruction,
- rollout metrics,
- acquisition metrics,
- coût solveur,
- temps mur.

### Artefacts
- checkpoints,
- manifests datasets,
- tableaux d’évaluation,
- figures benchmark.

## 6.2. Convention W&B recommandée

### Project
`pde-world-model`

### Group
- par benchmark,
- ou par famille d’ablations.

### Job type
- `data-gen`
- `train-ae`
- `train-dynamics`
- `train-baseline`
- `online-acquisition`
- `eval`

### Tags
- PDE,
- dimension,
- baseline,
- online/offline,
- multi-PDE.

---

## 7. Orchestration Ray

## 7.1. Pourquoi Ray

Ray est utile ici pour :
- piloter une flotte de solveurs,
- orchestrer la boucle online,
- séparer entraînement et acquisition,
- garder une API Python unifiée.

## 7.2. Acteurs principaux

### TrainerActor
Responsable de :
- charger dataset courant,
- entraîner ou fine-tuner le modèle,
- publier checkpoints,
- exposer stats du modèle.

### AcquisitionActor
Responsable de :
- lire le dernier checkpoint,
- générer/scorer les candidats,
- constituer les batches,
- suivre le budget.

### SolverWorkerActor
Responsable de :
- recevoir un candidat,
- décoder si nécessaire,
- lancer le solveur,
- retourner résultats et statut.

### DatasetWriterActor
Responsable de :
- écrire les nouveaux samples,
- maintenir le manifeste,
- garantir cohérence et atomicité.

### EvaluatorActor
Responsable de :
- lancer benchmarks fixes,
- produire résultats comparables,
- écrire tableaux et figures.

---

## 8. Boucle online distribuée

## 8.1. Étapes
1. `TrainerActor` publie le checkpoint courant.
2. `AcquisitionActor` construit un pool de candidats.
3. `AcquisitionActor` sélectionne un batch selon budget.
4. batch dispatché aux `SolverWorkerActor`.
5. les workers renvoient nouveaux samples et métadonnées.
6. `DatasetWriterActor` persiste les résultats.
7. `TrainerActor` recharge le nouveau dataset et fine-tune.
8. `EvaluatorActor` lance une évaluation périodique.
9. boucle jusqu’à épuisement du budget.

## 8.2. Granularité recommandée

### Unité de travail solveur
Un candidat = une unité de travail.

### Taille de batch
Déterminée par :
- mémoire disponible,
- coût solveur moyen,
- overhead Ray,
- objectif de latence.

---

## 9. Reprise et robustesse

## 9.1. Tout doit être relançable

Le système doit tolérer :
- crash worker,
- arrêt cluster,
- timeout solveur,
- corruption partielle d’un lot.

## 9.2. Mécanismes obligatoires
- checkpoints réguliers,
- écriture atomique des samples,
- journal des batches soumis,
- identifiants uniques par tâche solveur,
- mécanisme de retry borné,
- marquage explicite des échecs définitifs.

## 9.3. Invariants
- jamais de sample partiellement écrit comme valide,
- jamais de duplication silencieuse,
- jamais de mélange entre versions de modèles sans traçabilité.

---

## 10. Slurm

## 10.1. Rôles des scripts Slurm

### `data_generation.sbatch`
Génération offline massive.

### `train_model.sbatch`
Entraînement AE/dynamics/baselines.

### `ray_head.sbatch`
Lance le nœud head Ray.

### `ray_workers.sbatch`
Ajoute les workers.

### `online_loop.sbatch`
Lance la boucle online principale une fois le cluster prêt.

## 10.2. Variables critiques à exposer
- nombre de nœuds,
- CPU par worker,
- GPU par trainer,
- mémoire par tâche,
- durée max,
- file système utilisée,
- répertoire scratch.

## 10.3. Stratégie recommandée

### Développement
1 nœud, quelques workers, solveurs petits.

### Préproduction
plusieurs nœuds, contrôle I/O, monitoring du throughput.

### Production scientifique
cluster Ray stable + budget solveur explicite.

---

## 11. Gestion des artefacts

## 11.1. Types d’artefacts
- datasets,
- checkpoints,
- manifests,
- figures,
- tableaux benchmark.

## 11.2. Emplacement logique
- scratch pour génération transitoire,
- stockage persistant pour résultats versionnés,
- W&B pour indexation et traçabilité légère.

## 11.3. Convention de nommage
Inclure dans chaque artefact :
- benchmark,
- version,
- seed,
- date,
- git hash court.

---

## 12. Interfaces logiques minimales

## 12.1. Solveur
`simulate(initial_state, context, num_steps, dt, options) -> result`

## 12.2. Représentation
`encode(state) -> latent`
`decode(latent) -> state`

## 12.3. Dynamique
`predict_next(latent, context) -> latent_next`
`rollout(latent0, context, horizon) -> latent_traj`

## 12.4. Acquisition
`propose(pool_context) -> candidates`
`score(candidates, model_state) -> scores`
`select(candidates, scores, budget) -> batch`

## 12.5. Dataset writer
`append(samples, metadata) -> dataset_version`

---

## 13. Performance et observabilité

## 13.1. Métriques système
- occupation CPU,
- occupation GPU,
- temps moyen solveur,
- temps d’attente file,
- débit d’écriture,
- débit d’entraînement.

## 13.2. Points de vigilance
- I/O partagé saturé,
- trop petits jobs solveur,
- surcoût Ray > bénéfice,
- checkpoints trop fréquents,
- contention dataset writer.

---

## 14. Décisions à trancher tôt

- Ray nécessaire dès le début ou non,
- solveurs CPU ou GPU,
- écriture par sample ou par batch,
- fine-tune synchrone ou asynchrone,
- centralisation ou non du dataset writer,
- scratch local vs stockage réseau.

---

## 15. Critères de validation de l’architecture

L’architecture est validée si :
- elle supporte les benchmarks sans dette technique excessive,
- chaque module est remplaçable sans casser le reste,
- la boucle online est robuste aux échecs,
- les runs sont relançables et comparables,
- les coûts système sont mesurables et reportables dans le papier.

