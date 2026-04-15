# Spec 1 — Benchmarks, solveurs, datasets et baselines

## 1. Objectif de ce document

Ce document fixe le **cadre expérimental de référence** du projet afin de garantir :
- comparabilité avec la littérature,
- reproductibilité,
- crédibilité scientifique,
- simplicité d’implémentation progressive.

Il définit :
- les PDEs retenues,
- les solveurs de référence,
- les conventions de génération de données,
- les splits,
- les métriques de benchmark,
- les baselines à implémenter.

---

## 2. Benchmarks retenus

## 2.1. Principes de sélection

Les problèmes doivent être :
- standards dans la littérature scientific ML,
- assez simples pour un POC rapide,
- assez riches pour tester reconstruction, rollout, généralisation et acquisition,
- compatibles avec des baselines fortes de type neural operators.

## 2.2. Benchmarks retenus pour le projet

### Phase POC 1D

#### B1 — Burgers 1D paramétrique
Rôle :
- benchmark d’entrée,
- simple à simuler,
- utile pour calibrer AE, dynamique et acquisition,
- baseline FNO facile à mettre en place.

#### B2 — Kuramoto–Sivashinsky 1D
Rôle :
- benchmark chaotique,
- utile pour mesurer stabilité long rollout,
- utile pour valider l’intérêt du latent et de l’acquisition.

### Phase 2D standard

#### B3 — Diffusion-reaction 2D
Rôle :
- benchmark 2D de complexité intermédiaire,
- utile pour valider pipeline spatial 2D,
- moins coûteux que Navier–Stokes.

#### B4 — Incompressible Navier–Stokes 2D en vorticité
Rôle :
- benchmark principal du papier,
- comparable à FNO, F-FNO et benchmarks type PDEBench,
- test central pour rollouts, multi-échelles et conditionnement physique.

### Phase multi-PDE

#### B5 — Burgers 1D + KS 1D
Rôle :
- première validation du latent partagé.

#### B6 — Diffusion-reaction 2D + Navier–Stokes 2D
Rôle :
- validation multi-PDE 2D.

---

## 3. Solveurs de référence

## 3.1. Exigences générales

Chaque solveur doit :
- être encapsulé derrière une interface commune,
- être déterministe sous seed,
- exposer les paramètres de stabilité,
- logguer les warnings et les crashes,
- retourner des métadonnées normalisées.

### Interface logique commune

Entrée :
- état initial,
- contexte PDE,
- nombre de pas,
- `dt`,
- options solveur.

Sortie :
- trajectoire,
- états intermédiaires,
- temps de calcul,
- indicateurs de stabilité,
- statut de terminaison.

## 3.2. Solveur Burgers 1D

### Convention recommandée
- domaine périodique,
- schéma pseudo-spectral ou différences finies haute résolution,
- viscosité variable,
- IC aléatoires lisses,
- forcing optionnel selon variante.

### Paramètres à balayer
- viscosité,
- amplitude IC,
- bande fréquentielle initiale,
- forcing si activé.

### Critères de validation
- pas de NaN sur plage standard,
- stabilité sur horizon de référence,
- reproductibilité stricte.

## 3.3. Solveur KS 1D

### Convention recommandée
- domaine périodique,
- pseudo-spectral Fourier,
- ETDRK4 ou schéma stiff équivalent,
- IC aléatoires lisses à basse fréquence.

### Paramètres à balayer
- taille du domaine,
- amplitude IC,
- densité spectrale initiale.

### Critères de validation
- stabilité sur horizon long,
- bonne restitution des motifs spatio-temporels,
- temps de simulation prévisible.

## 3.4. Solveur diffusion-reaction 2D

### Convention recommandée
- grille cartésienne régulière,
- schéma explicite ou semi-implicite simple,
- paramètres de réaction balayer sur une plage limitée,
- BC périodiques ou homogènes selon choix de benchmark.

### Critères de validation
- stabilité CFL ou équivalent,
- motifs cohérents,
- performance suffisante pour génération distribuée.

## 3.5. Solveur Navier–Stokes 2D

### Convention recommandée
- formulation en vorticité,
- domaine périodique 2D,
- grille régulière,
- viscosité paramétrique,
- forcing conforme aux conventions benchmark choisies.

### Résolutions cibles
- 64×64 pour développement et benchmark principal rapide,
- 128×128 pour expérience de montée en résolution.

### Paramètres à balayer
- viscosité,
- forcing amplitude,
- forcing structure,
- statistiques des IC.

### Critères de validation
- stabilité numérique,
- temps par rollout mesuré,
- cohérence statistique entre runs,
- absence de dérive grossière sur quantités physiques pertinentes.

---

## 4. Génération de données

## 4.1. Deux régimes distincts

### Offline dataset
Utilisé pour :
- prétrain AE,
- initialiser dynamique,
- calibrer crash predictor,
- apprendre un premier sampler.

### Online dataset
Utilisé pour :
- acquisition active,
- adaptation continue,
- mesurer data efficiency,
- analyser coût solveur vs gain de performance.

## 4.2. Format d’un sample

Chaque sample doit contenir :
- `state`,
- `next_state`,
- `time_index`,
- `dt`,
- `pde_id`,
- `pde_params`,
- `bc_descriptor`,
- `forcing_descriptor`,
- `grid_descriptor`,
- `trajectory_id`,
- `sample_origin`,
- `solver_status`,
- `solver_runtime_sec`,
- `seed`.

## 4.3. Format disque recommandé

- Zarr pour gros volumes et accès parallèle,
- Parquet pour index et métadonnées,
- JSON/YAML versionné pour manifestes de dataset.

## 4.4. Politique de versioning

Chaque dataset doit avoir :
- `dataset_name`,
- `dataset_version`,
- `generator_git_hash`,
- `solver_version`,
- `parameter_space_signature`,
- `seed_policy`.

---

## 5. Splits d’évaluation

## 5.1. Split IID
Train/val/test classique.

## 5.2. Split OOD paramètres
Exemples :
- viscosités non vues,
- amplitudes de forcing plus fortes,
- IC plus énergétiques.

## 5.3. Split long horizon
Même distribution initiale, rollout beaucoup plus long que l’horizon d’entraînement.

## 5.4. Split rare regimes
Sous-régions rares de l’espace de paramètres.

## 5.5. Split multi-PDE
Pour l’étude latent partagé :
- train sur plusieurs PDEs,
- test sur sous-régimes ou combinaisons peu vues.

---

## 6. Baselines à implémenter

## 6.1. Baselines physiques directes

### BL1 — CNN autoregressive
Rôle : baseline simple.

### BL2 — U-Net autoregressive
Rôle : baseline convolutionnelle forte.

### BL3 — ResNet temporel en espace physique
Rôle : baseline plus profonde mais encore simple.

## 6.2. Neural operators

### BL4 — FNO
Doit être baseline de référence principale.

### BL5 — F-FNO
Doit être baseline forte secondaire, surtout sur Navier–Stokes.

### BL6 — DeepONet ou UNO-like
Optionnelle selon budget.

## 6.3. Reduced-order / latent baselines

### BL7 — POD + dynamique simple
### BL8 — AE + MLP latent dynamics
### BL9 — AE + conv latent dynamics
### BL10 — Koopman AE si budget suffisant

## 6.4. Baselines acquisition

### BA1 — random trajectories
### BA2 — random states
### BA3 — uncertainty only
### BA4 — diversity only
### BA5 — uncertainty + diversity greedy
### BA6 — full learned sampler

---

## 7. Conditions de comparaison équitable

Toutes les comparaisons doivent respecter :
- mêmes splits,
- mêmes horizons de rollout,
- mêmes métriques,
- budgets d’entraînement reportés,
- budgets solveur reportés,
- tuning raisonnable et comparable,
- seeds multiples.

### Seeds minimales
- 3 pour itérations rapides,
- 5 pour tableau final principal.

---

## 8. Métriques benchmark

## 8.1. Reconstruction
- RMSE,
- relative L2,
- MAE,
- erreur spectrale,
- erreur sur dérivées spatiales.

## 8.2. Dynamique
- one-step error,
- `k`-step rollout error,
- erreur à horizons fixes,
- horizon avant divergence.

## 8.3. Physique
- masse,
- énergie,
- enstrophie si pertinente,
- violation de contraintes physiques.

## 8.4. Acquisition
- performance vs appels solveur,
- performance vs temps mur,
- taux de crash,
- diversité batch,
- couverture latent.

## 8.5. Système
- temps d’entraînement,
- temps d’inférence,
- mémoire,
- coût par échantillon online.

---

## 9. Ordre de benchmark recommandé

### Étape A
- Burgers 1D,
- CNN/U-Net/FNO,
- AE + latent dynamics,
- offline only.

### Étape B
- KS 1D,
- étude long rollout,
- premières expériences acquisition.

### Étape C
- Diffusion-reaction 2D,
- AE latent grid,
- FNO/F-FNO/U-Net.

### Étape D
- Navier–Stokes 2D,
- benchmark principal,
- étude coût vs performance,
- offline vs online.

### Étape E
- multi-PDE shared latent,
- ablations invariance.

---

## 10. Critères de réussite scientifique

Le benchmark doit permettre de conclure clairement sur au moins un axe :
- meilleure data efficiency,
- meilleure stabilité long rollout,
- meilleure adaptation multi-PDE / multi-paramètre,
- meilleur compromis coût solveur / performance.

