# Plan de recherche et d’implémentation — World Model latent multi‑PDE, publiable et reproductible

## 0. Positionnement exact de la contribution

Le projet ne doit pas être présenté comme un simple "autoencodeur + surrogate + active learning", car ce triplet existe déjà sous plusieurs formes. La contribution doit être cadrée plus précisément.

### Thèse centrale

Construire un **world model pour PDEs** avec les propriétés suivantes :

1. **Espace latent d’état partagé** entre plusieurs familles de PDEs.
2. **Transition conditionnelle par la physique** : le passage `z_t -> z_{t+1}` est conditionné par une représentation explicite de l’opérateur PDE, de ses paramètres physiques, des conditions aux limites et des forçages.
3. **Active sampling online en espace latent** : au lieu de tirer des trajectoires complètes peu informatives, un agent sélectionne des états latents plausibles, diversifiés et informatifs à raffiner via solveur.
4. **Fidélité de reconstruction suffisamment élevée** pour que l’espace latent soit utilisable comme représentation calculable de l’état physique, et pas seulement comme représentation compressée qualitative.

### Angle publiable plausible

Le papier doit être vendu comme une combinaison de 4 idées, avec une rigueur de benchmark forte :
- **latent state space shared across PDE families**,
- **conditional latent dynamics driven by PDE descriptors**,
- **online active acquisition in latent space**,
- **solver-aware safety and diversity in acquisition**.

La publication ne sera crédible que si :
- les PDEs sont standards,
- les solveurs sont standards,
- les baselines sont fortes,
- les métriques incluent précision one-step, long rollout, coût compute, coût solveur, robustesse hors distribution, stabilité.

---

## 1. Choix des PDEs et simulations — version publiable

Pour être publiable, il faut éviter les PDEs "maison" et s’aligner sur des benchmarks reconnus.

## 1.1. Pourquoi PDEBench doit structurer le protocole

PDEBench est un benchmark très utilisé qui couvre des PDEs 1D et 2D standard, y compris Burgers, advection, diffusion-reaction, Darcy, Navier–Stokes et shallow-water. Il est explicitement conçu pour comparer neural operators, CNNs et autres approches sur des données de PDE homogènes.

### Conséquence méthodologique

Le projet doit séparer :
- **benchmarks standards de publication** : principalement issus de PDEBench ou alignés dessus,
- **benchmarks complémentaires orientés chaos/stiffness** : Kuramoto–Sivashinsky, très utile scientifiquement mais moins standardisé que Navier–Stokes et Burgers dans les comparaisons large communauté.

## 1.2. Jeux de problèmes recommandés

### Tier A — Benchmarks standards minimums pour publication

#### 1D
1. **Burgers 1D paramétrique**
2. **Advection 1D**
3. **Diffusion-reaction 1D**

Rôle :
- calibration du pipeline,
- étude reconstruction vs dynamique,
- coût faible,
- comparaison facile à baselines classiques.

#### 2D
4. **Incompressible Navier–Stokes 2D**
5. **Diffusion-reaction 2D**

Rôle :
- benchmark principal pour publication,
- domaine où FNO et variantes sont des baselines naturelles,
- test pertinent pour rollouts et dynamique conditionnelle.

### Tier B — Benchmarks différenciants scientifiquement

#### 1D
6. **Kuramoto–Sivashinsky 1D**

Rôle :
- chaos spatio-temporel,
- sensibilité aux erreurs de rollout,
- excellent terrain pour active sampling et stabilité long terme.

#### 2D
7. **Kuramoto–Sivashinsky 2D** si solveur robuste disponible

Rôle :
- test plus ambitieux,
- valeur scientifique forte,
- mais à n’utiliser qu’après verrouillage du pipeline standard.

## 1.3. Ordre de montée en charge recommandé

### Étape 1 — POC 1D standard
- Burgers 1D
- KS 1D

### Étape 2 — 2D standard
- Diffusion-reaction 2D
- Navier–Stokes 2D

### Étape 3 — Multi-PDE
- Burgers 1D + KS 1D pour valider le latent partagé en 1D
- puis Diffusion-reaction 2D + Navier–Stokes 2D
- puis éventuellement extension cross-dimensional si justifiée, mais pas dans le papier principal

### Recommandation importante

Pour la publication principale, je recommande de **ne pas commencer par “multi-dimension + multi-PDE”**. Le papier doit avoir un cœur expérimental lisible. Le meilleur compromis est :
- une histoire 1D claire pour l’analyse,
- un benchmark 2D standard fort (Navier–Stokes),
- puis une section multi-PDE conditionnelle bien contrôlée.

---

## 2. Solveurs de référence — ce qu’il faut standardiser

Le solveur doit être traité comme un composant scientifique de référence, pas comme un détail d’ingénierie.

## 2.1. Exigences

Chaque solveur de référence doit préciser :
- équation exacte,
- domaine spatial,
- discrétisation,
- schéma temporel,
- conditions aux limites,
- distribution des paramètres,
- contrôles de stabilité,
- format de sortie.

## 2.2. Recommandation solveurs par PDE

### Burgers 1D
- solveur pseudo-spectral ou différences finies haute résolution,
- conditions périodiques,
- contrôle CFL,
- génération paramétrique sur viscosité, forcing et IC.

### Kuramoto–Sivashinsky 1D
- pseudo-spectral Fourier + ETDRK4 ou schéma stiff adapté,
- domaine périodique,
- conditions initiales synthétiques standards,
- validation forte sur stabilité numérique.

### Diffusion-reaction 2D
- solveur simple et fiable,
- discrétisation régulière,
- utile pour validation 2D à coût modéré.

### Navier–Stokes 2D incompressible
Deux options :

#### Option A — alignement benchmark communautaire
Utiliser le protocole standard proche de celui de FNO / PDEBench : champ de vorticité sur grille régulière torique avec viscosité et éventuellement forcing.

#### Option B — solveur maison contrôlé mais fidèle au benchmark
Reproduire les conventions du benchmark de la littérature :
- domaine périodique 2D,
- sortie en vorticité,
- résolution 64×64 puis 128×128,
- mêmes distributions de forcing/viscosité que les références.

### Recommandation ferme

Pour être publiable rapidement, il faut **adopter les conventions données par les benchmarks existants** plutôt qu’inventer de nouvelles distributions de cas. L’originalité doit être dans le modèle et l’acquisition, pas dans le dataset.

---

## 3. Protocole datasets — offline, online, splits, formats

## 3.1. Jeux de données

Le projet doit distinguer strictement :

### Offline pretraining set
Grand dataset généré une fois pour :
- entraîner AE,
- entraîner dynamique initiale,
- calibrer crash predictor,
- initialiser sampler.

### Online acquisition set
Échantillons générés dynamiquement par le sampler :
- courts rollouts solveur,
- enrichissement ciblé,
- logs complets sur coût et utilité.

### Validation / test fixes
Toujours gelés dès le début :
- split in-distribution,
- split extrapolation paramètres,
- split long-horizon,
- split shifted ICs,
- split cross-PDE si applicable.

## 3.2. Splits obligatoires

### In-distribution
Train/val/test classiques.

### Parameter OOD
Exemples :
- viscosités absentes du train,
- forcing plus fort,
- amplitudes d’IC plus élevées.

### Rollout OOD
Même distribution initiale mais horizon bien plus long.

### PDE-family OOD
Pour la version multi-PDE :
- entraînement sur plusieurs familles,
- test sur combinaisons de paramètres rares,
- éventuellement leave-one-regime-out.

## 3.3. Format des échantillons

Chaque sample doit être un objet standard :
- `state`: champ physique,
- `next_state`: champ à `t+dt`,
- `time_step`,
- `pde_id`,
- `pde_params`,
- `bc_descriptor`,
- `forcing_descriptor`,
- `grid_descriptor`,
- `solver_metadata`,
- `trajectory_id`,
- `sample_origin` in `{offline, online}`.

### Format disque recommandé
- **Zarr** ou **HDF5** pour gros volumes,
- métadonnées JSON/YAML versionnées,
- index tabulaire parquet pour requêtes rapides.

---

## 4. Architecture du modèle — cadrage précis

## 4.1. Représentation d’état : encodeur/décodeur

Le point le plus critique est la fidélité de reconstruction. Si la reconstruction est trop faible, l’ensemble de la promesse “simuler dans l’espace latent” devient discutable.

### Choix principal recommandé

#### 1D
- encodeur CNN 1D résiduel,
- downsampling progressif,
- latent sous forme de petite carte 1D ou vecteur selon taille,
- décodeur symétrique avec skips.

#### 2D
- encodeur CNN 2D de type U-Net compact,
- latent sous forme de **latent grid spatiale** de petite résolution plutôt qu’un unique vecteur global,
- décodeur symétrique.

### Pourquoi une latent grid plutôt qu’un unique vecteur

Pour les PDEs 2D, surtout NS et KS, les structures locales et multi-échelles sont importantes. Une latent grid conserve mieux l’information spatiale qu’un pur vecteur comprimé.

### Invariance à la PDE

L’encodeur et le décodeur doivent être **non conditionnés par la PDE** dans le design principal. La PDE doit influencer uniquement la transition.

Formulation :
- `z = E(u)`
- `u_hat = D(z)`
- `z_next = F(z, a)`

### Contrôle de l’invariance

Ajouter un classifieur auxiliaire de `pde_id` sur `z` avec adversarial training léger :
- si `z` prédit trop facilement la PDE, le latent n’est pas assez partagé,
- mais cette pénalité doit être faible au début pour ne pas détruire la reconstruction.

### Pertes AE recommandées

#### Obligatoires
- L2 reconstruction,
- L1 reconstruction.

#### Fortement recommandées
- gradient loss,
- spectral loss en Fourier,
- conservation loss si quantité conservée pertinente.

### Métriques AE à reporter
- RMSE reconstruction,
- relative L2,
- erreur spectrale par bande de fréquence,
- erreur sur dérivées spatiales,
- erreurs sur quantités intégrales physiques.

## 4.2. Modèle de transition latent

La dynamique conditionnelle est le cœur scientifique.

### Entrées
- `z_t`,
- `a = (pde_id, paramètres, forcing, BC, dt, éventuellement résolution)`.

### Encodage de l’action physique `a`

Décomposer `a` en 4 blocs :
- `pde_family_embedding` : embedding discret,
- `physical_params_encoder` : MLP sur viscosité, coefficients, etc.,
- `bc_encoder` : représentation compacte des conditions aux limites,
- `forcing_encoder` : MLP/CNN selon forcing scalaire ou champ.

Puis concaténer tout cela dans un **physics context vector**.

### Architecture recommandée

#### Variante principale
- backbone convolutionnel sur latent grid,
- modulation par FiLM avec le physics context,
- résidu temporel : le réseau prédit `Δz` plutôt que `z_{t+1}`.

Forme :
- `Δz = F_theta(z_t, context)`
- `z_{t+1} = z_t + Δz`

#### Variante secondaire à évaluer
- hypernetwork léger générant certains paramètres de blocs de convolution.

#### Variante stabilisée long rollout
- ajout d’un terme de type Koopman/linear head dans le latent pour comparer stabilité vs expressivité.

### Apprentissage
- one-step loss,
- multi-step rollout loss,
- schedule de teacher forcing décroissant,
- loss en espace latent + loss après décodage en espace physique.

### Métriques dynamique
- error one-step en latent,
- error one-step décodée,
- error `k`-step rollout,
- horizon de stabilité avant divergence,
- drift des quantités physiques.

## 4.3. Générateur de latents candidats

Le sampler a besoin de candidats plausibles. Il ne faut jamais explorer un espace latent arbitraire sans garde-fou.

### Recommandation

Commencer par un **modèle de densité simple** sur `z` :
- GMM sur résumés de latent,
- ou normalizing flow léger,
- ou banque de latents + perturbations locales contrôlées.

### Stratégie de départ recommandée

Pour le POC, éviter la complexité diffusion/gros flow.

Approche pratique :
- mémoire de latents déjà observés,
- voisinage local par bruit contrôlé,
- rejet selon score de plausibilité,
- puis upgrade éventuel vers flow.

## 4.4. Crash predictor

Avant d’appeler le solveur, un classifieur prédit si le candidat risque de produire :
- NaN,
- violation forte des bornes,
- instabilité numérique,
- temps solveur anormalement long.

### Rôle
- réduire coût cluster,
- éviter de polluer le dataset,
- fournir un terme de reward au sampler.

---

## 5. Sampler RL / bandit — cadrage précis

Le sampler ne doit pas être introduit comme un RL lourd si ce n’est pas nécessaire. La version de base doit être simple, robuste et crédible.

## 5.1. Positionnement recommandé

### Version 1
**Contextual batch bandit**.

L’agent observe l’état du modèle courant et choisit un batch de candidats à envoyer au solveur.

### Version 2
Passage possible à un agent RL plus riche seulement si :
- la sélection séquentielle a un impact mesurable,
- on veut optimiser sous budget solveur évolutif,
- on exploite l’historique complet de campagne comme état.

## 5.2. Espace d’action du sampler

Trois formulations possibles :

### Option A — scorer des candidats
La plus simple et recommandée.
- on génère `N` candidats plausibles,
- la politique attribue un score,
- on sélectionne top-k sous contrainte de diversité.

### Option B — paramétrer une distribution de proposition
- la politique produit les paramètres d’une distribution sur `z`,
- plus élégant mais plus difficile à stabiliser.

### Option C — sélectionner source + perturbation
- choisir un latent mémoire puis une perturbation locale,
- très pratique pour POC.

### Recommandation ferme

Commencer par **Option A** avec batch re-ranking. C’est le meilleur ratio utilité/complexité.

## 5.3. Reward / acquisition objective

La fonction objectif doit combiner trois termes.

### Informativité
Proxy principale : **incertitude du surrogate**.

Mesures possibles :
- variance inter-modèles d’un ensemble,
- MC dropout,
- disagreement sur rollout court.

### Diversité
Le batch ne doit pas se concentrer sur une région latente minuscule.

Mesures possibles :
- k-center greedy,
- farthest point sampling,
- DPP approximation,
- diversité vis-à-vis du batch déjà acquis et vis-à-vis du dataset historique.

### Sûreté / coût solveur
Pénaliser :
- crash,
- solveur trop lent,
- état décodé non physique.

### Forme recommandée

Pour un batch `S` :
- somme des scores d’incertitude,
- plus terme de diversité de batch,
- moins pénalité crash/coût,
- éventuellement plus terme de nouveauté vis-à-vis du dataset acquis.

## 5.4. Baselines acquisition à comparer

Le sampler proposé doit impérativement être comparé à des stratégies simples fortes :

1. **Random in latent memory**
2. **Random in trajectory space**
3. **Uncertainty only**
4. **Diversity only**
5. **Uncertainty + diversity greedy**
6. **Full sampler policy**

Sinon il sera impossible de savoir si le RL apporte vraiment quelque chose.

### Métriques acquisition
- gain de performance par nombre d’appels solveur,
- gain de performance par heure GPU/CPU,
- taux de crash,
- redondance du batch,
- couverture du latent.

---

## 6. Baselines — ce qu’il faut absolument inclure

Le papier doit comparer à des baselines crédibles à plusieurs niveaux.

## 6.1. Baselines de prédiction physique directe

### Niveau 1 — très simples mais nécessaires
1. **CNN autoregressive en espace physique**
2. **U-Net autoregressive en espace physique**
3. **ResNet temporel en espace physique**

Rôle : montrer que le latent n’est pas juste une complication inutile.

## 6.2. Baselines neural operator

### Indispensables
1. **FNO** — baseline centrale, standard sur Burgers et Navier–Stokes.
2. **Factorized FNO / F-FNO** — baseline forte et plus scalable, explicitement évaluée sur Navier–Stokes avec paramètres physiques et forcing.

### Souhaitables selon budget
3. **UNO-like** ou équivalent convolutionnel fort
4. **DeepONet** si vous voulez une baseline "operator learning" plus classique
5. **PDEBench baselines** pour alignement benchmark large

### Option avancée si temps
6. **APEBench-style autoregressive emulator baselines** pour mieux cadrer stabilité et régimes d’entraînement autoregressifs.

## 6.3. Baselines de réduction d’ordre / latent dynamics

Il faut aussi se comparer à des systèmes "representation + dynamics" et pas seulement à des opérateurs directs.

### Baselines recommandées
1. **Autoencoder + MLP/RNN latent dynamics**
2. **Autoencoder + conv latent dynamics**
3. **Koopman autoencoder** si possible
4. **POD + latent dynamics**

### Pourquoi POD est important
Même si ce n’est pas deep learning moderne, POD est une baseline de réduction d’ordre historique et très pertinente scientifiquement.

## 6.4. Baselines pour active acquisition

1. random trajectories,
2. random states,
3. uncertainty sampling,
4. core-set diversity,
5. uncertainty + diversity,
6. votre agent.

## 6.5. Baselines à ne pas mettre si elles diluent le papier

Ne pas viser 15 baselines. Mieux vaut 6 à 8 baselines fortes et bien implémentées que 20 mal contrôlées.

### Set recommandé minimal pour le papier principal
- CNN autoregressive,
- U-Net autoregressive,
- FNO,
- F-FNO,
- POD + latent dynamics,
- AE + latent conv dynamics,
- votre modèle sans active sampling,
- votre modèle avec active sampling.

---

## 7. Expériences obligatoires pour prendre les décisions techniques

Cette section est la plus importante du plan. Chaque expérience doit servir une décision.

## 7.1. Bloc A — représentation

### A1. Taille du latent
Comparer plusieurs facteurs de compression.

But : trouver la frontière entre :
- reconstruction acceptable,
- dynamique apprenable,
- coût mémoire/computation.

### A2. Vecteur latent vs latent grid
But : déterminer si la structure spatiale est nécessaire.

### A3. Pertes de reconstruction
Comparer :
- L2 seule,
- L2 + L1,
- L2 + spectral,
- L2 + gradient + spectral.

Décision recherchée : quels termes sont indispensables pour préserver les hautes fréquences et les dérivées.

### A4. Invariance PDE
Comparer :
- sans contrainte,
- avec adversarial léger,
- avec facteur style/content si nécessaire.

Décision : peut-on vraiment avoir un latent partagé sans trop dégrader reconstruction.

## 7.2. Bloc B — dynamique

### B1. Conditionnement de la physique
Comparer :
- concat simple,
- FiLM,
- hypernetwork léger.

Décision : meilleur compromis précision/stabilité/coût.

### B2. Espace de prédiction
Comparer :
- prédire `z_next`,
- prédire `Δz`,
- prédire plusieurs pas.

### B3. Objectif d’entraînement
Comparer :
- one-step seulement,
- one-step + rollout,
- scheduled sampling.

### B4. Nombre de modèles dans l’ensemble
Comparer 1, 3, 5 pour l’incertitude.

Décision : utilité réelle de l’ensemble pour acquisition vs surcoût.

## 7.3. Bloc C — acquisition

### C1. Offline only vs online acquisition
Expérience majeure.

Question : à coût solveur égal, l’online acquisition améliore-t-il vraiment les performances ?

### C2. Reward components
Comparer :
- informativité seule,
- diversité seule,
- informativité + diversité,
- informativité + diversité + crash penalty.

### C3. RL/bandit vs heuristique gloutonne
Comparer la politique apprenante à un simple greedy acquisition.

Décision : le RL est-il justifié scientifiquement ou seulement décoratif ?

### C4. Sampling latent vs sampling trajectoires
Comparer votre stratégie à une stratégie plus standard qui choisit des segments de trajectoires.

Décision : vérifier que le passage au latent est réellement utile.

## 7.4. Bloc D — multi-PDE

### D1. Single-PDE -> multi-parametric
Avant multi-PDE, tester variation de paramètres dans une même PDE.

### D2. Two-family shared latent
Exemple : Burgers 1D + KS 1D.

### D3. Shared latent with and without invariance regularization
### D4. Transfer / adaptation tests
Exemple : prétrain multi-PDE puis fine-tune faible budget sur nouveau régime.

## 7.5. Bloc E — robustesse et publication-quality

### E1. Rollout long horizon
À horizon de plus en plus long.

### E2. Resolution transfer
Si possible : train 64, test 128 pour certaines baselines compatibles.

### E3. Parameter extrapolation
### E4. Stress tests hors distribution

---

## 8. Métriques à rapporter dans le papier

## 8.1. Précision
- RMSE,
- relative L2,
- MAE,
- erreur spectrale,
- erreur sur gradients,
- erreur sur quantités physiques.

## 8.2. Dynamique
- erreur one-step,
- erreur multi-step,
- erreur à horizons fixes,
- horizon avant divergence,
- drift énergétique / masse / enstrophie selon PDE.

## 8.3. Acquisition
- performance vs nombre d’appels solveur,
- performance vs heures de calcul,
- data efficiency,
- taux de crash,
- diversity score du batch.

## 8.4. Système
- temps de training,
- temps d’inférence,
- mémoire GPU,
- scaling Ray,
- coût solveur moyen par sample acquis.

---

## 9. Architecture logicielle — version simple, agile, publiable

Le dépôt doit être organisé autour des concepts scientifiques, pas des scripts ad hoc.

## 9.1. Structure de repo recommandée

```text
repo/
  README.md
  pyproject.toml
  configs/
    data/
    model/
    train/
    acquisition/
    cluster/
    experiment/
  src/
    pdewm/
      data/
      solvers/
      representations/
      dynamics/
      acquisition/
      baselines/
      training/
      evaluation/
      orchestration/
      utils/
  scripts/
    generate_offline_data.py
    train_autoencoder.py
    train_dynamics.py
    train_baseline.py
    run_online_acquisition.py
    evaluate_experiment.py
    launch_ray_slurm.py
  tests/
    unit/
    integration/
    regression/
    performance/
  slurm/
    train.sbatch
    ray_head.sbatch
    ray_workers.sbatch
    online_loop.sbatch
  notebooks/
    analysis_results.ipynb
  artifacts/
    schemas/
    example_configs/
```

## 9.2. Responsabilités des modules

### `data/`
- datasets offline/online,
- loaders,
- splitters,
- normalisation,
- versioning metadata.

### `solvers/`
- wrappers solveurs par PDE,
- génération trajectoires,
- validation de stabilité,
- interface unifiée `simulate(initial_state, pde_context, steps)`.

### `representations/`
- encodeurs,
- décodeurs,
- losses reconstruction,
- analyse latent.

### `dynamics/`
- encodeur de contexte physique,
- transition model,
- ensemble wrapper,
- rollout utilities.

### `acquisition/`
- candidate generators,
- uncertainty estimators,
- diversity selectors,
- crash predictor,
- bandit/policy,
- replay buffer acquisition.

### `baselines/`
- FNO,
- F-FNO,
- CNN/U-Net AR,
- POD,
- AE+latent conv.

### `training/`
- trainer AE,
- trainer dynamics,
- trainer baselines,
- trainer acquisition policy,
- callback W&B.

### `evaluation/`
- metrics,
- benchmark suites,
- rollout evaluators,
- tables publication,
- figures.

### `orchestration/`
- Ray actors,
- job coordination,
- checkpoint registry,
- dataset writer,
- cluster utilities.

## 9.3. Configuration

Utiliser Hydra ou une structure YAML stricte.

Chaque expérience doit être reconstruisible par :
- un hash git,
- un config complet immuable,
- un seed,
- une version dataset,
- une version solveur.

---

## 10. Plan Python précis du POC 1D puis montée 2D

## 10.1. POC 1D — KS 1D + Burgers 1D

### Sprint 1 — Génération et QA des données
Livrables :
- wrappers solveur 1D,
- génération train/val/test,
- validations sur stabilité,
- formats HDF5/Zarr,
- scripts de visualisation.

Tests :
- dimensions,
- absence de NaN,
- reproductibilité par seed,
- conservation/borne attendue,
- coût moyen par trajectoire.

### Sprint 2 — Autoencodeur
Livrables :
- AE 1D,
- losses L1/L2/gradient/spectral,
- dashboards W&B reconstruction.

Tests :
- surfit sur petit batch,
- convergence reconstruction,
- comparaison tailles latent,
- analyse FFT.

### Sprint 3 — Dynamique latent single-PDE
Livrables :
- transition model conditionné minimal,
- one-step puis rollout.

Tests :
- surfit d’un sous-ensemble,
- stabilité rollout court,
- sensibilité au teacher forcing.

### Sprint 4 — Baselines 1D
Livrables :
- CNN AR,
- U-Net AR,
- FNO 1D,
- POD+MLP.

Tests :
- protocole commun,
- métriques homogènes,
- coût d’entraînement comparable.

### Sprint 5 — Acquisition online
Livrables :
- candidate memory,
- uncertainty scoring,
- diversity selection,
- crash predictor,
- batch bandit v1.

Tests :
- décroissance des crashes,
- utilité vs random,
- budget solveur fixé.

### Sprint 6 — Multi-paramètre puis multi-PDE 1D
Livrables :
- conditioning complet `a`,
- tests d’invariance du latent,
- shared latent analysis.

---

## 10.2. Montée 2D — Diffusion-reaction puis Navier–Stokes

### Sprint 7 — Data pipeline 2D
Livrables :
- solveurs 2D,
- génération distribuée,
- caches compressés,
- streaming loaders.

### Sprint 8 — AE 2D latent grid
Tests :
- coût mémoire,
- reconstruction fine structures,
- comparaison compression.

### Sprint 9 — Dynamics 2D
Tests :
- stability horizon,
- rollouts à plusieurs horizons,
- comparaison FiLM vs concat.

### Sprint 10 — Baselines 2D standard
Livrables :
- FNO,
- F-FNO,
- U-Net AR,
- POD ou AE baseline.

### Sprint 11 — Online acquisition distribué
Livrables :
- Ray actors solveur,
- scheduler de batch acquisition,
- storage partagé,
- intégration W&B.

---

## 11. Orchestration Ray + Slurm — plan précis

## 11.1. Architecture distribuée recommandée

### Rôles Ray
- **Trainer actor** : entraîne modèle principal,
- **Solver worker actors** : exécutent les solveurs,
- **Acquisition actor** : propose et sélectionne les candidats,
- **Dataset writer actor** : sérialise les nouveaux samples,
- **Evaluator actor** : exécute benchmarks périodiques.

## 11.2. Boucle online distribuée

1. Le trainer publie le dernier checkpoint.
2. L’acquisition actor charge le checkpoint et score des candidats.
3. Les meilleurs candidats sont envoyés aux solver workers.
4. Les workers retournent nouveaux états, métadonnées et statuts.
5. Le dataset writer persiste les données.
6. Le trainer réentraîne ou fine-tune.
7. L’evaluator déclenche des benchmarks fixes.

## 11.3. Principes de robustesse cluster

- tous les jobs sont relançables,
- les solveur workers sont idempotents,
- toutes les sorties sont versionnées,
- pas d’état critique seulement en mémoire,
- timeouts explicites par solveur,
- files d’attente bornées.

## 11.4. Intégration W&B

W&B doit tracker :
- configs complètes,
- versions de dataset,
- hash git,
- temps de génération,
- coût solveur,
- courbes acquisition,
- tableaux finaux de benchmark.

### Structure W&B recommandée
- project = `pde-world-model`,
- groups par famille d’expériences,
- tags par PDE et baseline,
- artefacts pour datasets et checkpoints.

---

## 12. Tests d’implémentation

Le dépôt doit être testé comme un produit de recherche, pas juste comme un script.

## 12.1. Tests unitaires

### Représentation
- dimensions input/output,
- inversibilité approximative,
- stabilité backward,
- shape consistency multi-résolutions.

### Dynamique
- compatibilité `z, a -> z_next`,
- rollouts sur 2–3 pas,
- variance ensemble non nulle.

### Acquisition
- scoring déterministe sous seed,
- diversité croissante,
- filtrage crash predictor,
- top-k stable.

### Solveurs
- formes de sortie,
- reproductibilité,
- contrôles NaN,
- respect time step.

## 12.2. Tests d’intégration

- génération dataset -> entraînement AE -> entraînement dynamique -> évaluation,
- boucle online complète avec solveur mock,
- boucle online réelle sur petit budget,
- reprise depuis checkpoint.

## 12.3. Tests de régression scientifique

Conserver un petit benchmark gelé pour vérifier :
- reconstruction ne régresse pas,
- rollout horizon ne régresse pas,
- FNO baseline ne change pas involontairement,
- coût acquisition reste dans bornes.

---

## 13. Tests de performance

## 13.1. Performance locale
- throughput data loader,
- temps forward/backward AE,
- temps rollout surrogate,
- temps solveur moyen.

## 13.2. Performance cluster
- scaling nombre de solver workers,
- saturation I/O,
- coût synchronisation Ray,
- temps d’attente files de tâches.

## 13.3. KPI de production scientifique
- coût pour générer 1k samples online,
- coût pour atteindre un seuil d’erreur,
- coût pour battre FNO sur benchmark cible,
- coût marginal de l’acquisition RL vs heuristique.

---

## 14. Tableau de décision technique — ce qui doit être tranché tôt

## Décisions critiques phase 1
1. latent vector vs latent grid,
2. losses reconstruction minimales,
3. FiLM vs concat,
4. ensemble size,
5. heuristic acquisition vs bandit,
6. crash predictor utile ou non.

## Décisions critiques phase 2
7. shared latent réellement viable sur deux PDEs,
8. overhead online acquisition acceptable ou non,
9. Ray apporte-t-il un vrai gain vs simple multiprocessing + Slurm array,
10. baselines standard battues ou non sur budget réaliste.

---

## 15. Recommandation finale très concrète

### Design principal
- AE déterministe multi-échelle,
- latent grid partagé non conditionné par PDE,
- dynamique conditionnée par `a` via FiLM,
- prédiction résiduelle `Δz`,
- ensemble de 3 modèles,
- acquisition batch bandit avec score = incertitude + diversité − pénalité crash.

### Benchmarks principaux
- Burgers 1D,
- KS 1D,
- Navier–Stokes 2D benchmark standard.

### Baselines minimales sérieuses
- CNN AR,
- U-Net AR,
- FNO,
- F-FNO,
- POD + latent dynamics,
- AE + latent conv dynamics.

### Infra
- offline datasets figés,
- online acquisition via Ray actors sur Slurm,
- W&B artefacts + configs versionnés,
- tests unitaires + intégration + régression scientifique.

### Critère de succès du projet
Le système doit démontrer au moins un des trois gains suivants sur benchmark standard :
- meilleure data efficiency à coût solveur égal,
- meilleure stabilité long rollout à erreur comparable,
- meilleure adaptation multi-PDE / multi-paramètre qu’un baseline fort.

Sans l’un de ces trois gains, le projet restera intéressant techniquement mais difficile à vendre scientifiquement.

