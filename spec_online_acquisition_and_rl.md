# Spec 3 — Acquisition online, sampler, RL/bandit et budget solveur

## 1. Objectif de ce document

Ce document spécifie la boucle d’acquisition online.

Le but n’est pas seulement de produire plus de données, mais de produire **les données les plus utiles** au surrogate courant sous contrainte de coût solveur.

L’acquisition doit être :
- scientifiquement justifiable,
- techniquement simple au départ,
- extensible vers une politique plus sophistiquée,
- mesurable en termes de gain de performance par coût solveur.

---

## 2. Principe général

La boucle online repose sur 5 modules :
- générateur de candidats latents plausibles,
- estimateur d’informativité,
- mesure de diversité,
- prédicteur de crash/coût solveur,
- politique de sélection batch.

Le flux est :
1. proposer des candidats,
2. scorer les candidats,
3. sélectionner un batch,
4. décoder et soumettre au solveur,
5. écrire les nouveaux samples,
6. réentraîner le modèle,
7. répéter.

---

## 3. Pourquoi commencer par un contextual batch bandit

Le problème ressemble davantage à un **batch decision problem** qu’à un RL séquentiel profond classique.

### Raisons
- à chaque itération, on choisit un batch,
- la récompense est observée après appel solveur,
- l’horizon de décision est court,
- le coût d’une politique instable est élevé.

### Conclusion
Le design principal doit être :
- **contextual batch bandit** au départ,
- puis éventuellement RL plus riche en ablation avancée.

---

## 4. Espace d’action de l’agent

## 4.1. Option principale retenue

### Action = scorer des candidats
L’agent ne génère pas directement `z`.

Il reçoit un pool de candidats plausibles et assigne un score à chacun.

### Pourquoi ce choix
- stable,
- facile à déboguer,
- sépare génération de candidats et politique,
- compatible avec top-k + diversité.

## 4.2. Variantes secondaires

### Action = choisir une source mémoire + perturbation
Pertinent en POC.

### Action = paramétrer une distribution sur `z`
Réservé à une phase avancée.

---

## 5. Génération de candidats plausibles

## 5.1. Contrainte majeure

Le sampler ne doit jamais explorer arbitrairement l’espace latent. Les candidats doivent rester proches du manifold des états plausibles.

## 5.2. Sources de candidats recommandées

### Source A — mémoire de latents observés
Banque de latents issus du dataset offline et online.

### Source B — perturbations locales
Bruit contrôlé autour de latents mémoire.

### Source C — interpolation locale
Interpolation entre latents compatibles proches.

### Source D — modèle de densité simple
GMM ou flow léger sur résumés latents.

## 5.3. Politique de départ recommandée

Pour le POC :
- mémoire + perturbations locales + rejet par plausibilité.

---

## 6. Informativité

## 6.1. Signal principal

Le signal principal doit être le **désaccord du surrogate**.

## 6.2. Mesures recommandées

### U1 — variance one-step en latent
À partir de l’ensemble.

### U2 — variance one-step décodée en espace physique
Plus coûteuse, plus interprétable.

### U3 — variance rollout court
Très pertinente si le coût reste acceptable.

### U4 — erreur proxy via résidu physique
Si un estimateur cheap est disponible.

## 6.3. Recommandation

Score informatif principal :
- combinaison de `U1` et `U3`.

---

## 7. Diversité

## 7.1. Pourquoi elle est indispensable

Sans diversité, l’acquisition s’effondre sur une petite région difficile et sur-échantillonne des cas redondants.

## 7.2. Mesures recommandées

### D1 — farthest point sampling
Simple et efficace.

### D2 — k-center greedy
Très bon compromis.

### D3 — DPP approximation
À réserver si besoin.

## 7.3. Diversité à deux niveaux

### Batch diversity
Diversité interne au batch courant.

### Dataset novelty
Distance au dataset déjà acquis.

## 7.4. Recommandation

Commencer avec :
- score unitaire d’informativité,
- puis sélection gloutonne k-center sur les meilleurs candidats.

---

## 8. Crash predictor et coût solveur

## 8.1. Pourquoi il faut un module dédié

Dans une boucle online cluster, les états invalides coûtent cher.

Il faut prédire :
- crash numérique,
- temps solveur anormalement long,
- état physiquement aberrant.

## 8.2. Inputs du crash predictor
- latent candidat,
- résumé décodé éventuel,
- contexte physique,
- métadonnées solveur historiques.

## 8.3. Outputs
- probabilité de crash,
- coût solveur prédit,
- risque de rejet.

## 8.4. Usage
Le crash predictor est utilisé :
- comme filtre dur,
- ou comme pénalité douce dans le score.

---

## 9. Fonction objectif d’acquisition

## 9.1. Niveau candidat

Pour un candidat `c` :
- informativité `Info(c)`
- risque `Risk(c)`
- nouveauté `Novel(c)`

Score brut :
`score(c) = alpha*Info(c) + beta*Novel(c) - gamma*Risk(c)`

## 9.2. Niveau batch

Pour un batch `S` :
`Score(S) = sum(score(c)) + lambda*BatchDiversity(S)`

## 9.3. Sélection pratique recommandée

1. générer grand pool,
2. filtrer risques élevés,
3. garder top `M` par score unitaire,
4. faire sélection gloutonne diversifiée jusqu’à taille `K`.

---

## 10. Politique apprise

## 10.1. Version 1 — heuristique forte
Pas de politique neuronale. Score = combinaison analytique.

## 10.2. Version 2 — contextual scorer appris
Un petit réseau apprend à pondérer les critères à partir de l’état du système.

Entrées possibles :
- stats du modèle courant,
- stats du dataset,
- distribution des scores candidats,
- budget solveur restant.

Sortie :
- score par candidat,
- ou poids dynamiques `(alpha, beta, gamma, lambda)`.

## 10.3. Version 3 — batch bandit policy gradient
Réservée à une phase avancée.

---

## 11. Budget solveur

## 11.1. Le budget doit être une variable de benchmark

Toute expérience online doit préciser :
- nombre maximal d’appels solveur,
- nombre maximal d’heures CPU/GPU,
- taille des batches,
- horizon de rollout solveur demandé.

## 11.2. Modes de budget

### Mode A — budget en nombre d’échantillons
### Mode B — budget en nombre de trajectoires
### Mode C — budget en temps mur

## 11.3. Recommandation

Le papier doit au moins rapporter :
- performance vs appels solveur,
- performance vs temps mur.

---

## 12. Stratégie de labels solveur

## 12.1. Labels minimums
Pour chaque candidat résolu :
- `next_state`,
- statut solveur,
- runtime,
- warnings,
- résumé physique.

## 12.2. Rollout solveur court ou long

### Recommandation principale
Demander d’abord des **rollouts courts**.

### Raison
- moins coûteux,
- suffisant pour informer le surrogate,
- plus stable au début.

### Variante avancée
Échantillonnage adaptatif de l’horizon solveur.

---

## 13. Replay buffer acquisition

## 13.1. Rôle

Stocker :
- candidats testés,
- scores prévus,
- résultats solveur,
- gains sur modèle,
- taux de crash.

## 13.2. Utilisation
- entraînement du crash predictor,
- entraînement de la politique,
- analyse offline des acquisitions.

---

## 14. Baselines acquisition obligatoires

Comparer systématiquement à :
- random states,
- random trajectories,
- uncertainty only,
- diversity only,
- uncertainty + diversity greedy,
- policy apprise.

Sans ces comparaisons, le RL ne sera pas scientifiquement justifié.

---

## 15. Métriques acquisition

## 15.1. Métriques principales
- gain de performance du surrogate,
- gain par appel solveur,
- gain par heure de calcul,
- taux de crash,
- diversité du batch,
- couverture du latent.

## 15.2. Métriques secondaires
- calibration du crash predictor,
- évolution de l’incertitude moyenne,
- part des acquisitions réellement utiles.

---

## 16. Expériences obligatoires

## 16.1. Offline vs online
Question : l’online apporte-t-il un gain réel à budget solveur égal ?

## 16.2. Ablation du score
Comparer :
- info seule,
- diversité seule,
- info + diversité,
- info + diversité + risque.

## 16.3. Heuristique vs policy
Comparer greedy fort vs scorer appris.

## 16.4. Sampling latent vs sampling trajectoires
Question centrale pour la contribution.

## 16.5. Effet du crash predictor
Comparer sans filtre, filtre doux, filtre dur.

---

## 17. Critères de réussite

La boucle acquisition est validée si elle montre au moins un des gains suivants :
- meilleure précision à coût solveur égal,
- même précision à coût solveur réduit,
- meilleure stabilité long rollout,
- meilleure couverture des régimes rares.

