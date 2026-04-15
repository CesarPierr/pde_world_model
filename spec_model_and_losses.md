# Spec 2 — Modèle, représentation, dynamique conditionnelle et fonctions de coût

## 1. Objectif de ce document

Ce document fixe la spécification du modèle principal.

Le système cible est un **world model latent conditionnel multi-PDE** :
- un encodeur produit un latent d’état partagé,
- un décodeur reconstruit l’état physique,
- un modèle de transition prédit l’évolution du latent en fonction du contexte physique,
- un ensemble de modules auxiliaires fournit incertitude, stabilité et contraintes d’invariance.

---

## 2. Variables principales

## 2.1. État physique
Noté `u_t`.

Exemples :
- champ scalaire 1D,
- champ scalaire 2D,
- vorticité 2D,
- éventuellement champ multi-canaux.

## 2.2. Latent d’état
Noté `z_t`.

Contrainte conceptuelle :
- `z_t` doit représenter l’état physique,
- `z_t` ne doit pas dépendre explicitement de l’identité de la PDE dans le design principal,
- la physique doit intervenir via la transition, pas via l’encodage.

## 2.3. Contexte physique
Noté `a_t` ou `context`.

Il regroupe :
- `pde_id`,
- paramètres physiques,
- conditions aux limites,
- forcing,
- `dt`,
- éventuellement information de grille/résolution.

---

## 3. Encodeur / décodeur

## 3.1. Choix architectural principal

### 1D
- encodeur CNN résiduel 1D,
- downsampling progressif,
- latent sous forme de petite carte 1D ou vecteur selon compression,
- décodeur symétrique avec blocs upsampling + skip connections.

### 2D
- encodeur CNN 2D de type U-Net compact,
- latent sous forme de **latent grid 2D**,
- décodeur symétrique avec skips.

## 3.2. Pourquoi une latent grid

Une latent grid est préférable à un unique vecteur global en 2D parce que :
- elle conserve l’organisation spatiale,
- elle facilite la dynamique locale,
- elle améliore généralement la reconstruction fine,
- elle supporte mieux les phénomènes multi-échelles.

## 3.3. Principe d’invariance à la PDE

Le design principal impose :
- `z = E(u)`
- `u_hat = D(z)`
- `z_next = F(z, context)`

Le couple encodeur/décodeur n’est **pas conditionné** par la PDE.

### Raison
On veut que le latent soit un espace d’état partagé, pas un espace spécialisé par équation.

## 3.4. Variante secondaire

Si la reconstruction chute trop fortement sur le multi-PDE, prévoir une variante :
- latent partagé principal,
- petit composant résiduel de style/domaine.

Mais cette variante ne doit être étudiée qu’en ablation secondaire.

---

## 4. Structure du latent

## 4.1. Forme recommandée

### 1D
- latent shape type `(Cz, Lz)`

### 2D
- latent shape type `(Cz, Hz, Wz)`

## 4.2. Choix des compressions à tester

### 1D
- compression x4,
- compression x8,
- compression x16.

### 2D
- compression spatiale x4 et x8,
- variation sur nombre de canaux latents.

## 4.3. Contraintes de design

Le latent doit être :
- assez petit pour réduire coût dynamique,
- assez riche pour préserver les petites structures,
- stable numériquement sur rollout.

---

## 5. Encodeur du contexte physique

## 5.1. Décomposition recommandée

Le contexte physique doit être encodé par sous-modules séparés.

### Bloc A — PDE family encoder
Entrée : `pde_id`
Sortie : embedding discret appris.

### Bloc B — Physical parameters encoder
Entrée : vecteur continu de paramètres physiques.
Sortie : embedding par MLP.

### Bloc C — Boundary condition encoder
Entrée : descripteur des BC.
Sortie : embedding BC.

### Bloc D — Forcing encoder
Entrée : scalaire, vecteur ou champ de forcing.
Sortie : embedding forcing.

### Bloc E — Time-step / resolution encoder
Entrée : `dt`, éventuellement résolution.
Sortie : embedding scalaire.

## 5.2. Fusion

Tous les embeddings sont concaténés pour produire un `physics_context_vector`.

---

## 6. Modèle de transition latent

## 6.1. Principe général

Le modèle de transition prédit :
- soit `z_{t+1}` directement,
- soit `Δz_t`.

### Recommandation principale
Prédire le résidu :
- `Δz_t = F_theta(z_t, context)`
- `z_{t+1} = z_t + Δz_t`

## 6.2. Architecture principale

### Backbone
- blocs convolutionnels sur latent grid,
- résiduels,
- normalisation légère si utile,
- profondeur modérée.

### Conditionnement
- modulation par FiLM à plusieurs niveaux du backbone.

### Pourquoi FiLM
- simple,
- robuste,
- facile à interpréter,
- moins lourd qu’un hypernetwork complet.

## 6.3. Variantes à évaluer

### Variante V1 — concat simple
Concaténer `context` au latent via broadcast.

### Variante V2 — FiLM
Baseline principale.

### Variante V3 — hypernetwork léger
Le contexte génère certains paramètres du backbone.

### Variante V4 — tête linéaire/Koopman ajoutée
Pour évaluer stabilité long horizon.

---

## 7. Ensemble pour incertitude

## 7.1. Rôle

L’ensemble sert à :
- mesurer l’incertitude pour l’acquisition,
- estimer désaccord modèle,
- fournir un signal plus robuste qu’un unique réseau.

## 7.2. Design recommandé

- 3 modèles indépendants au départ,
- mêmes architectures,
- seeds distinctes,
- éventuellement sous-échantillonnage bootstrap.

## 7.3. Sorties dérivées

À partir de l’ensemble, calculer :
- variance one-step,
- variance rollout court,
- disagreement décodé en espace physique.

---

## 8. Contraintes d’invariance du latent

## 8.1. Besoin

Sans contrainte, le latent risque de se spécialiser implicitement par PDE.

## 8.2. Mécanisme principal recommandé

Ajouter un petit classifieur auxiliaire `C_pde(z)` qui tente de prédire `pde_id`.

Le training principal pousse à rendre cette prédiction difficile via mécanisme adversarial léger.

## 8.3. Réglage

La pénalité d’invariance doit être :
- nulle ou très faible au début,
- croissante après stabilisation reconstruction.

## 8.4. Métriques associées

- accuracy de classification PDE depuis `z`,
- reconstruction,
- performance dynamique.

L’objectif n’est pas de rendre `z` totalement non-informatif, mais d’éviter qu’il capture un biais fort de domaine.

---

## 9. Fonctions de coût

## 9.1. Pertes reconstruction AE

### L_rec_l2
Erreur quadratique reconstruction.

### L_rec_l1
Erreur absolue reconstruction.

### L_grad
Erreur sur gradients spatiaux.

### L_spec
Erreur dans le domaine fréquentiel.

### L_phys_rec
Erreur sur quantités physiques intégrales si pertinentes.

## 9.2. Pertes dynamique

### L_latent_1step
Erreur entre latent prédit et latent cible.

### L_phys_1step
Erreur entre état décodé prédit et état cible.

### L_rollout_k
Erreur cumulative sur rollout `k` pas.

### L_consistency
Pénalise incohérence entre dynamique latente et reconstruction physique.

## 9.3. Pertes invariance

### L_pde_adv
Terme adversarial pour réduire l’information PDE dans `z`.

## 9.4. Pertes incertitude / calibration

Optionnelles selon sophistication.

### L_ens_diversity
Encourager diversité utile entre membres d’ensemble.

### L_unc_calib
Calibration de l’incertitude si métrique retenue.

## 9.5. Fonction totale recommandée

### Phase AE seule
`L_AE = w1*L_rec_l2 + w2*L_rec_l1 + w3*L_grad + w4*L_spec + w5*L_phys_rec`

### Phase dynamique
`L_dyn = a1*L_latent_1step + a2*L_phys_1step + a3*L_rollout_k + a4*L_consistency + a5*L_pde_adv`

---

## 10. Régimes d’entraînement

## 10.1. Phase 1 — prétrain AE
Objectif : reconstruction très précise.

## 10.2. Phase 2 — dynamique gelant AE
Objectif : stabiliser le surrogate sans dégrader la représentation.

## 10.3. Phase 3 — fine-tuning joint optionnel
Objectif : ajuster AE + dynamique ensemble si bénéfice mesuré.

### Avertissement
Le fine-tuning joint peut dégrader la reconstruction et casser la stabilité si mal piloté. Il doit être optionnel et testé proprement.

## 10.4. Teacher forcing et scheduled sampling

### Régime initial
Teacher forcing fort.

### Régime intermédiaire
Scheduled sampling progressif.

### Régime final
Rollout loss plus longue si stable.

---

## 11. Ablations obligatoires

## 11.1. Représentation
- latent vector vs latent grid,
- skips on/off,
- taille du latent,
- pertes reconstruction.

## 11.2. Dynamique
- concat vs FiLM vs hypernetwork,
- direct next latent vs delta latent,
- one-step only vs rollout training.

## 11.3. Invariance
- sans contrainte,
- adversarial léger,
- variante style/content si nécessaire.

## 11.4. Ensemble
- 1 vs 3 vs 5 modèles.

---

## 12. Sorties et logging

À chaque entraînement, logger :
- reconstruction sample plots,
- spectres,
- rollout examples,
- métriques par horizon,
- coût compute,
- accuracy classif PDE sur `z`,
- statistiques de variance ensemble.

---

## 13. Critères de décision

Le modèle principal est validé si :
- reconstruction nettement meilleure qu’un AE naïf,
- dynamique latente au moins compétitive avec baselines physiques directes,
- stabilité long horizon raisonnable,
- signal d’incertitude exploitable,
- latent partagé compatible avec au moins deux PDEs sans effondrement majeur de performance.

