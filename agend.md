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
- les fichiers de suivi persistants ont été créés pour éviter de repartir de zéro.

## Règles de conduite du projet

- ne pas sauter directement au multi-PDE ou à Ray sans avoir un pipeline 1D stable;
- respecter l'invariance PDE du latent dans le design principal;
- garder les datasets et les expériences versionnés;
- avancer par commits courts et réversibles.

## Reprise recommandée au prochain échange

1. vérifier l'état git et le contenu de `IMPLEMENTATION_TRACKER.md`;
2. poursuivre le sprint 1 si non terminé;
3. mettre à jour ce fichier et le tracker après chaque bloc livré;
4. ne démarrer le sprint suivant qu'après validation locale minimale.

