# DesambigVerbes
Utilisation d'un réseau de neurones pour une tâche de désambiguisation des sens de verbes en contexte.


# Etapes

### RN
+ Lecture des données
+ Découpage train/test
+ Sélection traits
    + contexte linéaire (vecteur somme d'une fenêtre de contexte k=2)
    + contextes syntaxiques (surface/profond)
        + rep vectorielle : objet, sujet (basé sur les word embeddings)
        + C : sujet et objet canoniques (prendre les pos les plus précises, généralement PRO -> sujet humain etc...)
        + diathèse (tous les sens ne permettent pas d'avoir un passif)
        + sous-catégorisation (codé en dur, 1 dimension par sous-cat possible, valeur booléenne)
+ Mise en place du réseau
    + vecteurs pré-entraînés
        + modifiés à l'apprentissage
        + non-modifiés à l'apprentissage
    + vecteurs initialisés et modifiés au cours de l'apprentissage
+ Entraînement du classifieur sur les données train
+ Prédiction sur les données tests
+ Evaluation du classifieur

### MFS
+ Lecture des données
+ Découpage train/test
+ "most frequent sense" de chaque verbe à partir des données train
+ Prédiction sur les données test
+ Evaluation

⇒ Comparaison RN / MFS
