# Challenge Technique - Détection d'intentions - ILLUIN Technology - Andy Zhang

Ce repository implémente différents modèles de détection d'intention dans un chatbot.

## Méthodes de machine learning classiques

Une première étude des solutions possibles est proposée dans le notebook `intention_detection.ipynb` du répertoire `src`.
Ces solutions utilisent en particulier une représentation des verbatim sous forme d'une matrice de features TF-IDF et des algorithmes de classifications usuelles telles que la régression logistique ou SVM.

## Méthodes de Deep Learning - CamemBERT

La seconde solution proposée dans le notebook `bert_intention_detection.ipynb` du répertoire `src`, est l'utilisation de réseaux de neurones du type BERT pour effectuer cette classification des verbatim et est la plus performante.
Un modèle pré entrainé de CamemBERT (variante française de BERT) a été récupéré puis fine tuné sur nos données d'entraînements.

Note : Le modèle pré entrainé étant trop lourd, il est téléchargeable sur le lien suivant :

| Modèle | Lien |
| :-- | :-: |
| CamembertForSequenceClassification | [**Lien**](https://filesender.renater.fr/?s=download&token=0d52833d-41a0-4f26-85ef-c3094d2dc42c) |

Il suffit alors de placer le fichier `pytorch_model.bin` dans le dossier `/models` avant de pouvoir exécuter les scripts `evaluation.py` et `single_inference.py`

## Installation

``` shell
pip install -r requirements.txt
```

## Testing

Afin d'effectuer une inférence simple sur un verbatim donné, il suffit de se positionner dans le répertoire `src` de lancer la commande suivante :

``` shell
python3 single_inference.py "<verbatim>"
```

Afin d'effectuer une prédiction et une évaluation sur un dataset de test au format CSV, il suffit de se positionner dans le répertoire `src` et de lancer la commande suivante :

``` shell
python3 evaluation.py <test_set.csv>
```

Les résultats des prédictions se situeront dans le dossier `/results` au format CSV et les résultats de l'évaluation seront directement affichés dans le terminal.