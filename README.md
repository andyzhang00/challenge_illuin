# Challenge Technique - Détection d'intentions - ILLUIN Technology - Andy Zhang

Ce répository implémente différents modèles de détection d'intention dans un chatbot.

## Méthodes de machine learning classiques

Une première étude des solutions possibles est proposé dans le notebook `intention_detection.ipynb` du repertoire `src`.
Ces solutions utilisent en particulier une représentation des verbatim sous forme d'une matrice de features TF-IDF et des algorithmes de classfications usuelles tels que la regression logistique ou SVM.

## Méthodes de Deep Learning - CamemBERT

La seconde solution proposée, la plus performante est l'utilisation de réseaux de neurones de type BERT pour effectuer cette classification des verbatim.
Un modèle pré-entrainé de CamemBERT (variante française de BERT) a été récupéré puis fine-tuné sur nos données d'entraînements.

Note : Le modèle pré-entrainé étant trop lourd, il est téléchargeable sur le lien suivant :

| Modèle | Lien |
| :-- | :-: |
| CamembertForSequenceClassification | [**Lien**](https://filesender.renater.fr/?s=download&token=0d52833d-41a0-4f26-85ef-c3094d2dc42c) |

Il suffit alors de placer le fichier `pytorch_model.bin` dans le dossier `/models` avant de pouvoir éxecuter les scripts `evaluation.py' et 'single_inference.py`

## Utilisation

Afin d'effectuer une inférence simple sur un verbatim donné, il suffit de se positionner dans le répertoire `src` de lancer la commande suivante :

``` shell
python single_inference.py "<verbatim>"
```

Afin d'effectuer une prédiction et une évaluation sur un dataset de test au format CSV, il suffit de se positionner dans le répertoire `src` et de lancer la commande suivante :

``` shell
python evaluation.py <test_set.csv>
```

Les résultats des prédictions se situera dans le dossier `/results` au format CSV et les résultats de l'évaluation sera directement affiché dans le terminal.