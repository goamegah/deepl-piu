## Installation

1. Clonez le dépôt :

```sh
git clone https://github.com/goamegah/deepl-piu
cd deepl-piu
```

2. Créez un environnement virtuel et installez les dépendances 

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation
### Dataset
Les datasets utilisés dans ce projet proviennent de Kaggle. Vous pouvez les télécharger en suivant le lien ci-dessous :

[Kaggle](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data)

### Prétraitement des Données
Le prétraitement des données est géré par la classe DataPreprocessor. Voici un exemple d'utilisation :

```python
from piu.data.data_preprocessor import DataPreprocessor

# Initialisation du préprocesseur
preprocessor = DataPreprocessor(target_column='sii', feature_selection_method='lasso')

# Chargement des données
import pandas as pd
train_df = pd.read_csv('dataset/train.csv')

# Prétraitement des données
X, y = preprocessor.fit_transform(train_df)
```

### Entraînement des Modèles
Les modèles sont définis dans le répertoire models. Vous pouvez entraîner un modèle en utilisant le script principal main.py.

```bash
python piu/main.py --wandb_entity <your wandb entity name>
```

### Soumission des Prédictions
Pour générer et soumettre des prédictions, utilisez le script submission.py.

```bash
python piu/submission.py
```

## Structure des Fichiers
- checkpoints : Contient les checkpoints des modèles entraînés.
- dataset : Contient les fichiers de données.
- piu : Contient le code source du projet.
    - `data/` : Contient les scripts de prétraitement des données.
    - `models/` : Contient les définitions des modèles.
    - `notebook/` : Contient les notebooks Jupyter pour l'exploration des données.
    - `utils/` : Contient les fonctions utilitaires.
    - `wandb/` : Contient les configurations pour Weights & Biases.
- requirements.txt : Liste des dépendances du projet.
- setup.py : Script d'installation du projet.