from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import pandas as pd
from data.data_preprocessor import DataPreprocessor
from definitions import *
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
train_df = pd.read_csv(f'{DATASET_PATH}/train.csv')
test_df = pd.read_csv(f'{DATASET_PATH}/test.csv')

# Vérifier les colonnes communes entre train et test
common_columns = list(set(train_df.columns) & set(test_df.columns))
if "sii" in train_df.columns:
    common_columns.append("sii")
print(f"\n*Colonnes communes utilisées : {common_columns}")

# Garde uniquement les colonnes communes + la cible
train_df = train_df[common_columns].drop(columns=['id'], errors='ignore')

preprocessor = DataPreprocessor(
    target_column="sii",
    fts="pca",
    k_best=20,
    imp="mean",
    imb="class_weight",
    drop_missing_target=True
)

X, y, class_weights = preprocessor.fit_transform(train_df)

print(f"\n*Nombre de features après transformation : {X.shape[1]}\n")

# Stratified Split pour conserver la répartition des classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(1 - 0.8), stratify=y, random_state=42
)
print(f"\n*Répartition des classes dans train : {np.bincount(y_train.numpy())}")
print(f"*Répartition des classes dans test : {np.bincount(y_test.numpy())}")
print(f"*Taille du train set: {len(y_train)}, Taille du test set: {len(y_test)}")

X_train = X_train.numpy()
X_test = X_test.numpy()
y_train = y_train.numpy()
y_test = y_test.numpy()

clf = TabNetClassifier()  #TabNetRegressor()
clf.fit(
  X_train, 
  y_train,
  eval_set=[(X_test, y_test)],
  weights=1
)

# plot losses
# plt.plot(clf.history['loss'])

# plot accuracy
# plt.plot(clf.history['train_accuracy'])
# plt.plot(clf.history['valid_accuracy'])

preds = clf.predict(X_test)
print(preds)

