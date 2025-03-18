import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from piu.data.dataproc import DataPreprocessor
from piu.definitions import *

def get_tab_dataloader(args):
    train_df = pd.read_csv(f'{DATASET_PATH}/train.csv')
    test_df = pd.read_csv(f'{DATASET_PATH}/test.csv')

    # Vérifier les colonnes communes entre train et test
    common_columns = list(set(train_df.columns) & set(test_df.columns))
    if args.target_column in train_df.columns:
        common_columns.append(args.target_column)  # S'assurer que la colonne cible est présente dans train_df

    print(f" * Colonnes communes utilisées : {common_columns}")

    # Garde uniquement les colonnes communes + la cible
    train_df = train_df[common_columns].drop(columns=['id'], errors='ignore')

    preprocessor = DataPreprocessor(
        target_column=args.target_column,
        fts=args.fts,
        k_best=args.k,
        imp=args.imp,
        imb=args.imb,
        drop_missing_target=True,
        correlation_threshold=0.9,  # Pour éviter de tout supprimer
        target_corr_threshold=0.01
    )

    X, y, class_weights = preprocessor.fit_transform(train_df)

    print(f"\n * Nombre de features après transformation : {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - args.train_split), stratify=y, random_state=42
    )

    print(f" * Répartition des classes dans train : {np.bincount(y_train.numpy())}")
    print(f" * Répartition des classes dans test : {np.bincount(y_test.numpy())}")
    print(f" * Taille du train set: {len(y_train)}, Taille du test set: {len(y_test)} \n")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader, class_weights, preprocessor