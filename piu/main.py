import joblib
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from piu.data.data_preprocessor import DataPreprocessor
from piu.models.nn import MultiClassNN
from piu.utils.train import train_model, evaluate_model
from piu.definitions import *

def main(args):
    wandb.init(
        project="Problematic Internet Use", 
        name=f"mlp-lr-{args.learning_rate}-fs-{args.feature_selection_method}-k-{args.k_best}-balance-{args.balance_strategy}",
        entity=args.wandb_entity,
        config=vars(args)  # 🔥 Stocker directement tous les arguments dans wandb
    )
    
    # 🔥 Charger les données
    train_df = pd.read_csv(f'{DATASET_PATH}/train.csv')
    test_df = pd.read_csv(f'{DATASET_PATH}/test.csv')

    # ✅ Vérifier les colonnes communes entre train et test
    common_columns = list(set(train_df.columns) & set(test_df.columns))
    if args.target_column in train_df.columns:
        common_columns.append(args.target_column)  # ✅ S'assurer que la colonne cible est présente dans train_df

    print(f"✅ Colonnes communes utilisées : {common_columns}")

    # 🔥 Garde uniquement les colonnes communes + la cible
    train_df = train_df[common_columns].drop(columns=['id'], errors='ignore')

    # ✅ Initialisation du préprocesseur
    preprocessor = DataPreprocessor(
        target_column=args.target_column,
        feature_selection_method=args.feature_selection_method,
        k_best=args.k_best,
        imputation_method=args.imputation_method,
        balance_strategy='smote',  # 🔥 Tu peux modifier pour tester d'autres stratégies
        drop_missing_target=True
    )
    
    X, y, class_weights = preprocessor.fit_transform(train_df)

    # 🔥 Vérification du nombre de features après transformation
    print(f"✅ Nombre de features après transformation : {X.shape[1]}")

    # ✅ Stratified Split pour conserver la répartition des classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - args.train_split), stratify=y, random_state=42
    )

    print(f"✅ Répartition des classes dans train : {np.bincount(y_train.numpy())}")
    print(f"✅ Répartition des classes dans test : {np.bincount(y_test.numpy())}")


    print(f"✅ Taille du train set: {len(y_train)}, Taille du test set: {len(y_test)}")

    # 🔥 Création des DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    
    # ✅ Initialisation du modèle
    input_size = X_train.shape[1]
    num_classes = len(torch.unique(y_train))
    model = MultiClassNN(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes)

    # ✅ Gérer le cas où `class_weights` est None
    class_weights_tensor = class_weights.to(torch.float32) if class_weights is not None else None

    # create folder to save model based on the experiment
    CHECKPOINT_DIR = f"{CHECKPOINT_PATH}/mlp-fs-{args.feature_selection_method}-balance-{args.balance_strategy}"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 🔥 Entraînement avec Early Stopping
    train_model(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        class_weights=class_weights_tensor,    
        num_epochs=args.num_epochs, 
        learning_rate=args.learning_rate, 
        patience=args.patience,
        checkpoint_path=CHECKPOINT_DIR
    )
    
    # ✅ Évaluation du modèle
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, nn.CrossEntropyLoss())

    print(f"\n📊 **Résultats sur le test set**:")
    print(f"🔹 Perte : {test_loss:.4f}")
    print(f"🔹 Accuracy : {test_accuracy:.4f}")
    print(f"🔹 Precision : {test_precision:.4f}")
    print(f"🔹 Recall : {test_recall:.4f}")
    print(f"🔹 F1-score : {test_f1:.4f}")

    # ✅ Sauvegarde du modèle
    # torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/model.pth")

    # ✅ Sauvegarde du préprocesseur
    joblib.dump(preprocessor, f"{CHECKPOINT_DIR}/preprocessor.pkl")

    # ✅ Sauvegarde des arguments d'entraînement pour la reproductibilité
    joblib.dump(args, f"{CHECKPOINT_DIR}/train_args.pkl")

    print(f"\n✅ Modèle et préprocesseur sauvegardés dans `{CHECKPOINT_DIR}`")

    # Fin du logging avec WandB
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training pipeline for multi-class classification")
    
    # ✅ Ajout des arguments principaux
    parser.add_argument('--target_column', type=str, default='sii', help="Name of the target column")
    parser.add_argument('--feature_selection_method', type=str, default='correlation_threshold', 
                        choices=['k_best', 'pca', 'f_classif', 'chi2', 'f_classif', 'logistic_regression', 
                                 'lasso', 'variance_threshold', 'correlation_threshold', None], 
                        help="Feature selection method")
    parser.add_argument('--k_best', type=int, default=20, help="Number of best features to select")
    parser.add_argument('--imputation_method', type=str, default='mean', choices=['median', 'mean', 'knn'], help="Method for handling missing values")
    parser.add_argument('--train_split', type=float, default=0.8, help="Ratio of training data")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training and testing")
    parser.add_argument('--hidden_size', type=int, default=32, help="Number of hidden units in the model")
    parser.add_argument('--num_epochs', type=int, default=300, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for optimization") 
    parser.add_argument('--wandb_entity', type=str, required=True, help="Your WandB entity")
    parser.add_argument('--patience', type=int, default=15, help="Number of epochs to wait for early stopping")
    parser.add_argument('--balance_strategy', type=str, default='smote', 
                        choices=['class_weight', 'smote', 'random_over', 'random_under', None], 
                        help="Strategy to handle class imbalance")

    args = parser.parse_args()
    main(args)
