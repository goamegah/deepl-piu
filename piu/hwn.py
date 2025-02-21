import joblib
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from piu.data.data_preprocessor import DataPreprocessor
from piu.models.hwn import HighwayNet
from piu.utils.train_utils import train_model, evaluate_model
from piu.definitions import *
import wandb

def main(args):
    wandb.init(
        project="Problematic Internet Use", 
        name=f"mod={args.model}-act={args.act}-opt={args.optim}-lr={args.lr}-fts={args.fts}-k={args.k}-imb={args.imb}",
        entity=args.wandb_entity,
        config=vars(args)
    )
    
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
        drop_missing_target=True
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
    
    input_size = X_train.shape[1]
    num_classes = len(torch.unique(y_train))

    model = HighwayNet(
        input_size=input_size,
        hidden_size=8,
        num_classes=num_classes,
        num_layers=3,
        dropout_rate=0.5
    )

    class_weights_tensor = class_weights.to(torch.float32) if class_weights is not None else None

    if class_weights_tensor is not None:
        class_weights = class_weights_tensor.to(torch.float32) 
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'radam':
        optimizer = optim.RAdam(model.parameters(), lr=args.lr)
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f" /!\ Erreur : Optimiseur {args.optim} non reconnu")
    
    scheduler = None
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0)

    # create folder to save model based on the experiment
    CHECKPOINT_DIR = f"{CHECKPOINT_PATH}/mod={args.model}-lr={args.lr}-fts={args.fts}-k={args.k}-imb={args.imb}"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_model(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,   
        num_epochs=args.epochs, 
        patience=args.patience,
        checkpoint_path=CHECKPOINT_DIR,
        imbalanced=True if args.imb is not None else False
    )
    
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(
        model, 
        test_loader, 
        nn.CrossEntropyLoss()
    )

    print(f"\n **Résultats sur le test set**:")
    print(f" * Perte : {test_loss:.4f}")
    print(f" * Accuracy : {test_accuracy:.4f}")
    print(f" * Precision : {test_precision:.4f}")
    print(f" * Recall : {test_recall:.4f}")
    print(f" * F1-score : {test_f1:.4f}")

    joblib.dump(preprocessor, f"{CHECKPOINT_DIR}/preprocessor.pkl")
    joblib.dump(args, f"{CHECKPOINT_DIR}/train_args.pkl")

    print(f"\n ... Modèle et préprocesseur sauvegardés dans `{CHECKPOINT_DIR}`")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training pipeline for multi-class classification")
    
    # Ajout des arguments principaux
    parser.add_argument('--target_column', type=str, default='sii', help="Name of the target column")
    parser.add_argument(
        '--fts', type=str, default='k_best', 
        choices=['k_best', 'pca', 'f_classif', 'chi2', 'f_classif', 'logistic_regression', 
                 'lasso', 'variance_threshold', 'correlation_threshold', None],
        help="Feature selection method"
    )
    parser.add_argument('--k', type=int, default=20, help="Number of best features to select")
    parser.add_argument('--imp', type=str, default='mean', choices=['median', 'mean', 'knn'], 
                        help="Method for handling missing values")
    parser.add_argument('--train_split', type=float, default=0.8, help="Ratio of training data")
    parser.add_argument('--model', type=str, default='hwn', choices=['hwn'], help="Type of model to train")
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd', 'radam', 'rmsprop'], 
                        help="Type of optimizer to use")
    parser.add_argument('--scheduler', type=str, default=None, choices=['plateau', 'step', 'cosine', None],
                        help="Type of learning rate scheduler")
    parser.add_argument('--act', type=str, default='relu', choices=['relu', 'tanh', 'sigmoid', 'leaky_relu'], 
                        help="Activation function for hidden layers")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training and testing")
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimization") 
    parser.add_argument('--wandb_entity', type=str, required=True, help="Your WandB entity")
    parser.add_argument('--patience', type=int, default=15, help="Number of epochs to wait for early stopping")
    parser.add_argument('--imb', type=str, default=None, 
                        choices=['class_weight', 'smote', 'random_over', 'random_under', None], 
                        help="Strategy to handle class imbalance")

    args = parser.parse_args()
    main(args)
