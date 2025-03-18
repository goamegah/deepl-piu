import joblib
import argparse
import torch
import torch.nn as nn
from piu.models.hwn import HighwayNet
from piu.utils.train_utils import train_model, evaluate_model
from piu.data.tab_dataset import get_tab_dataloader
from piu.utils.optim_utils import get_scheduler, get_optimizer
from piu.definitions import *
import wandb

def main(args):
    wandb.init(
        project="Problematic Internet Use", 
        name=f"mod={args.model}-act={args.act}-opt={args.optim}-lr={args.lr}-sch={args.scheduler}-fts={args.fts}-k={args.k}-imb={args.imb}",
        entity=args.wandb_entity,
        config=vars(args)
    )

    train_loader, test_loader, class_weights, preprocessor = get_tab_dataloader(args)
    input_size = train_loader.dataset.tensors[0].shape[1]
    num_classes = len(torch.unique(train_loader.dataset.tensors[1]))

    model = HighwayNet(
        input_size=input_size,
        hidden_size=8,
        num_classes=num_classes,
        num_layers=1,
        dropout_rate=0.5
    )

    class_weights_tensor = class_weights.to(torch.float32) if class_weights is not None else None

    if class_weights_tensor is not None:
        class_weights = class_weights_tensor.to(torch.float32) 
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(model, args.optim, args.lr)
    scheduler = get_scheduler(optimizer, args.scheduler)

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
        use_balanced_accuracy=True if args.imb is not None else False
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
        '--fts', type=str, default='lasso', 
        choices=['k_best', 'pca', 'f_classif', 'chi2', 'f_classif', 'logistic_regression', 
                 'lasso', 'variance_threshold', 'correlation_threshold', None],
        help="Feature selection method"
    )
    parser.add_argument('--k', type=int, default=40, help="Number of best features to select")
    parser.add_argument('--imp', type=str, default='mean', choices=['median', 'mean', 'knn'], 
                        help="Method for handling missing values")
    parser.add_argument('--train_split', type=float, default=0.8, help="Ratio of training data")
    parser.add_argument('--model', type=str, default='hwn', choices=['hwn'], help="Type of model to train")
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd', 'radam', 'rmsprop'], 
                        help="Type of optimizer to use")
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['plateau', 'step', 'cosine', None],
                        help="Type of learning rate scheduler")
    parser.add_argument('--act', type=str, default='relu', choices=['relu', 'tanh', 'sigmoid', 'leaky_relu'], 
                        help="Activation function for hidden layers")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and testing")
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for optimization") 
    parser.add_argument('--wandb_entity', type=str, required=True, help="Your WandB entity")
    parser.add_argument('--patience', type=int, default=15, help="Number of epochs to wait for early stopping")
    parser.add_argument('--imb', type=str, default="class_weight", 
                        choices=['class_weight', 'smote', 'random_over', 'random_under', None], 
                        help="Strategy to handle class imbalance")

    args = parser.parse_args()
    main(args)
