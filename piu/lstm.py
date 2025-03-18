import os
import torch
import torch.nn as nn
import wandb
import argparse
from piu.models.rnn import LSTMWithTabular
from piu.data.lstm_dataset import MixedDataSequenceDataset, get_dataloaders, get_common_static_columns
from piu.utils.lstm_utils import train, evaluate
from definitions import *  # On suppose que TRAIN_DATA_PATH, SERIES_TRAIN_DATA_PATH et TEST_DATA_PATH sont définis
from piu.utils.optim import get_optimizer, get_scheduler


def main():
    args = get_args()
    wandb.init(
        project="Hybrid_LSTM_Tabulardata",
        config=vars(args),
        name=f"Hybrid_LSTM_{args.mode}_"
             f"fts={args.feature_selection}_"
             f"imb={args.imbalance_handling}_"
             f"da={args.data_augmentation}_"
             f"opt={args.optimizer}_"
             f"sch={args.scheduler}_"
             f"metric={args.metric}".replace(" ", "")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chemins définis dans definitions.py
    train_parquet_dir = SERIES_TRAIN_DATA_PATH
    train_csv_path = TRAIN_DATA_PATH
    test_csv_path = TEST_DATA_PATH

    # Calcul des colonnes statiques communes entre le CSV d'entraînement (avec "sii") et le CSV de test
    common_static_cols = get_common_static_columns(train_csv_path, test_csv_path, target="sii")

    if args.mode in ["train", "test"]:
        # Création des DataLoaders en passant la liste des colonnes communes et la méthode de feature selection
        train_loader, val_loader = get_dataloaders(
            train_parquet_dir,
            train_csv_path,
            batch_size=args.batch_size,
            split="both",
            feature_selection=args.feature_selection,
            common_static_cols=common_static_cols
        )
        # Détermination dynamique des dimensions d'entrée
        sample_X_seq, sample_X_static, _ = next(iter(train_loader))
        input_dim_seq = sample_X_seq.shape[-1]
        input_dim_static = sample_X_static.shape[-1]
        output_dim = len(torch.unique(torch.tensor([train_loader.dataset[i][2] for i in range(len(train_loader.dataset))])))

        # Initialisation du modèle
        model = LSTMWithTabular(
            input_dim_seq=input_dim_seq,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            input_dim_static=input_dim_static,
            output_dim=output_dim
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, args.optimizer, args.learning_rate)
        scheduler = get_scheduler(optimizer, args.scheduler)

        wandb.watch(model, log="all")

        if args.mode == "train":
            train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.num_epochs, metric=args.metric)
            torch.save(model.state_dict(), args.save_path)
            print(f"✅ Modèle sauvegardé sous {args.save_path}")

        elif args.mode == "test":
            if not os.path.exists(args.save_path):
                raise ValueError(f"❌ Modèle introuvable à {args.save_path}, assurez-vous de l'avoir entraîné.")
            model.load_state_dict(torch.load(args.save_path, map_location=device))
            evaluate(model, val_loader, criterion, device, metric=args.metric)

    elif args.mode == "inference":
        if args.inference_id is None:
            raise ValueError("❌ L'ID pour l'inférence doit être spécifié avec --inference_id")
        if not os.path.exists(args.save_path):
            raise ValueError(f"❌ Modèle introuvable à {args.save_path}, assurez-vous de l'avoir entraîné.")

        # Préparation d'un échantillon pour l'inférence à partir du CSV de test
        sample_X_seq, sample_X_static = MixedDataSequenceDataset.prepare_sample_for_inference(
            train_parquet_dir,
            test_csv_path,
            args.inference_id,
            common_static_cols=common_static_cols
        )
        input_dim_seq = sample_X_seq.shape[-1]
        input_dim_static = sample_X_static.shape[0] if len(sample_X_static.shape) == 1 else sample_X_static.shape[-1]
        # La dimension de sortie doit être identique à celle de l'entraînement (à stocker ou fixer manuellement)
        output_dim = 4  # À ajuster selon votre cas

        model = LSTMWithTabular(
            input_dim_seq=input_dim_seq,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            input_dim_static=input_dim_static,
            output_dim=output_dim
        ).to(device)

        model.load_state_dict(torch.load(args.save_path, map_location=device))
        # Ici, vous pouvez appeler votre fonction d'inférence, par exemple :
        # inference(model, train_parquet_dir, test_csv_path, args.inference_id, device, common_static_cols)


# 🔹 Définition du parser d'arguments
def get_args():
    parser = argparse.ArgumentParser(description="LSTM hybride avec données tabulaires et séquentielles.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd", "rmsprop"], default="adam")
    parser.add_argument("--scheduler", type=str, choices=["none", "step", "cosine"], default="none")
    # Extension du choix de la méthode de sélection des features
    parser.add_argument("--feature_selection", type=str, choices=["none", "pca", "mutual_info", "lasso", "rf_importance"], default="none",
                        help="Technique de sélection de features avancée")
    parser.add_argument("--imbalance_handling", type=str, choices=["none", "oversampling", "weighted_sampler"], default="none")
    parser.add_argument("--mode", type=str, choices=["train", "test", "inference"], default="train")
    parser.add_argument("--save_path", type=str, default="checkpoints/lstm_model.pth")
    parser.add_argument("--metric", type=str, choices=["accuracy", "balanced_accuracy"], default="balanced_accuracy",
                        help="Choix de la métrique d'évaluation")
    parser.add_argument("--data_augmentation", type=str, choices=["none", "jitter", "scaling", "smote"], default="none",
                        help="Choix de la technique d'augmentation des données")
    parser.add_argument("--inference_id", type=str, default=None, help="ID pour l'inférence")
    return parser.parse_args()


if __name__ == "__main__":
    main()
