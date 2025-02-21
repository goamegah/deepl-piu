import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
from piu.models.rnn import LSTMWithTabular
from piu.data.lstm_dataset import MixedDataSequenceDataset, get_dataloaders
from piu.utils.lstm_utils import train, evaluate
from definitions import SERIES_TRAIN_DATA_PATH, TRAIN_DATA_PATH


# 🔹 Optimiseur
def get_optimizer(model, args):
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Optimiseur non reconnu.")


# 🔹 Scheduler
def get_scheduler(optimizer, args):
    if args.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == "none":
        return None


# 🔹 Mode inférence
def inference(model, parquet_dir, csv_path, inference_id, device):
    print(f"🔍 Inférence pour l'ID {inference_id}")

    # Préparation des données
    X_seq, X_static, id_ = MixedDataSequenceDataset.prepare_sample_for_inference(parquet_dir, csv_path, inference_id)

    X_seq = X_seq.to(device).unsqueeze(0)
    X_static = X_static.to(device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(X_seq, X_static)
        prediction = torch.argmax(output, dim=1).item()

    print(f"🔮 Prédiction pour l'ID {id_}: {prediction}")
    return prediction


# 🔹 Fonction principale
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

    # Chargement des données
    train_parquet_dir = SERIES_TRAIN_DATA_PATH
    train_csv_path = TRAIN_DATA_PATH

    if args.mode in ["train", "test"]:
        train_loader, val_loader = get_dataloaders(
            train_parquet_dir, train_csv_path, batch_size=args.batch_size, split="both", feature_selection=args.feature_selection
        )

        # Déterminer les dimensions d'entrée dynamiquement
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
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        # Suivi WandB
        wandb.watch(model, log="all")

        # Entraînement
        if args.mode == "train":
            train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args)

            # Sauvegarde du modèle
            torch.save(model.state_dict(), args.save_path)
            print(f"✅ Modèle sauvegardé sous {args.save_path}")

        # Évaluation
        elif args.mode == "test":
            if not os.path.exists(args.save_path):
                raise ValueError(f"❌ Modèle introuvable à {args.save_path}, assurez-vous de l'avoir entraîné.")

            model.load_state_dict(torch.load(args.save_path, map_location=device))
            evaluate(model, val_loader, criterion, device)

    # Mode inférence
    elif args.mode == "inference":
        if args.inference_id is None:
            raise ValueError("❌ L'ID pour l'inférence doit être spécifié avec --inference_id")

        if not os.path.exists(args.save_path):
            raise ValueError(f"❌ Modèle introuvable à {args.save_path}, assurez-vous de l'avoir entraîné.")

        # 📌 Chargement du modèle
        model = LSTMWithTabular(
            input_dim_seq=input_dim_seq,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            input_dim_static=input_dim_static,
            output_dim=output_dim
        ).to(device)

        model.load_state_dict(torch.load(args.save_path, map_location=device))
        inference(model, train_parquet_dir, train_csv_path, args.inference_id, device)


# 🔹 Argument Parser
def get_args():
    parser = argparse.ArgumentParser(description="LSTM hybride avec données tabulaires et séquentielles.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd", "rmsprop"], default="adam")
    parser.add_argument("--scheduler", type=str, choices=["none", "step", "cosine"], default="none")
    parser.add_argument("--feature_selection", type=str, choices=["none", "pca", "mutual_info"], default="none")
    parser.add_argument("--imbalance_handling", type=str, choices=["none", "oversampling", "weighted_sampler"], default="none")
    parser.add_argument("--mode", type=str, choices=["train", "test", "inference"], default="train")
    parser.add_argument("--save_path", type=str, default="checkpoints/lstm_model.pth")
    parser.add_argument("--inference_id", type=str, default=None, help="ID pour inférence")
    parser.add_argument("--metric", type=str, choices=["accuracy", "balanced_accuracy"], default="accuracy",
                        help="Choix de la métrique d'évaluation")
    parser.add_argument("--data_augmentation", type=str, choices=["none", "jitter", "scaling", "smote"], default="none",
                    help="Choix de la technique d'augmentation des données")

    return parser.parse_args()


if __name__ == "__main__":
    main()