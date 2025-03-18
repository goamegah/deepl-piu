import os
import gc
import argparse
import wandb
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import numpy as np

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
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--save_path", type=str, default="model.pth")
    return parser.parse_args()

# 🔹 Initialisation de wandb
def init_wandb(args):
    wandb.init(project="Hybrid_LSTM_Tabulardata", config=vars(args))

# 🔹 Hyperparamètres fixes
MAX_SEQ_LENGTH = 1000
DOWNSAMPLE_FACTOR = 10

class MixedDataSequenceDataset(Dataset):
    def __init__(self, parquet_dir, csv_path, split="train", feature_selection="none"):
        self.parquet_dir = parquet_dir
        self.csv_data = pd.read_csv(csv_path, dtype={"id": str}).set_index("id")
        self.csv_data.dropna(subset=["sii"], inplace=True)

        # 🔹 Stratified Split
        train_ids, val_ids = train_test_split(self.csv_data.index, test_size=0.2, stratify=self.csv_data["sii"])
        self.ids = train_ids if split == "train" else val_ids

        available_ids = [f.split("=")[1] for f in os.listdir(parquet_dir) if os.path.isdir(os.path.join(parquet_dir, f))]
        self.ids = [id_ for id_ in available_ids if id_ in self.csv_data.index]

        self.categorical_cols = self.csv_data.select_dtypes(include=["object"]).columns.tolist()
        self.numeric_cols = self.csv_data.select_dtypes(include=["number"]).columns.tolist()
        self.numeric_cols.remove("sii")

        self.csv_data[self.numeric_cols] = self.csv_data[self.numeric_cols].fillna(self.csv_data[self.numeric_cols].mean())
        for col in self.categorical_cols:
            self.csv_data[col] = self.csv_data[col].fillna("Unknown")

        self.label_encoders = {col: LabelEncoder().fit(self.csv_data[col]) for col in self.categorical_cols}
        self.scaler = StandardScaler().fit(self.csv_data[self.numeric_cols] + 1e-8)

        # 🔹 Feature Selection
        if feature_selection == "pca":
            pca = PCA(n_components=10)
            self.csv_data[self.numeric_cols] = pca.fit_transform(self.csv_data[self.numeric_cols])
        elif feature_selection == "mutual_info":
            mi = mutual_info_classif(self.csv_data[self.numeric_cols], self.csv_data["sii"])
            top_features = np.array(self.numeric_cols)[np.argsort(mi)[-10:]]
            self.numeric_cols = list(top_features)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_ = self.ids[index]
        file_path = os.path.join(self.parquet_dir, f"id={id_}", "part-0.parquet")
        df = pq.ParquetFile(file_path).read().to_pandas()
        df = df.ffill().bfill().fillna(0)
        df = df.iloc[::DOWNSAMPLE_FACTOR]
        df = df.iloc[-MAX_SEQ_LENGTH:] if len(df) > MAX_SEQ_LENGTH else df
        df = df.select_dtypes(include=["number"])
        
        X_seq = torch.tensor(df.values, dtype=torch.float32)
        X_seq = torch.nan_to_num(X_seq, nan=0.0)

        csv_row = self.csv_data.loc[id_].copy()
        X_cat = torch.tensor([self.label_encoders[col].transform([csv_row[col]])[0] for col in self.categorical_cols], dtype=torch.float32)
        X_num = torch.tensor(self.scaler.transform([csv_row[self.numeric_cols].values])[0], dtype=torch.float32)
        X_static = torch.cat([X_num, X_cat])
        y = torch.tensor(csv_row["sii"], dtype=torch.long)

        return X_seq, X_static, y

def get_optimizer(model, args):
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=args.learning_rate)

def get_scheduler(optimizer, args):
    if args.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    return None

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args):
    for epoch in range(args.num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for X_seq, X_static, y in train_loader:
            X_seq, X_static, y = X_seq.to(device), X_static.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X_seq, X_static)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        if scheduler:
            scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"🔄 Epoch [{epoch+1}/{args.num_epochs}] - Train Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")
        wandb.log({"Train Loss": train_loss / len(train_loader), "Val Accuracy": val_acc})

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_seq, X_static, y in val_loader:
            X_seq, X_static, y = X_seq.to(device), X_static.to(device), y.to(device)
            outputs = model(X_seq, X_static)
            val_loss += criterion(outputs, y).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return val_loss / len(val_loader), 100 * correct / total

# 📌 Fonction principale
def main():
    args = get_args()
    init_wandb(args)

    # 📂 Chargement des données
    train_parquet_dir = "/path/to/train/parquet"
    train_csv_path = "/path/to/train.csv"

    train_dataset = MixedDataSequenceDataset(train_parquet_dir, train_csv_path, split="train", feature_selection=args.feature_selection)
    val_dataset = MixedDataSequenceDataset(train_parquet_dir, train_csv_path, split="val", feature_selection=args.feature_selection)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 📌 Définition du modèle
    input_dim_seq = train_dataset[0][0].shape[1]
    input_dim_static = train_dataset[0][1].shape[0]
    output_dim = len(train_dataset.csv_data["sii"].unique())

    model = LSTMWithTabular(input_dim_seq, args.hidden_dim, args.num_layers, input_dim_static, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 📌 Optimiseur et Scheduler
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    criterion = nn.CrossEntropyLoss()

    if args.mode == "train":
        train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args)
        torch.save(model.state_dict(), args.save_path)
        print(f"✅ Modèle sauvegardé sous {args.save_path}")

    elif args.mode == "test":
        model.load_state_dict(torch.load(args.save_path, map_location=device))
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"🔍 Test Accuracy: {val_acc:.2f}%")

    wandb.finish()

# 📌 Exécution
if __name__ == "__main__":
    main()
