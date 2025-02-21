import os
import wandb
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 🔹 Initialisation de wandb
wandb.init(project="Hybrid_LSTM_Tabulardata", config={
    "batch_size": 8,
    "learning_rate": 0.001,
    "hidden_dim": 64,
    "num_layers": 2,
    "num_epochs": 10
})


class MixedDataSequenceDataset(Dataset):
    def __init__(self, parquet_dir, csv_path):
        self.parquet_dir = parquet_dir
        self.csv_data = pd.read_csv(csv_path, dtype={"id": str})
        self.csv_data.set_index("id", inplace=True)
        self.csv_data.dropna(subset=["sii"], inplace=True)

        available_ids = [f.split("=")[1] for f in os.listdir(parquet_dir) if os.path.isdir(os.path.join(parquet_dir, f))]
        self.ids = [id_ for id_ in available_ids if id_ in self.csv_data.index]

        self.categorical_cols = self.csv_data.select_dtypes(include=["object"]).columns.tolist()
        self.numeric_cols = self.csv_data.select_dtypes(include=["number"]).columns.tolist()
        if "sii" in self.numeric_cols:
            self.numeric_cols.remove("sii")

        self.label_encoders = {col: LabelEncoder().fit(self.csv_data[col].fillna("Unknown")) for col in self.categorical_cols}
        self.scaler = StandardScaler().fit(self.csv_data[self.numeric_cols].fillna(self.csv_data[self.numeric_cols].mean()) + 1e-8)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_ = self.ids[index]
        file_path = os.path.join(self.parquet_dir, f"id={id_}", "part-0.parquet")

        pq_file = pq.ParquetFile(file_path)
        df = pq_file.read().to_pandas()
        df = df.ffill()
        df = df.iloc[::DOWNSAMPLE_FACTOR]
        if len(df) > MAX_SEQ_LENGTH:
            df = df.iloc[-MAX_SEQ_LENGTH:]
        df = df.select_dtypes(include=["number"])

        X_seq = torch.tensor(df.values, dtype=torch.float32)
        X_seq = torch.nan_to_num(X_seq, nan=0.0)

        csv_row = self.csv_data.loc[id_].copy()
        categorical_features = [self.label_encoders[col].transform([csv_row[col] if pd.notna(csv_row[col]) else "Unknown"])[0] for col in self.categorical_cols]
        X_cat = torch.tensor(categorical_features, dtype=torch.float32)

        X_num = torch.tensor(
            self.scaler.transform(pd.DataFrame([csv_row[self.numeric_cols].values], columns=self.numeric_cols))[0], 
            dtype=torch.float32
        )
        X_num = torch.nan_to_num(X_num, nan=0.0)

        X_static = torch.cat([X_num, X_cat])
        y = torch.tensor(csv_row["sii"], dtype=torch.long)

        return X_seq, X_static, y

def collate_fn(batch):
    sequences, static_features, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    static_features = torch.stack(static_features)
    labels = torch.stack(labels)
    return padded_sequences, static_features, labels

class LSTMWithTabular(nn.Module):
    def __init__(self, input_dim_seq, hidden_dim, num_layers, input_dim_static, output_dim):
        super(LSTMWithTabular, self).__init__()

        self.lstm = nn.LSTM(input_dim_seq, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        self.fc_static = nn.Sequential(
            nn.Linear(input_dim_static, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc_final = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, X_seq, X_static):
        lstm_out, _ = self.lstm(X_seq)
        lstm_out = lstm_out[:, -1, :]
        
        static_out = self.fc_static(X_static)

        combined = torch.cat((lstm_out, static_out), dim=1)
        
        output = self.fc_final(combined)
        return output

# 📂 Définir les chemins
train_parquet_dir = "/home/goamegah/Documents/workspace/develop/esgi/s1/deep-learning/deepl-piu/dataset/series_train.parquet"
train_csv_path = "/home/goamegah/Documents/workspace/develop/esgi/s1/deep-learning/deepl-piu/dataset/train.csv"

dataset = MixedDataSequenceDataset(train_parquet_dir, train_csv_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)

X_seq, X_static, y = next(iter(train_loader))

input_dim_seq = X_seq.shape[2]
input_dim_static = X_static.shape[1]
hidden_dim = 64
num_layers = 2
output_dim = len(dataset.csv_data["sii"].unique())

model = LSTMWithTabular(input_dim_seq, hidden_dim, num_layers, input_dim_static, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Training on: {device}")
model.to(device)

num_epochs = 10
clip_value = 5

for epoch in range(num_epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0

    for X_seq, X_static, y in train_loader:
        X_seq, X_static, y = X_seq.to(device), X_static.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X_seq, X_static)
        outputs = torch.nan_to_num(outputs, nan=0.0)
        
        loss = criterion(outputs, y)
        if torch.isnan(loss):
            print("❌ NaN detected in loss, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    train_accuracy = 100 * correct / total
    wandb.log({"Train Loss": train_loss / len(train_loader), "Train Accuracy": train_accuracy})

    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_seq, X_static, y in val_loader:
            X_seq, X_static, y = X_seq.to(device), X_static.to(device), y.to(device)
            outputs = model(X_seq, X_static)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    val_accuracy = 100 * correct / total
    wandb.log({"Val Loss": val_loss / len(val_loader), "Val Accuracy": val_accuracy})

print("✅ Entraînement terminé ! 🎉")
wandb.finish()
