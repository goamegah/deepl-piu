import os
import pandas as pd
import torch
import pyarrow.parquet as pq
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

# 🔹 Hyperparamètres
MAX_SEQ_LENGTH = 1000
DOWNSAMPLE_FACTOR = 10

def collate_fn(batch):
    X_seq, X_static, y = zip(*batch)
    X_seq_padded = pad_sequence(X_seq, batch_first=True, padding_value=0)
    X_static = torch.stack(X_static)
    y = torch.stack(y)
    return X_seq_padded, X_static, y

# 🔹 Fonction pour créer le DataLoader
def get_dataloaders(parquet_dir, csv_path, batch_size, split="train", test_size=0.2, feature_selection="none"):
    """
    Retourne un DataLoader en fonction du split.
    
    :param parquet_dir: Chemin vers les fichiers Parquet
    :param csv_path: Chemin vers le fichier CSV
    :param batch_size: Taille des batchs
    :param split: "train", "val" ou "both"
    :param test_size: Taille de la validation (si applicable)
    :param feature_selection: Méthode de sélection des features
    :return: DataLoader ou tuple (train_loader, val_loader)
    """
    datasets = MixedDataSequenceDataset.return_splits(parquet_dir, csv_path, split, test_size, feature_selection)

    if split == "both":
        train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
        return train_loader, val_loader
    else:
        return DataLoader(datasets, batch_size=batch_size, shuffle=(split == "train"), num_workers=4, collate_fn=collate_fn)


class MixedDataSequenceDataset(Dataset):
    def __init__(self, parquet_dir, csv_path, ids, feature_selection="none"):
        """
        Dataset pour gérer les séquences temporelles et les données tabulaires.

        :param parquet_dir: Chemin du dossier contenant les fichiers Parquet
        :param csv_path: Chemin du fichier CSV contenant les données tabulaires
        :param ids: Liste des IDs à utiliser (train ou val)
        :param feature_selection: Méthode de sélection des features ("none", "pca", "mutual_info")
        """
        self.parquet_dir = parquet_dir
        self.csv_data = pd.read_csv(csv_path, dtype={"id": str}).set_index("id")
        self.csv_data.dropna(subset=["sii"], inplace=True)

        # 🔹 Filtrer uniquement les IDs valides présents dans les fichiers Parquet
        available_ids = {f.split("=")[1] for f in os.listdir(parquet_dir) if os.path.isdir(os.path.join(parquet_dir, f))}
        self.ids = [id_ for id_ in ids if id_ in available_ids]

        # Vérification : S'assurer qu'on a bien des IDs valides
        if not self.ids:
            raise ValueError("Aucun ID valide trouvé après le filtrage. Vérifiez le dataset et le split.")

        # 🔹 Détection des colonnes
        self.categorical_cols = self.csv_data.select_dtypes(include=["object"]).columns.tolist()
        self.numeric_cols = self.csv_data.select_dtypes(include=["number"]).columns.tolist()
        self.numeric_cols.remove("sii")

        # 🔹 Gestion des NaN
        self.csv_data[self.numeric_cols] = self.csv_data[self.numeric_cols].fillna(self.csv_data[self.numeric_cols].mean())
        for col in self.categorical_cols:
            self.csv_data[col] = self.csv_data[col].fillna("Unknown")

        # 🔹 Sélection des features
        if feature_selection == "pca":
            pca = PCA(n_components=min(10, len(self.numeric_cols)))
            self.csv_data[self.numeric_cols] = pca.fit_transform(self.csv_data[self.numeric_cols])
        elif feature_selection == "mutual_info":
            mi = mutual_info_classif(self.csv_data[self.numeric_cols], self.csv_data["sii"])
            top_features = np.array(self.numeric_cols)[np.argsort(mi)[-10:]]
            self.numeric_cols = list(top_features)

        # 🔹 Stocker les colonnes utilisées
        self.fitted_numeric_cols = self.numeric_cols

        # 🔹 Encodage et Normalisation
        self.label_encoders = {col: LabelEncoder().fit(self.csv_data[col]) for col in self.categorical_cols}
        self.scaler = StandardScaler().fit(self.csv_data[self.fitted_numeric_cols])

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

        X_num = pd.DataFrame([csv_row[self.fitted_numeric_cols]], columns=self.fitted_numeric_cols)
        X_num = torch.tensor(self.scaler.transform(X_num)[0], dtype=torch.float32)

        X_static = torch.cat([X_num, X_cat])
        y = torch.tensor(csv_row["sii"], dtype=torch.long)

        return X_seq, X_static, y

    @classmethod
    def return_splits(cls, parquet_dir, csv_path, split="train", test_size=0.2, feature_selection="none"):
        """
        Retourne train et val selon le split.

        :param parquet_dir: Chemin vers les fichiers Parquet
        :param csv_path: Chemin vers le fichier CSV
        :param split: "train", "val" ou "both"
        :param test_size: Fraction des données utilisées pour la validation
        :param feature_selection: "none", "pca" ou "mutual_info"
        :return: `train_dataset` si `split="train"`, `val_dataset` si `split="val"`, `(train_dataset, val_dataset)` si `split="both"`
        """
        csv_data = pd.read_csv(csv_path, dtype={"id": str}).set_index("id")
        csv_data.dropna(subset=["sii"], inplace=True)

        # 🔹 Split stratifié
        train_ids, val_ids = train_test_split(csv_data.index, test_size=test_size, stratify=csv_data["sii"])

        if split == "train":
            return cls(parquet_dir, csv_path, ids=train_ids, feature_selection=feature_selection)
        elif split == "val":
            return cls(parquet_dir, csv_path, ids=val_ids, feature_selection=feature_selection)
        elif split == "both":
            return (
                cls(parquet_dir, csv_path, ids=train_ids, feature_selection=feature_selection),
                cls(parquet_dir, csv_path, ids=val_ids, feature_selection=feature_selection),
            )
        else:
            raise ValueError("split doit être 'train', 'val' ou 'both' !")
        
    @staticmethod
    def prepare_sample_for_inference(parquet_dir, csv_path, inference_id):
        """
        Prépare un échantillon unique pour l'inférence.
        """
        file_path = os.path.join(parquet_dir, f"id={inference_id}", "part-0.parquet")
        
        if not os.path.exists(file_path):
            raise ValueError(f"❌ Le fichier Parquet pour l'ID {inference_id} est introuvable !")

        df = pq.ParquetFile(file_path).read().to_pandas()

        # 🔹 Remplissage des valeurs manquantes
        df = df.ffill().bfill().fillna(0)
        df = df.iloc[::DOWNSAMPLE_FACTOR]
        df = df.iloc[-MAX_SEQ_LENGTH:] if len(df) > MAX_SEQ_LENGTH else df
        df = df.select_dtypes(include=["number"])

        X_seq = torch.tensor(df.values, dtype=torch.float32)

        # 🔹 Chargement des données tabulaires
        csv_data = pd.read_csv(csv_path, dtype={"id": str}).set_index("id")

        if inference_id not in csv_data.index:
            raise ValueError(f"❌ L'ID {inference_id} n'existe pas dans {csv_path}")

        csv_row = csv_data.loc[inference_id]

        # 🔹 Encodage des variables catégoriques
        categorical_cols = csv_data.select_dtypes(include=["object"]).columns.tolist()
        label_encoders = {col: LabelEncoder().fit(csv_data[col]) for col in categorical_cols}
        X_cat = torch.tensor([label_encoders[col].transform([csv_row[col]])[0] for col in categorical_cols], dtype=torch.float32)

        # 🔹 Normalisation des variables numériques
        numeric_cols = csv_data.select_dtypes(include=["number"]).columns.tolist()
        if "sii" in numeric_cols:
            numeric_cols.remove("sii")

        scaler = StandardScaler().fit(csv_data[numeric_cols])
        X_num = pd.DataFrame([csv_row[numeric_cols]], columns=numeric_cols)
        X_num = torch.tensor(scaler.transform(X_num)[0], dtype=torch.float32)

        X_static = torch.cat([X_num, X_cat])

        return X_seq, X_static

