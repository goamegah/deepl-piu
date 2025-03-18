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

# --- Méthode avancée de sélection de features ---
def advanced_feature_selection(df, numeric_cols, target, method="none", n_features=10):
    """
    Applique une technique de sélection avancée sur les colonnes numériques du DataFrame.
    """
    if method == "none":
        return numeric_cols, df.copy()
    elif method == "pca":
        from sklearn.decomposition import PCA
        n_components = min(n_features, len(numeric_cols))
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(df[numeric_cols])
        df_transformed = df.copy()
        # Create new column names for the PCA components
        pca_columns = [f"PC{i+1}" for i in range(n_components)]
        df_transformed[pca_columns] = pca_result
        return pca_columns, df_transformed
    elif method == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(df[numeric_cols], df[target])
        selected = list(np.array(numeric_cols)[np.argsort(mi)[-n_features:]])
        return selected, df.copy()
    elif method == "lasso":
        from sklearn.linear_model import LassoCV
        X = df[numeric_cols]
        y = df[target]
        # Augmentez max_iter pour donner plus de temps à la convergence
        lasso = LassoCV(cv=5, max_iter=10000, tol=0.0001).fit(X, y)
        selected = list(np.array(numeric_cols)[lasso.coef_ != 0])
        if not selected:
            selected = list(np.array(numeric_cols)[np.argsort(np.abs(lasso.coef_))[-n_features:]])
        return selected, df.copy()

    elif method == "rf_importance":
        from sklearn.ensemble import RandomForestRegressor
        X = df[numeric_cols]
        y = df[target]
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X, y)
        importances = rf.feature_importances_
        idxs = np.argsort(importances)[-n_features:]
        selected = list(np.array(numeric_cols)[idxs])
        return selected, df.copy()
    else:
        raise ValueError(f"Méthode de sélection '{method}' inconnue.")

def get_common_static_columns(train_csv_path, test_csv_path, target="sii"):
    train_data = pd.read_csv(train_csv_path, dtype={"id": str})
    test_data = pd.read_csv(test_csv_path, dtype={"id": str})
    train_static_cols = set(train_data.columns) - {target, "id"}
    test_static_cols = set(test_data.columns) - {"id"}
    print(f"Colonnes statiques dans le CSV d'entraînement : {train_static_cols} ({len(train_static_cols)} colonnes)")
    print(f"Colonnes statiques dans le CSV de test : {test_static_cols} ({len(test_static_cols)} colonnes)")
    common = list(train_static_cols.intersection(test_static_cols))
    print(f"Colonnes statiques communes : {common} ({len(common)} colonnes)")
    return common

def collate_fn(batch):
    X_seq, X_static, y = zip(*batch)
    X_seq_padded = pad_sequence(X_seq, batch_first=True, padding_value=0)
    X_static = torch.stack(X_static)
    y = torch.stack(y) if y[0] is not None else None
    return X_seq_padded, X_static, y

def get_dataloaders(parquet_dir, csv_path, batch_size, split="train", test_size=0.2, feature_selection="none", common_static_cols=None):
    datasets = MixedDataSequenceDataset.return_splits(
        parquet_dir, csv_path, split=split, test_size=test_size, feature_selection=feature_selection, common_static_cols=common_static_cols
    )
    if split == "both":
        train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
        return train_loader, val_loader
    else:
        return DataLoader(datasets, batch_size=batch_size, shuffle=(split == "train"), num_workers=4, collate_fn=collate_fn)

MAX_SEQ_LENGTH = 1000
DOWNSAMPLE_FACTOR = 10

class MixedDataSequenceDataset(Dataset):
    def __init__(self, parquet_dir, csv_path, ids, feature_selection="none", common_static_cols=None, is_train=True):
        self.parquet_dir = parquet_dir
        self.csv_data = pd.read_csv(csv_path, dtype={"id": str}).set_index("id")
        if is_train:
            self.csv_data.dropna(subset=["sii"], inplace=True)
        available_ids = {f.split("=")[1] for f in os.listdir(parquet_dir) if os.path.isdir(os.path.join(parquet_dir, f))}
        self.ids = [id_ for id_ in ids if id_ in available_ids]
        if not self.ids:
            raise ValueError("Aucun ID valide trouvé après le filtrage. Vérifiez le dataset et le split.")
        if common_static_cols is not None:
            self.static_columns = [col for col in common_static_cols if col in self.csv_data.columns]
        else:
            self.static_columns = list(self.csv_data.columns)
            if is_train and "sii" in self.static_columns:
                self.static_columns.remove("sii")
        self.categorical_cols = self.csv_data[self.static_columns].select_dtypes(include=["object"]).columns.tolist()
        self.numeric_cols = self.csv_data[self.static_columns].select_dtypes(include=["number"]).columns.tolist()
        self.csv_data[self.numeric_cols] = self.csv_data[self.numeric_cols].fillna(self.csv_data[self.numeric_cols].mean())
        for col in self.categorical_cols:
            self.csv_data[col] = self.csv_data[col].fillna("Unknown")
        if feature_selection != "none" and is_train:
            selected, self.csv_data = advanced_feature_selection(self.csv_data, self.numeric_cols, target="sii", method=feature_selection, n_features=10)
            self.numeric_cols = selected
        self.fitted_numeric_cols = self.numeric_cols
        self.label_encoders = {col: LabelEncoder().fit(self.csv_data[col]) for col in self.categorical_cols}
        self.scaler = StandardScaler().fit(self.csv_data[self.fitted_numeric_cols])
        self.is_train = is_train

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
        X_cat = torch.tensor([self.label_encoders[col].transform([csv_row[col]])[0] 
                            for col in self.categorical_cols], dtype=torch.float32)

        # Construction d'un DataFrame avec les colonnes utilisées lors du fit du scaler
        X_num_df = pd.DataFrame([csv_row[self.fitted_numeric_cols]], columns=self.fitted_numeric_cols)
        if hasattr(self.scaler, "feature_names_in_"):
            X_num_df = X_num_df[self.scaler.feature_names_in_]
        X_num_scaled = self.scaler.transform(X_num_df)
        X_num_tensor = torch.tensor(X_num_scaled[0], dtype=torch.float32)
        X_static = torch.cat([X_num_tensor, X_cat])
        y = torch.tensor(csv_row["sii"], dtype=torch.long) if self.is_train else None
        return X_seq, X_static, y


    @classmethod
    def return_splits(cls, parquet_dir, csv_path, split="train", test_size=0.2, feature_selection="none", common_static_cols=None):
        csv_data = pd.read_csv(csv_path, dtype={"id": str}).set_index("id")
        if split in ["train", "both"]:
            csv_data.dropna(subset=["sii"], inplace=True)
            train_ids, val_ids = train_test_split(csv_data.index, test_size=test_size, stratify=csv_data["sii"])
        elif split == "val":
            train_ids, val_ids = train_test_split(csv_data.index, test_size=test_size, stratify=csv_data["sii"])
        else:
            raise ValueError("split doit être 'train', 'val' ou 'both' !")
        if split == "train":
            return cls(parquet_dir, csv_path, ids=train_ids, feature_selection=feature_selection, common_static_cols=common_static_cols, is_train=True)
        elif split == "val":
            return cls(parquet_dir, csv_path, ids=val_ids, feature_selection=feature_selection, common_static_cols=common_static_cols, is_train=True)
        elif split == "both":
            return (
                cls(parquet_dir, csv_path, ids=train_ids, feature_selection=feature_selection, common_static_cols=common_static_cols, is_train=True),
                cls(parquet_dir, csv_path, ids=val_ids, feature_selection=feature_selection, common_static_cols=common_static_cols, is_train=True),
            )

    @staticmethod
    def prepare_sample_for_inference(parquet_dir, csv_path, inference_id, common_static_cols=None):
        file_path = os.path.join(parquet_dir, f"id={inference_id}", "part-0.parquet")
        if not os.path.exists(file_path):
            raise ValueError(f"❌ Le fichier Parquet pour l'ID {inference_id} est introuvable !")
        df = pq.ParquetFile(file_path).read().to_pandas()
        df = df.ffill().bfill().fillna(0)
        df = df.iloc[::DOWNSAMPLE_FACTOR]
        df = df.iloc[-MAX_SEQ_LENGTH:] if len(df) > MAX_SEQ_LENGTH else df
        df = df.select_dtypes(include=["number"])
        X_seq = torch.tensor(df.values, dtype=torch.float32)
        
        csv_data = pd.read_csv(csv_path, dtype={"id": str}).set_index("id")
        if inference_id not in csv_data.index:
            raise ValueError(f"❌ L'ID {inference_id} n'existe pas dans {csv_path}")
        csv_row = csv_data.loc[inference_id]
        if common_static_cols is not None:
            static_cols = [col for col in common_static_cols if col in csv_data.columns]
        else:
            static_cols = list(csv_data.columns)
        cat_cols = csv_data[static_cols].select_dtypes(include=["object"]).columns.tolist()
        num_cols = csv_data[static_cols].select_dtypes(include=["number"]).columns.tolist()
        label_encoders = {col: LabelEncoder().fit(csv_data[col]) for col in cat_cols}
        X_cat = torch.tensor([label_encoders[col].transform([csv_row[col]])[0] for col in cat_cols], dtype=torch.float32)
        
        scaler = StandardScaler().fit(csv_data[num_cols])
        X_num_df = pd.DataFrame([csv_row[num_cols]], columns=num_cols)
        if hasattr(scaler, "feature_names_in_"):
            X_num_df = X_num_df[scaler.feature_names_in_]
        X_num = torch.tensor(scaler.transform(X_num_df)[0], dtype=torch.float32)
        
        X_static = torch.cat([X_num, X_cat])
        return X_seq, X_static

