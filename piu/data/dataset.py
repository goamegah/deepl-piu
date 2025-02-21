import pandas as pd
import torch
import pyarrow.parquet as pq
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class TabularSequenceDataset(Dataset):
    """ Dataset PyTorch qui fusionne les CSV et Parquet sans tout charger en mémoire. """

    def __init__(self, csv_path, parquet_path):
        """
        :param csv_path: Chemin du fichier CSV contenant les features statiques.
        :param parquet_path: Dossier contenant les fichiers Parquet partitionnés.
        """
        self.csv_path = csv_path
        self.parquet_path = parquet_path

        # ✅ Charger uniquement les features statiques du CSV en mémoire
        self.df_static = pd.read_csv(csv_path)
        self.df_static.set_index("id", inplace=True)  # Utilisation de l'ID comme clé

        # ✅ Stocker uniquement les IDs pour éviter de stocker les données Parquet en RAM
        self.ids = self.df_static.index.tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """ Récupère un ID, charge sa séquence Parquet en streaming, et associe avec ses features statiques. """
        id_value = self.ids[idx]
    
        # ✅ Lire la séquence Parquet en streaming pour éviter de la charger en RAM
        sequence = self._load_sequence(id_value)
    
        # ✅ Charger les features statiques associées
        static_features = self.df_static.loc[id_value]
    
        # 🔥 Conversion explicite en float32
        static_features = pd.to_numeric(static_features, errors='coerce').astype("float32")
    
        # ✅ Définir `target` si présent (classification)
        target = static_features.iloc[-1] if "sii" in self.df_static.columns else None
        if target is not None:
            static_features = static_features[:-1]  # Enlever `sii` des features
    
        return sequence, torch.tensor(static_features.values, dtype=torch.float32), target
    
    def _load_sequence(self, id_value):
        """ Charge dynamiquement la séquence associée à un ID depuis le Parquet en streaming. """
        # ✅ Lire uniquement les lignes Parquet qui concernent l'ID actuel
        df_seq = pq.read_table(self.parquet_path, filters=[("id", "=", id_value)]).to_pandas()

        if df_seq.empty:
            return torch.zeros((1, len(df_seq.columns) - 1), dtype=torch.float32)  # Cas où aucune séquence n'existe

        return torch.tensor(df_seq.drop(columns=["id"]).values, dtype=torch.float32)
    

def collate_fn(batch):
    """
    Regroupe les séquences de longueurs différentes avec du padding.
    """
    sequences, static_features, targets = zip(*batch)

    # Convertir en tenseurs PyTorch
    sequences = [torch.tensor(seq, dtype=torch.float32).clone().detach() for seq in sequences]
    static_features = torch.stack([torch.tensor(stat, dtype=torch.float32).clone().detach() for stat in static_features])


    # Appliquer le padding pour uniformiser la longueur des séquences
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # Convertir les targets en tenseur (classification)
    targets = torch.tensor([int(t) for t in targets], dtype=torch.long)

    return sequences_padded, static_features, targets

