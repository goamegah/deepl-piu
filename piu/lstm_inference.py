import os
import torch
import pandas as pd
from piu.models.rnn import LSTMWithTabular
from piu.data.lstm_dataset import MixedDataSequenceDataset
from definitions import SERIES_TEST_DATA_PATH, TEST_DATA_PATH

# 🔹 Fonction pour effectuer l'inférence sur toutes les données de test
def batch_inference(model, parquet_dir, csv_path, device):
    print("🚀 Début de l'inférence sur l'ensemble des données test...")

    # 🔹 Charger les IDs depuis le fichier test.csv
    test_data = pd.read_csv(csv_path, dtype={"id": str})
    test_ids = test_data["id"].tolist()

    predictions = []

    for inference_id in test_ids:
        try:
            X_seq, X_static = MixedDataSequenceDataset.prepare_sample_for_inference(parquet_dir, csv_path, inference_id)
            X_seq, X_static = X_seq.to(device).unsqueeze(0), X_static.to(device).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                output = model(X_seq, X_static)
                prediction = torch.argmax(output, dim=1).item()

            predictions.append((inference_id, prediction))

        except Exception as e:
            print(f"⚠️ Erreur avec l'ID {inference_id}: {e}")
            predictions.append((inference_id, -1))  # -1 pour signaler une erreur

    # 🔹 Créer et enregistrer le fichier de soumission
    submission_df = pd.DataFrame(predictions, columns=["id", "sii"])
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"✅ Fichier de soumission généré : {submission_path}")

# 🔹 Fonction principale
def main():
    # 🔹 Définir le device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 Vérifier que le modèle existe
    model_path = "model.pth"
    if not os.path.exists(model_path):
        raise ValueError(f"❌ Modèle introuvable à {model_path}, assurez-vous de l'avoir entraîné.")

    # 🔹 Charger un exemple pour déterminer les dimensions d'entrée
    test_data = pd.read_csv(TEST_DATA_PATH, dtype={"id": str})
    sample_id = test_data["id"].iloc[0]  # Prendre le premier ID
    sample_X_seq, sample_X_static = MixedDataSequenceDataset.prepare_sample_for_inference(SERIES_TEST_DATA_PATH, TEST_DATA_PATH, sample_id)

    input_dim_seq = sample_X_seq.shape[-1]
    input_dim_static = sample_X_static.shape[-1]
    output_dim = 3  # ⚠️ Remplace avec ton vrai nombre de classes

    # 🔹 Initialiser le modèle avec les dimensions correctes
    model = LSTMWithTabular(
        input_dim_seq=input_dim_seq,
        hidden_dim=64,
        num_layers=2,
        input_dim_static=input_dim_static,
        output_dim=output_dim
    ).to(device)

    # 🔹 Charger les poids du modèle entraîné
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 🔹 Lancer l'inférence sur tout le dataset test
    batch_inference(model, SERIES_TEST_DATA_PATH, TEST_DATA_PATH, device)

if __name__ == "__main__":
    main()
