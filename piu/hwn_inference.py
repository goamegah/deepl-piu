import torch
import pandas as pd
import numpy as np
import joblib
import argparse
from piu.models.hwn import HighwayNet 
from piu.definitions import * 


CHECKPOINT_DIR = f"{CHECKPOINT_PATH}/mlp-fs-correlation_threshold-balance-class_weight"
MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pth"
PREPROCESSOR_PATH = f"{CHECKPOINT_DIR}/preprocessor.pkl"
DATA_PATH = f"{TEST_DATA_PATH}" 


def load_model(model_path, input_size, hidden_size, num_classes, num_layers):
    """Charge un modèle PyTorch entraîné."""
  
    model = HighwayNet(input_size, hidden_size, num_classes, num_layers)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    return model

def align_columns(df, preprocessor):
    """Aligne les colonnes du test set avec celles utilisées pendant l'entraînement."""
    train_columns = preprocessor.train_features  # Sauvegardé à l'entraînement
    missing_cols = set(train_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(train_columns)

    # Ajouter les colonnes manquantes (avec valeurs nulles ou 0)
    for col in missing_cols:
        df[col] = 0 if col in preprocessor.cat_features else np.nan
    df = df.drop(columns=extra_cols, errors='ignore')
    df = df[train_columns]

    return df

def predict(model, X):
    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    return predictions.numpy(), probabilities.numpy()

def batch_predict(model, X, batch_size=64):
    predictions = []
    probabilities = []
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_predictions, batch_probabilities = predict(model, torch.tensor(batch_X, dtype=torch.float32))
        predictions.extend(batch_predictions)
        probabilities.extend(batch_probabilities)
    return predictions, probabilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference pipeline for multi-class classification")
    parser.add_argument('--hidden_size', type=int, default=32, help="Hidden layer size of the model")
    parser.add_argument('--num_classes', type=int, default=4, help="Number of output classes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for inference")

    args = parser.parse_args()

    # Charger les données de test
    df = pd.read_csv(f"{TEST_DATA_PATH}")
    
    # Vérifier si la colonne ID est présente
    ids = df['id'] if 'id' in df.columns else None
    df = df.drop(columns=['id'], errors='ignore')  # Supprimer la colonne ID si elle existe

    # Charger le préprocesseur
    preprocessor = joblib.load(f"{CHECKPOINT_DIR}/preprocessor.pkl")

    # Aligner les colonnes avant transformation
    df_aligned = align_columns(df, preprocessor)

    # Appliquer la transformation avec le pipeline déjà entraîné
    X = preprocessor.pipeline.transform(df_aligned)

    # Appliquer la sélection de features (si elle a été faite à l'entraînement)
    if preprocessor.selector:
        X = preprocessor.selector.transform(X)
    if preprocessor.selected_features_ is not None:
        X = X[:, preprocessor.selected_features_]

    # Vérification de cohérence avec le modèle
    input_size = X.shape[1]
    expected_input_size = len(preprocessor.selected_features_) if preprocessor.selected_features_ is not None else input_size

    print(f"* Expected input size: {expected_input_size}")
    print(f"* Actual input size: {input_size}")

    if input_size != expected_input_size:
        raise ValueError(
            f"/!\ Erreur : X.shape[1] ({input_size}) != expected_input_size ({expected_input_size})\n"
            f"Vérifiez que `feature_selection_method` et `k_best` sont identiques à l'entraînement."
        )
 
    print(f"* Features alignées avec succès : {input_size} (doit être identique au training)")

    # Charger le modèle entraîné
    model = load_model(
        model_path=MODEL_PATH,
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes,
        num_layers=3
    )
    
    # Faire des prédictions par lots
    predictions, probabilities = batch_predict(model, X, batch_size=args.batch_size)

    # Sauvegarde des résultats
    df_results = pd.DataFrame({'prediction': predictions})
    if ids is not None:
        df_results.insert(0, 'id', ids)  # Ajouter la colonne ID si disponible

    df_results.to_csv(f"{CHECKPOINT_DIR}/submission.csv", index=False)

    # Affichage des résultats
    print("... Prédictions réussies !")
    print("... Aperçu des prédictions :\n", df_results.head())
