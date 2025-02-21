import torch
import torch.nn as nn

class LSTMWithTabular(nn.Module):
    def __init__(self, input_dim_seq, hidden_dim, num_layers, input_dim_static, output_dim):
        super(LSTMWithTabular, self).__init__()

        # 🔹 LSTM pour la séquence temporelle
        self.lstm = nn.LSTM(input_dim_seq, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # 🔹 MLP pour les features tabulaires
        self.fc_static = nn.Sequential(
            nn.Linear(input_dim_static, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 🔹 Fusion et prédiction finale
        self.fc_final = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, X_seq, X_static):
        # 🔹 LSTM : récupérer la dernière sortie cachée
        lstm_out, _ = self.lstm(X_seq)
        lstm_out = lstm_out[:, -1, :]  # Dernière sortie de la séquence
        
        # 🔹 MLP pour les features tabulaires
        static_out = self.fc_static(X_static)

        # 🔹 Fusion et prédiction
        combined = torch.cat((lstm_out, static_out), dim=1)
        output = self.fc_final(combined)
        
        return output
