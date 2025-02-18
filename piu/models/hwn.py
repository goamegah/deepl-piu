import torch
import torch.nn as nn
import torch.nn.functional as F

class HighwayNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=3, dropout_rate=0.3):
        """
        input_size: Nombre de features en entrée
        hidden_size: Nombre de neurones dans toutes les couches cachées (identique)
        num_classes: Nombre de classes pour la classification
        num_layers: Nombre total de couches de transformation
        dropout_rate: Probabilité du Dropout pour régularisation
        """
        super(HighwayNet, self).__init__()
        self.num_layers = num_layers

        # 🔥 Projeter input_size vers hidden_size si nécessaire
        self.input_projection = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

        # 🔥 Création des couches
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.transform_gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

        # 🔥 Normalisation et Dropout
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers)])

        # 🔥 Couche de sortie pour classification
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.input_projection(x)  # 🔥 Ajustement si input_size ≠ hidden_size

        for i in range(self.num_layers):
            h = self.layers[i](x)  # Transformation principale
            h = self.batch_norms[i](h)  # Normalisation pour stabiliser l'entraînement
            h = F.relu(h)  # Activation non linéaire
            h = self.dropouts[i](h)  # Dropout pour éviter l'overfitting

            t = torch.sigmoid(self.transform_gates[i](x))  # Gate de contrôle
            x = t * h + (1 - t) * x  # 🔥 Fusion entre transformation et identité

        return self.output_layer(x)  # ✅ Couche finale pour classification
