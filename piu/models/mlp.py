import torch.nn as nn

class MultiClassNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiClassNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        """
        Modèle MLP

        :param input_size: Nombre de features en entrée
        :param hidden_sizes: Liste contenant le nombre de neurones par couche cachée (ex: [128, 64, 32])
        :param num_classes: Nombre de classes en sortie
        :param dropout_rate: Taux de dropout 
        """
        super(MultiLayerPerceptron, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            # layers.append(nn.Tanh())
            layers.append(nn.ReLU())
            # layers.append(nn.BatchNorm1d(hidden_size))
            # layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
