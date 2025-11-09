import torch.nn as nn

################
# MLP 3-layers #
################
class Net_3layers(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden1=64, hidden2=32, dropout=0.6):
        super(Net_3layers, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)


################
# MLP 5-layers #
################
class Net_5layers(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden1=128, hidden2=64, hidden3=32, hidden4=16, dropout=0.6):
        super(Net_5layers, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden3, hidden4),
            nn.ReLU(),

            nn.Linear(hidden4, output_dim)
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)
