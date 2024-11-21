import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim) -> None:
        super(Model, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    
    def forward(self, x):
        return self.network(x)