import torch
import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.feedforward = nn.Sequential(
            nn.BatchNorm1d(63),
            nn.Linear(in_features=63, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=5),
        )
    
    def forward(self, x):
        x = self.feedforward(x)
        return x