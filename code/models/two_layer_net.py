import torch
from torch import nn
from torch.nn import functional as F

class TwoLayerNet(nn.Module):
    def __init__(self, inp=32, hidden=512, latent=100):
        super(TwoLayerNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inp*inp, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.classifier=nn.Linear(latent,10)

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_relu_stack(x)
        embeddings = torch.sigmoid(out)
        out = self.classifier(embeddings)
        return embeddings, out


class SynthNetwork(nn.Module):
    def __init__(self, in_dim=2, h_dim=3, out_dim=2, num_classes=4):
        super(SynthNetwork, self).__init__()

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)
        
    def forward(self, x):
        y=F.relu(self.fc1(x))
        y=torch.sigmoid(self.fc2(y))
        out = self.classifier(y)
        return y, out
