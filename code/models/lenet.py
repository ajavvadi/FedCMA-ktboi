'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, embedding_dim=84):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, embedding_dim)
        self.classifier   = nn.Linear(embedding_dim, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        embeddings = F.relu(self.fc2(out))
        out = self.classifier(embeddings)
        return embeddings, out

    def forward_embeddings(self, x):
        return self.classifier(x)

class LeNetBounded(LeNet):
    def __init__(self,embedding_dim=84):
        super(LeNetBounded, self).__init__(embedding_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        embeddings = torch.sigmoid(self.fc2(out))
        out = self.classifier(embeddings)
        return embeddings, out

    def forward_embeddings(self, x):
        return self.classifier(x)


class LeNet1(nn.Module):
    def __init__(self, embedding_dim=84):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, embedding_dim)
        # self.fc2   = nn.Linear(120, 84)
        self.classifier   = nn.Linear(embedding_dim, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        embeddings = F.relu(self.fc1(out))
        # embeddings = F.relu(self.fc2(out))
        out = self.classifier(embeddings)
        return embeddings, out

    def forward_embeddings(self, x):
        return self.classifier(x)        


class LeNetBounded1(LeNet1):
    def __init__(self, embedding_dim=84):
        super(LeNetBounded1, self).__init__(embedding_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        embeddings = torch.sigmoid(self.fc1(out))
        out = self.classifier(embeddings)
        return embeddings, out

    def forward_embeddings(self, x):
        return self.classifier(x)
