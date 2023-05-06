import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset


class SingleLayerNet(nn.Module):
    def __init__(self, num_workers=2, hsize=84):
        super(SingleLayerNet, self).__init__()
        self.classifier   = nn.Linear(hsize, num_workers)

    def forward(self, x):
        out = self.classifier(x)
        return out


def inference(model, loader, device='cpu'):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100*correct/total


def train_discriminator(
    model, 
    data_loader, 
    device='cpu', 
    num_epochs=100,
    lr=0.01,
    weight_decay=5e-4,
    writer=None,
    data_epoch=0,
    ):

    optimizer = optim.SGD(
        model.parameters(), 
        lr=lr,momentum=0.9, 
        weight_decay=weight_decay
    )

    loss_fn = nn.CrossEntropyLoss()


    iter = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        total = 0
        correct = 0

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            iter += 1
            inputs, targets = inputs.to(device), targets.to(device)
            if torch.isnan(inputs).any():
                print('nan embeddings')
                raise

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_accuracy = 100*correct/total
        if writer is not None:
            writer.add_scalar(f'Divergence/Acc-{data_epoch}', train_accuracy, epoch)
            writer.add_scalar(f'Divergence/Loss-{data_epoch}', train_loss/(batch_idx+1), epoch)

    final_accuracy = inference(model, data_loader, device)/100

    return final_accuracy, 2*final_accuracy - 1


def create_dataset_from_saved_embeddings(loaded_data, labels=(), workers=()):
    label_count=0
    dataset = []
    targets = []
    for worker in workers:
        for label in labels:
            temp = loaded_data[worker]['embeddings'][loaded_data[worker]['labels'] == label]
            size = temp.shape[0]
            temp_label = torch.LongTensor([label_count]*size)

            dataset.append(temp)
            targets.append(temp_label)

        label_count += 1

    dataset = torch.cat(dataset)
    targets = torch.cat(targets)

    return TensorDataset(dataset, targets)

def create_dataset_from_saved_embeddings_fedavg(loaded_data, fed_workers=(), new_worker=()):
    dataset = []
    targets = []
    for worker in fed_workers:
        temp = loaded_data[worker]['embeddings']
        size = temp.shape[0]
        temp_label = torch.LongTensor([1]*size)

        dataset.append(temp)
        targets.append(temp_label)

    for worker in new_worker:
        temp = loaded_data[worker]['embeddings']
        size = temp.shape[0]
        temp_label = torch.LongTensor([0]*size)

        dataset.append(temp)
        targets.append(temp_label)    

    dataset = torch.cat(dataset)
    targets = torch.cat(targets)

    return TensorDataset(dataset, targets)
