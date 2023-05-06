import numpy as np
import sklearn
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import os
from time import sleep
import pandas as pd
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_embeddings(model, loader):

    model.eval()
    embeddings = []
    outputs = []
    targets = []
    datapoints = []

    for data, target in loader:
        data=data.to(device)
        target = target.to(device)
        emb, out = model.forward(data)
        _, predicted = out.max(1)

        embeddings.append(emb.detach().cpu())
        outputs.append(predicted.detach().cpu())
        targets.append(target.detach().cpu())
        datapoints.append(data)
    
    datapoints = torch.cat(datapoints)
    embeddings = torch.cat(embeddings)
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)

    return datapoints.cpu(), embeddings, outputs, targets


def get_decision_boundary(model, low=(0,0), high=(1,1), latent_space=True, num_points=10000, latent_space_dim=2):

    X = np.random.uniform(low=low, high=high, size=(num_points,latent_space_dim))
    X = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        if latent_space:
            pred = model.classifier(X)
        else:
            _, pred = model(X)

    _, pred = pred.max(1)
    return X.cpu(), pred


def plot_space(worker_ind, model, epoch, loader1, loader2, loader3, loader4, logdir, latent_space_dim=2):

    low=[-5]*2
    high=[5]*2
    X, pred = get_decision_boundary(model, low, high, latent_space=False, latent_space_dim=latent_space_dim)

    if worker_ind == 0:
        d1, e1, o1, t1 = get_embeddings(model, loader1)
    elif worker_ind == 1:
        d1, e1, o1, t1 = get_embeddings(model, loader2)
    elif worker_ind == 2:
        d1, e1, o1, t1 = get_embeddings(model, loader3)
    elif worker_ind == 3:
        d1, e1, o1, t1 = get_embeddings(model, loader4)

    fig, ax = plt.subplots()

    plt.scatter(X[pred==0, 0],X[pred==0,1], color='lightgray')
    plt.scatter(X[pred==1, 0],X[pred==1,1], color='lightgreen')
    plt.scatter(X[pred==2, 0],X[pred==2,1], color='lavender')
    plt.scatter(X[pred==3, 0],X[pred==3,1], color='beige')

    plt.scatter(d1[t1==0,0], d1[t1==0,1], marker='+', color='blue')
    plt.scatter(d1[t1==1,0], d1[t1==1,1], marker='x', color='red')
    plt.scatter(d1[t1==2,0], d1[t1==2,1], marker='_', color='magenta')
    plt.scatter(d1[t1==3,0], d1[t1==3,1], marker='.', color='darkgreen')

    plt.suptitle(f'Original Space of worker:{worker_ind} at epoch:{epoch}')
    fig.tight_layout()
    plt.savefig(os.path.join(logdir, f'original-{worker_ind}-{epoch}.pdf'), format="pdf") 
    plt.savefig(os.path.join(logdir, f'original-{worker_ind}-{epoch}.png'), format="png") 
    plt.close()

    ###############################

    low=[0]*latent_space_dim
    high=[1]*latent_space_dim
    X, pred = get_decision_boundary(model, low, high, latent_space=True, latent_space_dim=latent_space_dim)

    fig, ax = plt.subplots()
    plt.scatter(X[pred==0, 0],X[pred==0,1], color='lightgray')
    plt.scatter(X[pred==1, 0],X[pred==1,1], color='lightgreen')
    plt.scatter(X[pred==2, 0],X[pred==2,1], color='lavender')
    plt.scatter(X[pred==3, 0],X[pred==3,1], color='beige')

    plt.scatter(e1[t1==0,0], e1[t1==0,1], marker='+', color='blue')
    plt.scatter(e1[t1==1,0], e1[t1==1,1], marker='x', color='red')
    plt.scatter(e1[t1==2,0], e1[t1==2,1], marker='_', color='magenta')
    plt.scatter(e1[t1==3,0], e1[t1==3,1], marker='.', color='darkgreen')
    
    plt.suptitle(f'Latent Space of worker:{worker_ind} at epoch:{epoch}')
    fig.tight_layout()
    plt.savefig(os.path.join(logdir, f'latent-{worker_ind}-{epoch}.pdf'), format="pdf") 
    plt.savefig(os.path.join(logdir, f'latent-{worker_ind}-{epoch}.png'), format="png") 
    plt.close()

    return


def plot_moments(global_moments, local_moments, logdir, epoch):

    # global_moments : num_classes x embedding_dim
    # local_moments : num_workers x num_classes x embedding_dim
    markers = {
        0: '+',
        1: 'x',
        2: '_',
        3: '.'
    }

    colors=['blue','red','magenta','darkgreen']
    plt.figure()
    ax = plt.axes()
    ax.set_facecolor('gainsboro')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])

    num_classes=global_moments.shape[0]
    num_workers=local_moments.shape[0]

    for cls_ in range(num_classes):
        plt.scatter(global_moments[cls_, 0], global_moments[cls_, 1], marker=markers[cls_], color='black')

    for wrk in range(num_workers):
        for cls_ in range(num_classes):
            clr=colors[wrk]
            plt.scatter(local_moments[wrk, cls_, 0], local_moments[wrk, cls_, 1], marker=markers[cls_], color=clr)

    plt.grid()

    plt.title(f'Latent Space Moments at epoch:{epoch}')
    plt.savefig(os.path.join(logdir, f'latentmoments-{epoch}.pdf'), format="pdf") 
    plt.savefig(os.path.join(logdir, f'latentmoments-{epoch}.png'), format="png") 
    plt.close()


