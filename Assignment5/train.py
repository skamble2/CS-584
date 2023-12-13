from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch,idx_train,idx_val):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    '''print('Epoch: {:04d}'.format(epoch+1),
           'loss_train: {:.4f}'.format(loss_train.item()),
           'acc_train: {:.4f}'.format(acc_train.item()),
           'loss_val: {:.4f}'.format(loss_val.item()),
           'acc_val: {:.4f}'.format(acc_val.item()),
           'time: {:.4f}s'.format(time.time() - t))
           '''
    return [epoch+1,loss_train.item(),acc_train.item(),loss_val.item(),acc_val.item(),time.time()-t]


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Accuracy= {:.4f}".format(acc_test.item()))
    return [acc_test.item()]


start_time = time.time()


overall_results = []
evaluation_results = []


training_ranges = [60, 120, 180, 240, 300]


for train_range in training_ranges:

    epoch_results = []
    for epoch in range(args.epochs):
        
        epoch_data = train(epoch, idx_train=range(train_range), idx_val=range(train_range, 500))
        epoch_results.append(epoch_data)

    
    epoch_results = np.array(epoch_results)
    overall_results.append(epoch_results)

    
    evaluation_results.append(test())



overall_results = np.array(overall_results)
evaluation_results = np.array(evaluation_results)


fig_test = plt.figure(figsize=(5, 5))
plt.suptitle("Testing Set Results")
plt.style.use('bmh')

plt.plot(training_ranges, evaluation_results, marker='s', color='maroon')
plt.title("Data vs Testing Accuracy")

plt.tight_layout()
plt.show()


