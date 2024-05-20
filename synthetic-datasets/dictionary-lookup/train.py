import os
import dgl
import torch
import random
import argparse
import numpy as np
from torch import nn
from torchinfo import summary
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from data import DictionaryLookupDataset
from model import SIRModel, GCNModel, SAGEModel, GATModel, GINModel

loss_fn = nn.CrossEntropyLoss()
acc_fn = lambda logits, labels: torch.mean(logits.argmax(dim=-1) == labels, dtype=torch.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def train(model, train_loader, device, optimizer):
    model.train()

    total_loss, total = 0, 0
    for graphs in train_loader:
        graphs = graphs.to(device)

        optimizer.zero_grad()
        torch.cuda.empty_cache()

        feats = graphs.ndata.pop('feat')
        mask = graphs.ndata.pop('mask')
        labels = feats[:, 1].to(torch.int64)
        logits = model(graphs, feats)
        loss = loss_fn(logits[mask], labels[~mask])

        loss.backward()
        optimizer.step()

        total = total + labels[~mask].shape[0]
        total_loss = total_loss + loss.item() * labels[~mask].shape[0]
    
    return total_loss / total

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss, total_acc, total = 0, 0, 0
    for graphs in dataloader:
        graphs = graphs.to(device)

        feats = graphs.ndata.pop('feat')
        mask = graphs.ndata.pop('mask')
        labels = feats[:, 1].to(torch.int64)
        logits = model(graphs, feats)
        loss = loss_fn(logits[mask], labels[~mask])
        acc = acc_fn(logits[mask], labels[~mask])

        total = total + labels[~mask].shape[0]
        total_loss = total_loss + loss.item() * labels[~mask].shape[0]
        total_acc = total_acc + acc.item() * labels[~mask].shape[0]
    
    return total_loss / total, total_acc / total


def run(model, train_loader, test_loader, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    
    for epoch in range(args.epochs):
        loss = train(model, train_loader, device, optimizer)
        loss, acc = evaluate(model, train_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step(loss)

        if (epoch + 1) == args.epochs or (epoch + 1) % args.log_every == 0:
            print(f'Epoch {epoch+1:04d} | loss: {loss:.4f} | acc: {acc:.4f} | '
                f'test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}')
    
        if loss < 1e-3 and test_loss < 1e-3:
            break

    return acc, test_acc


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'SIR-GCN/GCN/GraphSAGE/GATv2/GIN implementation on DictionaryLookup',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    argparser.add_argument('--seed', type=int, default=0, help='seed')
    
    argparser.add_argument('--model', type=str, default='SIR', help='model name', choices=['SIR', 'GCN', 'SAGE', 'GAT', 'GIN'])
    argparser.add_argument('--nhidden', type=int, default=64, help='number of hidden units')
    argparser.add_argument('--nlayers', type=int, default=1, help='number of graph convolution layers')
    argparser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    argparser.add_argument('--nheads', type=int, default=1, help='number of attention heads for GAT')
    argparser.add_argument('--nlayers-mlp', type=int, default=2, help='number of MLP layers for GIN')
    
    argparser.add_argument('--nodes', type=int, default=10, help='number of nodes in the bipartite graph')
    argparser.add_argument('--samples', type=int, default=5000, help='number of sample permutations')
    argparser.add_argument('--train-size', type=float, default=0.8, help='fraction of samples for training')

    argparser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    argparser.add_argument('--batch-size', type=int, default=256, help='batch size')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--wd', type=float, default=0, help='weight decay')
    argparser.add_argument('--factor', type=float, default=0.5, help='factor for learning rate decay')
    argparser.add_argument('--patience', type=int, default=10, help='patience for learning rate decay')
    
    argparser.add_argument('--nruns', type=int, default=10, help='number of runs')
    argparser.add_argument('--log-every', type=int, default=20, help='log every LOG_EVERY epochs')
    args = argparser.parse_args()

    train_accs, test_accs = [], []
    for i in range(args.nruns):
        # Seed
        set_seed(args.seed + i)

        # Load dataset
        device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
        dataset = DictionaryLookupDataset(args.nodes, args.samples)
        train_sampler = SubsetRandomSampler(torch.arange(int(args.train_size * len(dataset))))
        test_sampler = SubsetRandomSampler(torch.arange(int(args.train_size * len(dataset)), len(dataset)))
        train_loader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=False)
        test_loader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=args.batch_size, drop_last=False)

        # Load model
        Model = {'SIR': SIRModel, 'GCN': GCNModel, 'SAGE': SAGEModel, 'GAT': GATModel, 'GIN': GINModel}
        model = Model[args.model](args.nodes, args.nhidden, args.nodes, args.nlayers, args.dropout, 
                                  num_heads=args.nheads, mlp_layers=args.nlayers_mlp).to(device)
        summary(model)

        # Training
        train_acc, test_acc = run(model, train_loader, test_loader, device, args)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    print(args)
    print(f'Runned {args.nruns} times')
    print(f'Train accuracy: {train_accs}')
    print(f'Test accuracy: {test_accs}')
    print(f'Average train accuracy: {np.mean(train_accs):.6f} ± {np.std(train_accs):.6f}')
    print(f'Average test accuracy: {np.mean(test_accs):.6f} ± {np.std(test_accs):.6f}')
