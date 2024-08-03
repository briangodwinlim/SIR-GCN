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

from data import GraphHeterophilyDataset
from model import SIRModel, GCNModel, SAGEModel, GATModel, GINModel, PNAModel

loss_fn = nn.MSELoss()


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
    for graphs, targets in train_loader:
        graphs = graphs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        torch.cuda.empty_cache()

        labels = graphs.ndata.pop('label')
        pred_targets = model(graphs, labels)
        loss = loss_fn(targets, pred_targets)
        
        loss.backward()
        optimizer.step()

        total = total + labels.shape[0]
        total_loss = total_loss + loss.item() * labels.shape[0]
    
    return total_loss / total

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss, total = 0, 0
    for graphs, targets in dataloader:
        graphs = graphs.to(device)
        targets = targets.to(device)

        labels = graphs.ndata.pop('label')
        pred_targets = model(graphs, labels)
        loss = loss_fn(targets, pred_targets)
        
        total = total + labels.shape[0]
        total_loss = total_loss + loss.item() * labels.shape[0]

    return total_loss / total


def run(model, train_loader, test_loader, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    
    for epoch in range(args.epochs):
        loss = train(model, train_loader, device, optimizer)
        loss = evaluate(model, train_loader, device)
        test_loss = evaluate(model, test_loader, device)
        scheduler.step(loss)

        if (epoch + 1) == args.epochs or (epoch + 1) % args.log_every == 0:
            print(f'Epoch {epoch+1:04d} | loss: {loss:.4f} | test_loss: {test_loss:.4f}')

        if loss < 1e-3 and test_loss < 1e-3:
            break

    return loss, test_loss


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'SIR-GCN/GCN/GraphSAGE/GATv2/GIN/PNA implementation on GraphHeterophily',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    argparser.add_argument('--seed', type=int, default=0, help='seed')
    
    argparser.add_argument('--model', type=str, default='SIR', help='model name', choices=['SIR', 'GCN', 'SAGE', 'GAT', 'GIN', 'PNA'])
    argparser.add_argument('--nhidden', type=int, default=64, help='number of hidden units')
    argparser.add_argument('--nlayers', type=int, default=1, help='number of graph convolution layers')
    argparser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    argparser.add_argument('--nheads', type=int, default=1, help='number of attention heads for GAT')
    argparser.add_argument('--nlayers-mlp', type=int, default=1, help='number of MLP layers for GIN')
    
    argparser.add_argument('--nodes', type=int, default=50, help='maximum number of nodes in random graphs')
    argparser.add_argument('--classes', type=int, default=5, help='number of classes for node labels')
    argparser.add_argument('--normalize', action='store_true', help='normalize target with number of edges')
    argparser.add_argument('--samples', type=int, default=5000, help='number of random graphs')
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

    train_losses, test_losses = [], []
    for i in range(args.nruns):
        # Seed
        set_seed(args.seed + i)

        # Load dataset
        device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
        dataset = GraphHeterophilyDataset(args.nodes, args.classes, args.samples, args.normalize)
        train_sampler = SubsetRandomSampler(torch.arange(int(args.train_size * len(dataset))))
        test_sampler = SubsetRandomSampler(torch.arange(int(args.train_size * len(dataset)), len(dataset)))
        train_loader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=False)
        test_loader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=args.batch_size, drop_last=False)

        # Load model
        Model = {'SIR': SIRModel, 'GCN': GCNModel, 'SAGE': SAGEModel, 'GAT': GATModel, 'GIN': GINModel, 'PNA': PNAModel}
        model = Model[args.model](args.classes, args.nhidden, 1, args.nlayers, args.dropout, 
                                  num_heads=args.nheads, mlp_layers=args.nlayers_mlp).to(device)
        summary(model)

        # Training
        train_loss, test_loss = run(model, train_loader, test_loader, device, args)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    print(args)
    print(f'Runned {args.nruns} times')
    print(f'Train loss: {train_losses}')
    print(f'Test loss: {test_losses}')
    print(f'Average train loss: {np.mean(train_losses):.6f} ± {np.std(train_losses):.6f}')
    print(f'Average test loss: {np.mean(test_losses):.6f} ± {np.std(test_losses):.6f}')
