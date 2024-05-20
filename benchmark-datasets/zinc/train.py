import os
import dgl
import torch
import random
import argparse
import numpy as np
from torch import nn
from torchinfo import summary
from dgl.data import ZINCDataset
from dgl.dataloading import GraphDataLoader

from model import SIRModel, GINModel

loss_fn = nn.L1Loss()
mae_fn = nn.L1Loss()


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


def warmup_lr(optimizer, lr, epoch, size):
    if epoch <= size:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * epoch / size


def load_dataset(args):
    dataset = ZINCDataset(mode='train')
    dataset.output_dim = 1
    
    transform = dgl.transforms.AddSelfLoop() if args.add_self_loop else None
    train_loader = GraphDataLoader(ZINCDataset(mode='train', transform=transform), batch_size=args.batch_size, shuffle=True)
    val_loader = GraphDataLoader(ZINCDataset(mode='valid', transform=transform), batch_size=args.batch_size, shuffle=False)
    test_loader = GraphDataLoader(ZINCDataset(mode='test', transform=transform), batch_size=args.batch_size, shuffle=False)
    
    return dataset, train_loader, val_loader, test_loader


def regularizer(model, args):
    l1 = torch.sum(torch.stack([torch.sum(torch.abs(param)) for param in model.parameters()]))
    l2 = torch.sum(torch.stack([torch.sum(torch.pow(param, 2)) for param in model.parameters()]))
    return (args.l1 * l1 + args.l2 * l2) * int(model.training)


def train(model, train_loader, device, optimizer, args):
    model.train()

    total_loss, total = 0, 0
    for graphs, targets in train_loader:
        graphs = graphs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        torch.cuda.empty_cache()

        nfeats = graphs.ndata.pop('feat')
        efeats = graphs.edata.pop('feat')
        preds = model(graphs, nfeats, efeats).squeeze()
        loss = loss_fn(preds, targets) + regularizer(model, args)

        loss.backward()
        optimizer.step()

        total = total + 1
        total_loss = total_loss + loss.item()
        
    return total_loss / total - regularizer(model, args).item()

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss, total_mae, total = 0, 0, 0
    for graphs, targets in dataloader:
        graphs = graphs.to(device)
        targets = targets.to(device)

        nfeats = graphs.ndata.pop('feat')
        efeats = graphs.edata.pop('feat')
        preds = model(graphs, nfeats, efeats).squeeze()
        loss = loss_fn(preds, targets)
        mae = mae_fn(preds, targets)

        total = total + 1
        total_loss = total_loss + loss.item()
        total_mae = total_mae + mae.item()

    return total_loss / total, total_mae / total


def run(model, train_loader, val_loader, test_loader, device, args, iter):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    best_val_mae = 1e10

    for epoch in range(args.epochs):
        warmup_lr(optimizer, args.lr, epoch + 1, 10)
        loss = train(model, train_loader, device, optimizer, args)
        loss, mae = evaluate(model, train_loader, device)
        val_loss, val_mae = evaluate(model, val_loader, device)
        test_loss, test_mae = evaluate(model, test_loader, device)
        scheduler.step(loss)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            result = {
                'val_loss': val_loss,
                'val_mae': val_mae,
                'test_loss': test_loss,
                'test_mae': test_mae,
            }

        if (epoch + 1) == args.epochs or (epoch + 1) % args.log_every == 0:
            print(f'Epoch {epoch+1:04d} | loss: {loss:.4f} | mae: {mae:.4f} | '
                  f'val_loss: {val_loss:.4f} | val_mae: {val_mae:.4f} | '
                  f'test_loss: {test_loss:.4f} | test_mae: {test_mae:.4f}')

    return result


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'SIR-GCN/GIN implementation on ZINCDataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    argparser.add_argument('--seed', type=int, default=0, help='seed')

    argparser.add_argument('--model', type=str, default='SIR', help='model name', choices=['SIR', 'GIN']) 
    argparser.add_argument('--nhidden', type=int, default=64, help='number of hidden units')
    argparser.add_argument('--nlayers', type=int, default=4, help='number of graph convolution layers')
    argparser.add_argument('--input-dropout', type=float, default=0, help='input dropout rate')
    argparser.add_argument('--edge-dropout', type=float, default=0, help='edge dropout rate')
    argparser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    argparser.add_argument('--norm', type=str, default='none', help='type of normalization', choices=['gn', 'cn', 'bn', 'ln', 'none'])
    argparser.add_argument('--readout-layers', type=int, default=1, help='number of MLP layers for graph readout')
    argparser.add_argument('--readout-dropout', type=float, default=0, help='dropout rate for graph readout')
    argparser.add_argument('--readout-pooling', type=str, default='sum', help='type of graph readout pooling', choices=['sum', 'mean'])
    argparser.add_argument('--jumping-knowledge', action='store_true', help='use jumping knowledge for graph readout')
    argparser.add_argument('--residual', action='store_true', help='add residual connections')
    argparser.add_argument('--resid-layers', type=int, default=0, help='number of MLP layers for residual connections')
    argparser.add_argument('--resid-dropout', type=float, default=0, help='dropout rate for residual connections')
    argparser.add_argument('--feat-dropout', type=float, default=0, help='dropout rate for SIR inner linear transformations')
    argparser.add_argument('--agg-type', type=str, default='sum', help='aggregation type', choices=['sum', 'max', 'mean', 'sym'])
    argparser.add_argument('--nlayers-mlp', type=int, default=2, help='number of MLP layers for GIN')

    argparser.add_argument('--add-self-loop', action='store_true', help='add self-loop to graphs')
    
    argparser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    argparser.add_argument('--batch-size', type=int, default=32, help='batch size')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--wd', type=float, default=0, help='weight decay')
    argparser.add_argument('--l1', type=float, default=0, help='weight for L1 regularization')
    argparser.add_argument('--l2', type=float, default=0, help='weight for L2 regularization')
    argparser.add_argument('--factor', type=float, default=0.5, help='factor for learning rate decay')
    argparser.add_argument('--patience', type=int, default=10, help='patience for learning rate decay')
    
    argparser.add_argument('--nruns', type=int, default=10, help='number of runs')
    argparser.add_argument('--log-every', type=int, default=20, help='log every LOG_EVERY epochs')
    args = argparser.parse_args()
    
    if args.model == 'GIN' and args.agg_type == 'sym':
        raise ValueError('GIN cannot use agg_type == sym')

    val_maes, test_maes = [], []
    for i in range(args.nruns):
        # Set seed
        set_seed(args.seed + i)

        # Load dataset
        device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
        dataset, train_loader, val_loader, test_loader = load_dataset(args)

        # Load model
        Model = {'SIR': SIRModel, 'GIN': GINModel}
        model = Model[args.model](dataset.num_atom_types, dataset.num_bond_types, args.nhidden, dataset.output_dim, args.nlayers, args.input_dropout, args.edge_dropout, args.dropout, args.norm, 
                                  args.readout_layers, args.readout_dropout, args.readout_pooling, args.jumping_knowledge,
                                  args.residual, args.resid_layers, args.resid_dropout, feat_dropout=args.feat_dropout, agg_type=args.agg_type, 
                                  mlp_layers=args.nlayers_mlp).to(device)
        summary(model)
        
        # Training
        result = run(model, train_loader, val_loader, test_loader, device, args, i)
        val_maes.append(result['val_mae'])
        test_maes.append(result['test_mae'])

    print(args)
    print(f'Runned {args.nruns} times')
    print(f'Val MAE: {val_maes}')
    print(f'Test MAE: {test_maes}')
    print(f'Average val MAE: {np.mean(val_maes):.6f} ± {np.std(val_maes):.6f}')
    print(f'Average test MAE: {np.mean(test_maes):.6f} ± {np.std(test_maes):.6f}')

# SIR-GCN
# Namespace(cpu=False, gpu=0, seed=0, model='SIR', nhidden=75, nlayers=4, input_dropout=0.0, edge_dropout=0.0, dropout=0.0, norm='bn', readout_layers=2, readout_dropout=0.0, readout_pooling='sum', jumping_knowledge=False, residual=True, resid_layers=1, resid_dropout=0.0, feat_dropout=0.0, agg_type='sym', nlayers_mlp=2, add_self_loop=False, epochs=500, batch_size=128, lr=0.001, wd=0, l1=1e-07, l2=1e-07, factor=0.5, patience=10, nruns=10, log_every=20)
# Runned 10 times
# Val MAE: [0.3003452941775322, 0.31924823485314846, 0.2992000598460436, 0.2989423889666796, 0.3174314256757498, 0.28990969993174076, 0.2804796826094389, 0.3460285849869251, 0.2717738803476095, 0.2988874576985836]
# Test MAE: [0.2670917548239231, 0.3026384189724922, 0.2689075209200382, 0.26405042223632336, 0.29450452513992786, 0.26250266656279564, 0.2529272064566612, 0.3327321894466877, 0.2528698015958071, 0.28352559730410576]
# Average val MAE: 0.302225 ± 0.020065
# Average test MAE: 0.278175 ± 0.024087
