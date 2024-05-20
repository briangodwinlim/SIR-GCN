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
from dgl.data import MNISTSuperPixelDataset, CIFAR10SuperPixelDataset

from model import SIRModel, GINModel

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


def warmup_lr(optimizer, lr, epoch, size):
    if epoch <= size:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * epoch / size


def load_dataset(name, args):
    Dataset = MNISTSuperPixelDataset if name == 'MNIST' else CIFAR10SuperPixelDataset
    
    transform = dgl.transforms.AddSelfLoop() if args.add_self_loop else None
    dataset = Dataset(split='train', use_feature=args.use_feature, transform=transform)
    dataset.input_dim = dataset[0][0].ndata['feat'].shape[1]
    dataset.edge_dim = dataset[0][0].edata['feat'].shape[1]
    dataset.output_dim = np.unique(dataset[:][1]).shape[0]
    test_dataset = Dataset(split='test', use_feature=args.use_feature, transform=transform)
    
    train_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(np.arange(len(dataset))[5000:]), batch_size=args.batch_size)
    val_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(np.arange(len(dataset))[:5000]), batch_size=args.batch_size)
    test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return dataset, train_loader, val_loader, test_loader


def regularizer(model, args):
    l1 = torch.sum(torch.stack([torch.sum(torch.abs(param)) for param in model.parameters()]))
    l2 = torch.sum(torch.stack([torch.sum(torch.pow(param, 2)) for param in model.parameters()]))
    return (args.l1 * l1 + args.l2 * l2) * int(model.training)


def train(model, train_loader, device, optimizer, args):
    model.train()

    total_loss, total = 0, 0
    for graphs, labels in train_loader:
        graphs = graphs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        torch.cuda.empty_cache()

        nfeats = graphs.ndata.pop('feat')
        efeats = graphs.edata.pop('feat')
        labels = labels.to(torch.int64)
        logits = model(graphs, nfeats, efeats)
        loss = loss_fn(logits, labels) + regularizer(model, args)

        loss.backward()
        optimizer.step()

        total = total + labels.shape[0]
        total_loss = total_loss + loss.item() * labels.shape[0]
        
    return total_loss / total - regularizer(model, args).item()

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss, total_acc, total = 0, 0, 0
    for graphs, labels in dataloader:
        graphs = graphs.to(device)
        labels = labels.to(device)

        nfeats = graphs.ndata.pop('feat')
        efeats = graphs.edata.pop('feat')
        labels = labels.to(torch.int64)
        logits = model(graphs, nfeats, efeats)
        loss = loss_fn(logits, labels)
        acc = acc_fn(logits, labels)

        total = total + labels.shape[0]
        total_loss = total_loss + loss.item() * labels.shape[0]
        total_acc = total_acc + acc.item() * labels.shape[0]

    return total_loss / total, total_acc / total


def run(model, train_loader, val_loader, test_loader, device, args, iter):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    best_val_loss = 1e10

    for epoch in range(args.epochs):
        warmup_lr(optimizer, args.lr, epoch + 1, 10)
        loss = train(model, train_loader, device, optimizer, args)
        loss, acc = evaluate(model, train_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step(loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            result = {
                'val_loss': val_loss,
                'val_acc': val_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }

        if (epoch + 1) == args.epochs or (epoch + 1) % args.log_every == 0:
            print(f'Epoch {epoch+1:04d} | loss: {loss:.4f} | acc: {acc:.4f} | '
                  f'val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | '
                  f'test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}')

    return result


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'SIR-GCN/GIN implementation on SuperPixelDataset',
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

    argparser.add_argument('--dataset', type=str, default='MNIST', help='name of dataset', choices=['MNIST', 'CIFAR10'])
    argparser.add_argument('--add-self-loop', action='store_true', help='add self-loop to graphs')
    argparser.add_argument('--use-feature', action='store_true', help='use features in addition to super-pixel locations')
    
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

    val_accs, test_accs = [], []
    for i in range(args.nruns):
        # Set seed
        set_seed(args.seed + i)

        # Load dataset
        device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
        dataset, train_loader, val_loader, test_loader = load_dataset(args.dataset, args)

        # Load model
        Model = {'SIR': SIRModel, 'GIN': GINModel}
        model = Model[args.model](dataset.input_dim, dataset.edge_dim, args.nhidden, dataset.output_dim, args.nlayers, args.input_dropout, args.edge_dropout, args.dropout, args.norm, 
                                  args.readout_layers, args.readout_dropout, args.readout_pooling, args.jumping_knowledge,
                                  args.residual, args.resid_layers, args.resid_dropout, feat_dropout=args.feat_dropout, agg_type=args.agg_type, 
                                  mlp_layers=args.nlayers_mlp).to(device)
        summary(model)
        
        # Training
        result = run(model, train_loader, val_loader, test_loader, device, args, i)
        val_accs.append(result['val_acc'])
        test_accs.append(result['test_acc'])

    print(args)
    print(f'Runned {args.nruns} times')
    print(f'Val accuracy: {val_accs}')
    print(f'Test accuracy: {test_accs}')
    print(f'Average val accuracy: {np.mean(val_accs):.6f} ± {np.std(val_accs):.6f}')
    print(f'Average test accuracy: {np.mean(test_accs):.6f} ± {np.std(test_accs):.6f}')

# SIR-GCN (MNIST)
# Namespace(cpu=False, gpu=0, seed=0, model='SIR', nhidden=80, nlayers=4, input_dropout=0.0, edge_dropout=0.0, dropout=0.0, norm='bn', readout_layers=3, readout_dropout=0.0, readout_pooling='mean', jumping_knowledge=False, residual=True, resid_layers=1, resid_dropout=0.2, feat_dropout=0.1, agg_type='max', nlayers_mlp=2, dataset='MNIST', add_self_loop=False, use_feature=False, epochs=200, batch_size=128, lr=0.001, wd=0, l1=1e-06, l2=1e-06, factor=0.5, patience=10, nruns=10, log_every=20)
# Runned 10 times
# Val accuracy: [0.9812, 0.9782, 0.9798, 0.9792, 0.9796, 0.9802, 0.9788, 0.9794, 0.98, 0.9816]
# Test accuracy: [0.9802, 0.9788, 0.9787, 0.9794, 0.9795, 0.9788, 0.9798, 0.9784, 0.9795, 0.9772]
# Average val accuracy: 0.979800 ± 0.000976
# Average test accuracy: 0.979030 ± 0.000806

# SIR-GCN (CIFAR10)
# Namespace(cpu=False, gpu=0, seed=0, model='SIR', nhidden=80, nlayers=4, input_dropout=0.0, edge_dropout=0.0, dropout=0.0, norm='bn', readout_layers=3, readout_dropout=0.0, readout_pooling='mean', jumping_knowledge=False, residual=True, resid_layers=1, resid_dropout=0.2, feat_dropout=0.1, agg_type='max', nlayers_mlp=2, dataset='CIFAR10', add_self_loop=False, use_feature=False, epochs=200, batch_size=128, lr=0.001, wd=0, l1=1e-06, l2=1e-06, factor=0.5, patience=10, nruns=10, log_every=20)
# Runned 10 times
# Val accuracy: [0.7246, 0.7332, 0.736, 0.7356, 0.7432, 0.739, 0.7252, 0.733, 0.7292, 0.7236]
# Test accuracy: [0.7175, 0.723, 0.7246, 0.725, 0.7233, 0.7207, 0.7136, 0.719, 0.7174, 0.7139]
# Average val accuracy: 0.732260 ± 0.006201
# Average test accuracy: 0.719800 ± 0.003979
