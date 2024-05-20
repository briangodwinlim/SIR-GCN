import os
import dgl
import torch
import random
import argparse
import numpy as np
from torch import nn
from torchinfo import summary
from dgl.dataloading import GraphDataLoader
from dgl.data import PATTERNDataset, CLUSTERDataset

from model import SIRModel, GATModel


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
    Dataset = PATTERNDataset if name == 'PATTERN' else CLUSTERDataset
    transform = dgl.transforms.AddSelfLoop() if args.add_self_loop else None
    dataset = Dataset(mode='train', transform=transform)
    dataset.input_dim = torch.unique(dataset[0].ndata['feat'], dim=0).shape[0]
    
    train_loader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = GraphDataLoader(Dataset(mode='valid', transform=transform), batch_size=args.batch_size, shuffle=False)
    test_loader = GraphDataLoader(Dataset(mode='test', transform=transform), batch_size=args.batch_size, shuffle=False)
    
    return dataset, train_loader, val_loader, test_loader
    

def regularizer(model, args):
    l1 = torch.sum(torch.stack([torch.sum(torch.abs(param)) for param in model.parameters()]))
    l2 = torch.sum(torch.stack([torch.sum(torch.pow(param, 2)) for param in model.parameters()]))
    return (args.l1 * l1 + args.l2 * l2) * int(model.training)

def loss_fn(logits, labels):
    weight = torch.tensor([torch.sum(labels == c) for c in range(logits.shape[1])], device=labels.device)
    weight = (labels.shape[0] - weight) * (weight > 0) / labels.shape[0]
    loss_fn_ = nn.CrossEntropyLoss(weight=weight)
    return loss_fn_(logits, labels)

def acc_fn(logits, labels):
    preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
    classes = torch.unique(torch.cat([labels, preds], dim=-1))
    return torch.mean(torch.tensor([torch.mean(preds[labels == c] == c, dtype=torch.float64) if (labels == c).any() else 0 for c in classes]))


def train(model, train_loader, device, optimizer, args):
    model.train()

    total_loss, total = 0, 0
    for graphs in train_loader:
        graphs = graphs.to(device)

        optimizer.zero_grad()
        torch.cuda.empty_cache()

        feats = graphs.ndata.pop('feat')
        labels = graphs.ndata.pop('label').to(torch.int64)
        logits = model(graphs, feats)
        loss = loss_fn(logits, labels) + regularizer(model, args)

        loss.backward()
        optimizer.step()

        total = total + 1
        total_loss = total_loss + loss.item()
        
    return total_loss / total - regularizer(model, args).item()

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss, total_acc, total = 0, 0, 0
    for graphs in dataloader:
        graphs = graphs.to(device)
        
        feats = graphs.ndata.pop('feat')
        labels = graphs.ndata.pop('label').to(torch.int64)
        logits = model(graphs, feats)
        
        loss = loss_fn(logits, labels)
        acc = acc_fn(logits, labels)

        total = total + 1
        total_loss = total_loss + loss.item()
        total_acc = total_acc + acc.item()

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
        'SIR-GCN/GATv2 implementation on SBMDataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    argparser.add_argument('--seed', type=int, default=0, help='seed')

    argparser.add_argument('--model', type=str, default='SIR', help='model name', choices=['SIR', 'GAT']) 
    argparser.add_argument('--nhidden', type=int, default=64, help='number of hidden units')
    argparser.add_argument('--nlayers', type=int, default=4, help='number of graph convolution layers')
    argparser.add_argument('--input-dropout', type=float, default=0, help='input dropout rate')
    argparser.add_argument('--edge-dropout', type=float, default=0, help='edge dropout rate')
    argparser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    argparser.add_argument('--norm', type=str, default='none', help='type of normalization', choices=['cn', 'bn', 'ln', 'none'])
    argparser.add_argument('--readout-layers', type=int, default=1, help='number of MLP layers for node readout')
    argparser.add_argument('--readout-dropout', type=float, default=0, help='dropout rate for node readout')
    argparser.add_argument('--jumping-knowledge', action='store_true', help='use jumping knowledge for node readout')
    argparser.add_argument('--residual', action='store_true', help='add residual connections')
    argparser.add_argument('--resid-layers', type=int, default=0, help='number of MLP layers for SIR residual')
    argparser.add_argument('--resid-dropout', type=float, default=0, help='dropout rate for SIR residual')
    argparser.add_argument('--feat-dropout', type=float, default=0, help='dropout rate for SIR inner linear transformations')
    argparser.add_argument('--agg-type', type=str, default='mean', help='aggregation type for SIR', choices=['sum', 'max', 'mean', 'sym'])
    argparser.add_argument('--nheads', type=int, default=1, help='number of attention heads for GAT')
    argparser.add_argument('--attn-dropout', type=float, default=0, help='dropout rate for GAT attention')
    
    argparser.add_argument('--dataset', type=str, default='PATTERN', help='name of dataset', choices=['PATTERN', 'CLUSTER'])
    argparser.add_argument('--add-self-loop', action='store_true', help='add self-loop to graph')

    argparser.add_argument('--epochs', type=int, default=500, help='number of epochs')
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

    val_accs, test_accs = [], []
    for i in range(args.nruns):
        # Set seed
        set_seed(args.seed + i)

        # Load dataset
        device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
        dataset, train_loader, val_loader, test_loader = load_dataset(args.dataset, args)

        # Load model
        Model = {'SIR': SIRModel, 'GAT': GATModel}
        model = Model[args.model](dataset.input_dim, args.nhidden, dataset.num_classes, args.nlayers, args.input_dropout, args.edge_dropout, args.dropout, args.norm,
                                  args.readout_layers, args.readout_dropout, args.jumping_knowledge, args.residual,
                                  resid_layers=args.resid_layers, resid_dropout=args.resid_dropout, 
                                  feat_dropout=args.feat_dropout, agg_type=args.agg_type, 
                                  num_heads=args.nheads, attn_dropout=args.attn_dropout).to(device)
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

# SIR-GCN (PATTERN)
# Namespace(cpu=False, gpu=0, seed=0, model='SIR', nhidden=80, nlayers=4, input_dropout=0.0, edge_dropout=0.0, dropout=0.0, norm='bn', readout_layers=1, readout_dropout=0, jumping_knowledge=False, residual=True, resid_layers=1, resid_dropout=0.0, feat_dropout=0.0, agg_type='sym', nheads=1, attn_dropout=0, dataset='PATTERN', add_self_loop=False, epochs=200, batch_size=128, lr=0.001, wd=0, l1=1e-07, l2=1e-07, factor=0.5, patience=10, nruns=10, log_every=20)
# Runned 10 times
# Val accuracy: [0.8553708328519818, 0.8559300003697947, 0.8557156018893362, 0.8552037567717476, 0.855003024428295, 0.8552268848676241, 0.8555905842171652, 0.8555918125178158, 0.8556887522088549, 0.8554824850724898]
# Test accuracy: [0.8578409226125989, 0.8573532380193513, 0.8578173063190699, 0.8577266946454847, 0.8571742415992891, 0.857161994407504, 0.8575154752307285, 0.8576665802377175, 0.8573040269497781, 0.8578633206813249]
# Average val accuracy: 0.855480 ± 0.000266
# Average test accuracy: 0.857542 ± 0.000263

# SIR-GCN (CLUSTER)
# Namespace(cpu=False, gpu=0, seed=0, model='SIR', nhidden=80, nlayers=4, input_dropout=0.0, edge_dropout=0.0, dropout=0.0, norm='bn', readout_layers=1, readout_dropout=0, jumping_knowledge=False, residual=True, resid_layers=1, resid_dropout=0.0, feat_dropout=0.0, agg_type='sym', nheads=1, attn_dropout=0, dataset='CLUSTER', add_self_loop=False, epochs=200, batch_size=128, lr=0.001, wd=0, l1=1e-07, l2=1e-07, factor=0.5, patience=10, nruns=10, log_every=20)
# Runned 10 times
# Val accuracy: [0.6307322425536895, 0.6312298769172006, 0.6311445039759314, 0.6348065737717603, 0.6321617616787373, 0.6294313856295919, 0.6317468506670699, 0.6348588066364323, 0.6311169393196531, 0.6310404236199434]
# Test accuracy: [0.6341520069890344, 0.6333402545724814, 0.6339084697597396, 0.6372307379755761, 0.6336662163850936, 0.6308592198616454, 0.6317459834106022, 0.6358317491022101, 0.6328929586602562, 0.6315072863878071]
# Average val accuracy: 0.631827 ± 0.001645
# Average test accuracy: 0.633513 ± 0.001854
