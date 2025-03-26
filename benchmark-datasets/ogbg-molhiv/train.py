import os
import dgl
import torch
import random
import argparse
import numpy as np
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator, collate_dgl

from model import SIRModel, GINModel


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
    dataset = DglGraphPropPredDataset(name=name)
    dataset.max_degree = max([data[0].in_degrees().max() for data in dataset])
    evaluator = Evaluator(name=name)

    if args.add_self_loop:
        for i in range(len(dataset)):
            dataset.graphs[i] = dgl.remove_self_loop(dataset.graphs[i])
            dataset.graphs[i] = dgl.add_self_loop(dataset.graphs[i])

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=args.batch_size, shuffle=True, collate_fn=collate_dgl)
    val_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size, shuffle=False, collate_fn=collate_dgl)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=args.batch_size, shuffle=False, collate_fn=collate_dgl)

    return dataset, (train_loader, val_loader, test_loader), evaluator


def regularizer(model, args):
    l1 = torch.sum(torch.stack([torch.sum(torch.abs(param)) for param in model.parameters()]))
    l2 = torch.sum(torch.stack([torch.sum(torch.pow(param, 2)) for param in model.parameters()]))
    return (args.l1 * l1 + args.l2 * l2) * int(model.training)

def loss_fn(logits, labels, model, args):
    return F.binary_cross_entropy(torch.sigmoid(logits).float(), labels.float()) + regularizer(model, args)

def eval_fn(logits, labels, evaluator, eval_metric):
    return evaluator.eval({'y_pred': logits, 'y_true': labels})[eval_metric].item()


def train(model, train_loader, device, optimizer, args):
    model.train()

    total_loss, total = 0, 0
    for graphs, labels in train_loader:
        graphs = graphs.to(device)
        labels = labels.to(device)

        nfeats = graphs.ndata.pop('feat')
        efeats = graphs.edata.pop('feat')

        optimizer.zero_grad()
        torch.cuda.empty_cache()

        m, perturb = 1, 0
        if args.flag:
            m = args.m + 1
            perturb = torch.zeros((nfeats.shape[0], args.nhidden), device=device)
            perturb.uniform_(-args.step_size, args.step_size)
            perturb.requires_grad_()
        
        batch_loss = 0
        for _ in range(m):
            logits = model(graphs, nfeats, efeats, perturb)
            loss = loss_fn(logits, labels, model, args) / m
            batch_loss = batch_loss + loss.item()
            loss.backward()
        
            if args.flag:
                perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
                perturb.data = perturb_data.data
                perturb.grad[:] = 0

        optimizer.step()

        total = total + labels.shape[0]
        total_loss = total_loss + batch_loss * labels.shape[0]
        
    return total_loss / total - regularizer(model, args).item()

@torch.no_grad()
def evaluate(model, dataloader, device, evaluator, args):
    model.eval()

    total_loss, total = 0, 0
    logits_list, labels_list = [], []
    for graphs, labels in dataloader:
        graphs = graphs.to(device)
        labels = labels.to(device)

        nfeats = graphs.ndata.pop('feat')
        efeats = graphs.edata.pop('feat')
        logits = model(graphs, nfeats, efeats)
        loss = loss_fn(logits, labels, model, args)
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())
        
        total = total + labels.shape[0]
        total_loss = total_loss + loss.item() * labels.shape[0]
    
    logits_list = torch.cat(logits_list, dim=0).numpy()
    labels_list = torch.cat(labels_list, dim=0).numpy()

    return total_loss / total, eval_fn(logits_list, labels_list, evaluator, args.eval_metric)


def run(model, train_loader, val_loader, test_loader, device, evaluator, args, iter):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    best_val_loss = 1e10
    
    for epoch in range(args.epochs):
        warmup_lr(optimizer, args.lr, epoch + 1, 10)
        loss = train(model, train_loader, device, optimizer, args)
        loss, auc = evaluate(model, train_loader, device, evaluator, args)
        val_loss, val_auc = evaluate(model, val_loader, device, evaluator, args)
        test_loss, test_auc = evaluate(model, test_loader, device, evaluator, args)
        scheduler.step(loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            result = {
                'val_loss': val_loss,
                'val_auc': val_auc,
                'test_loss': test_loss,
                'test_auc': test_auc,
            }

        if (epoch + 1) == args.epochs or (epoch + 1) % args.log_every == 0:
            print(f'Epoch {epoch+1:04d} | loss: {loss:.4f} | auc: {auc:.4f} | '
                  f'val_loss: {val_loss:.4f} | val_auc: {val_auc:.4f} | '
                  f'test_loss: {test_loss:.4f} | test_auc: {test_auc:.4f}')

    return result


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'SIR-GCN/GIN implementation on ogbg-molhiv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    argparser.add_argument('--seed', type=int, default=0, help='seed')

    argparser.add_argument('--model', type=str, default='SIR', help='model name', choices=['SIR', 'GIN']) 
    argparser.add_argument('--nhidden', type=int, default=256, help='number of hidden units')
    argparser.add_argument('--nlayers', type=int, default=1, help='number of graph convolution layers')
    argparser.add_argument('--input-dropout', type=float, default=0, help='input dropout rate')
    argparser.add_argument('--edge-dropout', type=float, default=0, help='edge dropout rate')
    argparser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    argparser.add_argument('--norm', type=str, default='none', help='type of normalization', choices=['gn', 'cn', 'bn', 'ln', 'none'])
    argparser.add_argument('--readout-layers', type=int, default=1, help='number of MLP layers for graph readout')
    argparser.add_argument('--readout-dropout', type=float, default=0, help='dropout rate for graph readout')
    argparser.add_argument('--readout-pooling', type=str, default='sum', help='type of graph readout pooling', choices=['sum', 'mean'])
    argparser.add_argument('--jumping-knowledge', action='store_true', help='use jumping knowledge for graph readout')
    argparser.add_argument('--virtual-node', action='store_true', help='add a virtual node')
    argparser.add_argument('--vn-layers', type=int, default=0, help='number of MLP layers for virtual node')
    argparser.add_argument('--vn-dropout', type=float, default=0, help='dropout rate for virtual node')
    argparser.add_argument('--vn-residual', action='store_true', help='add residual connections for virtual node')
    argparser.add_argument('--rand-feat', action='store_true', help='add a random feature to node features')
    argparser.add_argument('--centrality-encoder', action='store_true', help='use centrality encoder')
    argparser.add_argument('--residual', action='store_true', help='add residual connections')
    argparser.add_argument('--resid-layers', type=int, default=0, help='number of MLP layers for residual connections')
    argparser.add_argument('--resid-dropout', type=float, default=0, help='dropout rate for residual connections')
    argparser.add_argument('--feat-dropout', type=float, default=0, help='dropout rate for SIR inner linear transformations')
    argparser.add_argument('--agg-type', type=str, default='sum', help='aggregation type', choices=['sum', 'max', 'mean', 'sym'])
    argparser.add_argument('--nlayers-mlp', type=int, default=2, help='number of MLP layers for GIN')
    
    argparser.add_argument('--add-self-loop', action='store_true', help='add self-loop to graphs')
    
    argparser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    argparser.add_argument('--batch-size', type=int, default=64, help='batch size')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--wd', type=float, default=0, help='weight decay')
    argparser.add_argument('--l1', type=float, default=0, help='weight for L1 regularization')
    argparser.add_argument('--l2', type=float, default=0, help='weight for L2 regularization')
    argparser.add_argument('--factor', type=float, default=0.5, help='factor for learning rate decay')
    argparser.add_argument('--patience', type=int, default=10, help='patience for learning rate decay')
    
    argparser.add_argument('--flag', action='store_true', help='use FLAG for training')
    argparser.add_argument('--m', type=int, default=5, help='mini batch size for FLAG')
    argparser.add_argument('--step-size', type=float, default=1e-3, help='step size for FLAG')
    
    argparser.add_argument('--nruns', type=int, default=10, help='number of runs')
    argparser.add_argument('--log-every', type=int, default=20, help='log every LOG_EVERY epochs')
    args = argparser.parse_args()

    if args.model == 'GIN' and args.agg_type == 'sym':
        raise ValueError('GIN cannot use agg_type == sym')
    
    val_aucs, test_aucs = [], []
    for i in range(args.nruns):
        # Set seed
        set_seed(args.seed + i)

        # Load dataset
        device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
        dataset, loaders, evaluator = load_dataset('ogbg-molhiv', args)
        train_loader, val_loader, test_loader = loaders
        args.eval_metric = dataset.eval_metric
        
        # Load model
        Model = {'SIR': SIRModel, 'GIN': GINModel}
        max_degree = dataset.max_degree if args.centrality_encoder else 0
        model = Model[args.model](args.nhidden, dataset.num_tasks, args.nlayers, args.input_dropout, args.edge_dropout, args.dropout, args.norm, 
                                  args.readout_layers, args.readout_dropout, args.readout_pooling, args.jumping_knowledge,
                                  args.virtual_node, args.vn_layers, args.vn_dropout, args.vn_residual, args.rand_feat, max_degree, 
                                  args.residual, args.resid_layers, args.resid_dropout, feat_dropout=args.feat_dropout, agg_type=args.agg_type, 
                                  mlp_layers=args.nlayers_mlp).to(device)
        summary(model)

        # Training
        result = run(model, train_loader, val_loader, test_loader, device, evaluator, args, i)
        val_aucs.append(result['val_auc'])
        test_aucs.append(result['test_auc'])
    
    print(args)
    print(f'Runned {args.nruns} times')
    print(f'Val ROC-AUC: {val_aucs}')
    print(f'Test ROC-AUC: {test_aucs}')
    print(f'Average val ROC-AUC: {np.mean(val_aucs):.6f} ± {np.std(val_aucs):.6f}')
    print(f'Average test ROC-AUC: {np.mean(test_aucs):.6f} ± {np.std(test_aucs):.6f}')

# SIR-GCN (100k)
# Namespace(cpu=False, gpu=0, seed=0, model='SIR', nhidden=80, nlayers=4, input_dropout=0.2, edge_dropout=0, dropout=0, norm='bn', readout_layers=1, readout_dropout=0, readout_pooling='mean', jumping_knowledge=False, virtual_node=False, vn_layers=0, vn_dropout=0, vn_residual=False, rand_feat=False, centrality_encoder=False, residual=True, resid_layers=0, resid_dropout=0, feat_dropout=0.2, agg_type='max', nlayers_mlp=2, add_self_loop=False, epochs=100, batch_size=64, lr=0.001, wd=0.0001, l1=0, l2=0, factor=0.5, patience=10, flag=False, m=5, step_size=0.001, nruns=10, log_every=20, eval_metric='rocauc')
# Runned 10 times
# Val ROC-AUC: [0.838615152851264, 0.8255223642955123, 0.8070528365667256, 0.8289119145600626, 0.8288506760728983, 0.8175582990397805, 0.8139146090534979, 0.8125153096217912, 0.8317595042132078, 0.7856132422104645]
# Test ROC-AUC: [0.7810521640047122, 0.7676374592016069, 0.7727263948705074, 0.7627493771606251, 0.7736630680391665, 0.7808648293709805, 0.7730875451437842, 0.7826358176094556, 0.794580814615964, 0.7740879507136098]
# Average val ROC-AUC: 0.819031 ± 0.014535
# Average test ROC-AUC: 0.776309 ± 0.008434