import os
import dgl
import torch
import random
import argparse
import numpy as np
from torch import nn
from torchinfo import summary
from sklearn.metrics import roc_auc_score
from dgl.data import RomanEmpireDataset, AmazonRatingsDataset, MinesweeperDataset, TolokersDataset, QuestionsDataset

from model import SIRModel

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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


def load_dataset(dataset, device, args, iter):
    dataset = {'roman-empire': RomanEmpireDataset, 'amazon-ratings': AmazonRatingsDataset,
               'minesweeper': MinesweeperDataset, 'tolokers': TolokersDataset, 'questions': QuestionsDataset}[dataset]()
    graph = dataset[0]
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph)) if args.add_self_loop else graph
    graph = graph.to(device)
    labels = graph.ndata['label'].to(device)
    labels = labels.float() if dataset.num_classes == 2 else labels
    train_idx = graph.ndata['train_mask'][:, iter].to(device)
    val_idx = graph.ndata['val_mask'][:, iter].to(device)
    test_idx = graph.ndata['test_mask'][:, iter].to(device)
    dataset.input_dim = graph.ndata['feat'].shape[1]
    dataset.output_dim = 1 if dataset.num_classes == 2 else dataset.num_classes
    
    acc_fn = lambda logits, labels: torch.mean(logits.argmax(dim=-1) == labels, dtype=torch.float32).item()
    roc_auc = lambda logits, labels: roc_auc_score(labels.cpu().numpy(), logits.cpu().numpy()).item()
    
    args.loss_fn = nn.BCEWithLogitsLoss() if dataset.output_dim == 1 else nn.CrossEntropyLoss()
    args.metric = roc_auc if dataset.output_dim == 1 else acc_fn
    args.metric_name = 'auc' if dataset.output_dim == 1 else 'accuracy'
    
    return dataset, graph, labels, (train_idx.to(torch.bool), val_idx.to(torch.bool), test_idx.to(torch.bool))
    

def regularizer(model, args):
    l1 = torch.sum(torch.stack([torch.sum(torch.abs(param)) for param in model.parameters()]))
    l2 = torch.sum(torch.stack([torch.sum(torch.pow(param, 2)) for param in model.parameters()]))
    return (args.l1 * l1 + args.l2 * l2) * int(model.training)


def train(model, graph, labels, masks, args, optimizer, scaler):
    model.train()
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    train_idx, _, _ = masks
    feats = graph.ndata['feat']
    
    with torch.amp.autocast(device_type=graph.device.type, enabled=args.use_amp):
        logits = model(graph, feats)
        loss = args.loss_fn(logits[train_idx], labels[train_idx]) + regularizer(model, args)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item() - regularizer(model, args).item()

@torch.no_grad()
def evaluate(model, graph, labels, masks, args):
    model.eval()

    train_idx, val_idx, test_idx = masks
    feats = graph.ndata['feat']
    
    with torch.amp.autocast(device_type=graph.device.type, enabled=args.use_amp):
        logits = model(graph, feats)
        
        return {
            'loss': args.loss_fn(logits[train_idx], labels[train_idx]).item(),
            f'{args.metric_name}': args.metric(logits[train_idx], labels[train_idx]),
            'val_loss': args.loss_fn(logits[val_idx], labels[val_idx]).item(),
            f'val_{args.metric_name}': args.metric(logits[val_idx], labels[val_idx]),
            'test_loss': args.loss_fn(logits[test_idx], labels[test_idx]).item(),
            f'test_{args.metric_name}': args.metric(logits[test_idx], labels[test_idx]),
        }


def run(model, graph, labels, masks, args, iter):
    scaler = torch.amp.GradScaler(device=graph.device.type, enabled=args.use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    best_val_loss = 1e10

    for epoch in range(args.epochs):
        warmup_lr(optimizer, args.lr, epoch + 1, 10)
        loss = train(model, graph, labels, masks, args, optimizer, scaler)
        metrics = evaluate(model, graph, labels, masks, args)
        scheduler.step(loss)

        if metrics['val_loss'] < best_val_loss:
            best_val_loss = metrics['val_loss']
            result = metrics
            
        if (epoch + 1) == args.epochs or (epoch + 1) % args.log_every == 0:
            print(f'Epoch {epoch+1:04d} | ' + ' | '.join([f'{metric}: {value:.4f}' for metric, value in metrics.items()]))
            
    return result


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'SIR-GCN implementation on HeterophilousGraphs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    argparser.add_argument('--seed', type=int, default=0, help='seed')
    argparser.add_argument('--use-amp', action='store_true', help='use automatic mixed precision')

    argparser.add_argument('--model', type=str, default='SIR', help='model name', choices=['SIR']) 
    argparser.add_argument('--nhidden', type=int, default=512, help='number of hidden units')
    argparser.add_argument('--nlayers', type=int, default=5, help='number of graph convolution layers')
    argparser.add_argument('--input-dropout', type=float, default=0, help='input dropout rate')
    argparser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    argparser.add_argument('--norm', type=str, default='none', help='type of normalization', choices=['bn', 'ln', 'none'])
    argparser.add_argument('--residual', action='store_true', help='add residual connections')
    argparser.add_argument('--feat-dropout', type=float, default=0, help='dropout rate for SIR inner linear transformations')
    argparser.add_argument('--agg-type', type=str, default='mean', help='aggregation type for SIR', choices=['sum', 'max', 'mean', 'sym'])
    
    argparser.add_argument('--dataset', type=str, default='roman-empire', help='dataset name', choices=['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'])
    argparser.add_argument('--add-self-loop', action='store_true', help='add self-loop to graph')

    argparser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    argparser.add_argument('--wd', type=float, default=0, help='weight decay')
    argparser.add_argument('--l1', type=float, default=0, help='weight for L1 regularization')
    argparser.add_argument('--l2', type=float, default=0, help='weight for L2 regularization')
    argparser.add_argument('--factor', type=float, default=0.5, help='factor for learning rate decay')
    argparser.add_argument('--patience', type=int, default=50, help='patience for learning rate decay')
    
    argparser.add_argument('--nruns', type=int, default=1, help='number of runs')
    argparser.add_argument('--log-every', type=int, default=100, help='log every LOG_EVERY epochs')
    args = argparser.parse_args()

    val_metrics, test_metrics = [], []
    for i in range(args.nruns):
        for idx in range(10):
            # Set seed
            set_seed(args.seed + i)

            # Load dataset
            device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
            dataset, graph, labels, masks = load_dataset(args.dataset, device, args, idx)
            
            # Load model
            Model = {'SIR': SIRModel}
            model = Model[args.model](dataset.input_dim, args.nhidden, dataset.output_dim, args.nlayers, args.input_dropout, args.dropout, args.norm,
                                      args.residual, feat_dropout=args.feat_dropout, agg_type=args.agg_type).to(device)
            summary(model)

            # Training
            result = run(model, graph, labels, masks, args, i)
            val_metrics.append(result[f'val_{args.metric_name}'])
            test_metrics.append(result[f'test_{args.metric_name}'])
    
    print(args)
    print(f'Runned {args.nruns} times')
    print(f'Val {args.metric_name}: {val_metrics}')
    print(f'Test {args.metric_name}: {test_metrics}')
    print(f'Average val {args.metric_name}: {np.mean(val_metrics):.6f} ± {np.std(val_metrics):.6f}')
    print(f'Average test {args.metric_name}: {np.mean(test_metrics):.6f} ± {np.std(test_metrics):.6f}')

# SIR-GCN (roman-empire)
# Namespace(cpu=False, gpu=0, seed=0, use_amp=True, model='SIR', nhidden=512, nlayers=5, input_dropout=0.2, dropout=0.2, norm='ln', residual=True, feat_dropout=0.2, agg_type='max', dataset='roman-empire', add_self_loop=False, epochs=1000, lr=3e-05, wd=0, l1=0, l2=0, factor=0.5, patience=1000, nruns=1, log_every=100, loss_fn=CrossEntropyLoss(), metric=<function load_dataset.<locals>.<lambda> at 0x7ba7b41b1c60>, metric_name='accuracy')
# Runned 10 times
# Val accuracy: [0.8789055347442627, 0.8808472752571106, 0.8681376576423645, 0.8810238242149353, 0.8831420540809631, 0.870432436466217, 0.8676080703735352, 0.8688437342643738, 0.880317747592926, 0.8762577176094055]
# Test accuracy: [0.876985490322113, 0.8792799115180969, 0.87239670753479, 0.8780444264411926, 0.8798093795776367, 0.8806918263435364, 0.8746911287307739, 0.8722202181816101, 0.8753970861434937, 0.8776914477348328]
# Average val accuracy: 0.875552 ± 0.005825
# Average test accuracy: 0.876721 ± 0.002819

# SIR-GCN (amazon-ratings)
# Namespace(cpu=False, gpu=0, seed=0, use_amp=True, model='SIR', nhidden=256, nlayers=3, input_dropout=0.2, dropout=0.2, norm='bn', residual=True, feat_dropout=0.2, agg_type='max', dataset='amazon-ratings', add_self_loop=False, epochs=1000, lr=3e-05, wd=0, l1=0, l2=0, factor=0.5, patience=1000, nruns=1, log_every=100, loss_fn=CrossEntropyLoss(), metric=<function load_dataset.<locals>.<lambda> at 0x7f1838fd40e0>, metric_name='accuracy')
# Runned 10 times
# Val accuracy: [0.4674179255962372, 0.4677445590496063, 0.4652947783470154, 0.4693777561187744, 0.4806467294692993, 0.47346070408821106, 0.47770699858665466, 0.457782119512558, 0.45582231879234314, 0.46431487798690796]
# Test accuracy: [0.4688878059387207, 0.47346070408821106, 0.4677445590496063, 0.46660134196281433, 0.47476726770401, 0.4719908535480499, 0.4732973873615265, 0.4571288526058197, 0.4608851969242096, 0.4585987329483032]
# Average val accuracy: 0.467957 ± 0.007482
# Average test accuracy: 0.467336 ± 0.006125

# SIR-GCN (minesweeper)
# Namespace(cpu=False, gpu=0, seed=0, use_amp=True, model='SIR', nhidden=256, nlayers=5, input_dropout=0.2, dropout=0.2, norm='bn', residual=True, feat_dropout=0.2, agg_type='sym', dataset='minesweeper', add_self_loop=False, epochs=1000, lr=3e-05, wd=0, l1=0, l2=0, factor=0.5, patience=1000, nruns=1, log_every=100, loss_fn=BCEWithLogitsLoss(), metric=<function load_dataset.<locals>.<lambda> at 0x7a03c1762de0>, metric_name='auc')
# Runned 10 times
# Val auc: [0.9283999999999999, 0.9450915, 0.9402619999999999, 0.9360255, 0.9434885, 0.9331245, 0.9461635, 0.9434469999999999, 0.9423875, 0.944088]
# Test auc: [0.9408155, 0.945147, 0.9454065, 0.9362545, 0.9396869999999999, 0.9383505, 0.9335964999999999, 0.946256, 0.9398575, 0.946338]
# Average val auc: 0.940248 ± 0.005545
# Average test auc: 0.941171 ± 0.004241

# SIR-GCN (tolokers)
# Namespace(cpu=False, gpu=0, seed=0, use_amp=True, model='SIR', nhidden=256, nlayers=5, input_dropout=0.2, dropout=0.2, norm='ln', residual=True, feat_dropout=0.2, agg_type='sym', dataset='tolokers', add_self_loop=False, epochs=1000, lr=3e-05, wd=0, l1=0, l2=0, factor=0.5, patience=1000, nruns=1, log_every=100, loss_fn=BCEWithLogitsLoss(), metric=<function load_dataset.<locals>.<lambda> at 0x7609c4e83e20>, metric_name='auc')
# Runned 10 times
# Val auc: [0.843844406517775, 0.8351435624004595, 0.8248134102909808, 0.8147052513954345, 0.7968717965428801, 0.8365807478252132, 0.8361459262548048, 0.8381761119008728, 0.81965902656994, 0.8320251347912925]
# Test auc: [0.8316160741156471, 0.8397034940311093, 0.8390914217699802, 0.8206157189375023, 0.8237977490923978, 0.8223265388567602, 0.8288939454327073, 0.8206818064740029, 0.8226047843309501, 0.8357955177060372]
# Average val auc: 0.827797 ± 0.013367
# Average test auc: 0.828513 ± 0.007204

# SIR-GCN (questions)
# Namespace(cpu=False, gpu=0, seed=0, use_amp=True, model='SIR', nhidden=512, nlayers=3, input_dropout=0.2, dropout=0.2, norm='ln', residual=True, feat_dropout=0.2, agg_type='sym', dataset='questions', add_self_loop=False, epochs=1000, lr=3e-05, wd=0, l1=0, l2=0, factor=0.5, patience=1000, nruns=1, log_every=100, loss_fn=BCEWithLogitsLoss(), metric=<function load_dataset.<locals>.<lambda> at 0x717fa0229620>, metric_name='auc')
# Runned 10 times
# Val auc: [0.7362157606405393, 0.7737087900986555, 0.7535020117878646, 0.73655484936125, 0.7421078456840368, 0.7471873647022149, 0.7694029290707676, 0.7684462763163211, 0.755138342887161, 0.7574334551374193]
# Test auc: [0.7446740889706748, 0.7758145178234579, 0.7462847689611622, 0.7350374847902029, 0.7605344151241374, 0.7413784982533266, 0.7396048107982055, 0.7730803331263031, 0.7571000602619664, 0.7598433650651453]
# Average val auc: 0.753970 ± 0.012869
# Average test auc: 0.753335 ± 0.013396
