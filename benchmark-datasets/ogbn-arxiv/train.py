import os
import dgl
import torch
import random
import argparse
import numpy as np
from torchinfo import summary
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

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


def load_dataset(name, device, args):
    dataset = DglNodePropPredDataset(name=name)
    evaluator = Evaluator(name=name)

    train_idx = dataset.get_idx_split()['train']
    val_idx = dataset.get_idx_split()['valid']
    test_idx = dataset.get_idx_split()['test']
    graph, labels = dataset[0]

    if args.add_reverse_edge:
        graph = dgl.to_bidirected(graph, copy_ndata=True)
    else:
        # Edge is from reference paper to new paper
        graph = dgl.reverse(graph, copy_ndata=True)

    if args.use_xrt_emb:
        xrt_emb = torch.from_numpy(np.load(f'dataset/{name.replace("-", "_")}_xrt/X.all.xrt-emb.npy'))
        graph.ndata['feat'] = xrt_emb

    if args.add_self_loop:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        graph.create_formats_()

    graph = graph.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    return graph, labels, (train_idx, val_idx, test_idx), evaluator


def regularizer(model, args):
    l1 = torch.sum(torch.stack([torch.sum(torch.abs(param)) for param in model.parameters()]))
    l2 = torch.sum(torch.stack([torch.sum(torch.pow(param, 2)) for param in model.parameters()]))
    return (args.l1 * l1 + args.l2 * l2) * int(model.training)

def loss_fn(logits, labels, model, args):
    eps = 1 - np.log(2)
    loss = F.cross_entropy(logits, labels.squeeze(), reduction='none')
    loss = torch.log(loss + eps) - np.log(eps)
    return torch.mean(loss) + regularizer(model, args)

# def consistency_loss_fn(prob_list, args):
#     prob_list = torch.stack(prob_list, dim=2)
#     avg_prob = torch.mean(prob_list, dim=2)
#     sharp_prob = torch.pow(avg_prob, 1 / args.consis_temp) / torch.sum(torch.pow(avg_prob, 1 / args.consis_temp), dim=1, keepdim=True)
#     sharp_prob = sharp_prob.detach().unsqueeze(dim=2)
#     loss = torch.mean(torch.sum(torch.pow(prob_list - sharp_prob, 2)[avg_prob.max(dim=1)[0] > args.consis_lb], dim=1))
#     return loss

def kd_loss_fn(student_logits, teacher_logits, model, args):
    return (args.kd_temp ** 2) * F.kl_div(F.log_softmax(student_logits / args.kd_temp, dim=1), 
                                          F.softmax(teacher_logits / args.kd_temp, dim=1), reduction='batchmean') + regularizer(model, args)

def acc_fn(logits, labels, evaluator):
    return evaluator.eval({'y_pred': logits.argmax(dim=-1, keepdim=True), 'y_true': labels})['acc']


def add_labels(feats, labels, mask, device):
    one_hot = torch.zeros((feats.shape[0], torch.unique(labels).shape[0]), device=device)
    one_hot[mask, labels[mask, 0]] = 1.0
    return torch.cat((feats, one_hot), dim=-1)


def train(model, graph, labels, masks, device, args, optimizer, iter):
    model.train()
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    train_idx, val_idx, test_idx = masks
    output_dim = torch.unique(labels).shape[0]
    feats = graph.ndata['feat']
    mask = torch.rand(train_idx.shape) < args.mask_rate
    train_idx_ = train_idx[mask]

    m, perturb = 1, 0
    if args.flag:
        m = args.m + 1
        perturb = torch.zeros(feats.shape, device=device)
        perturb.uniform_(-args.untrain_step_size, args.untrain_step_size)
        perturb[train_idx] = args.train_step_size / args.untrain_step_size * perturb[train_idx]
        perturb.requires_grad_()
    
    total_loss = 0
    for _ in range(m):
        if args.use_labels:
            feats = add_labels(feats, labels, train_idx[~mask], device)
            perturb = torch.cat((perturb, torch.zeros(feats.shape[0], output_dim)), dim=-1) if args.flag else perturb
        
        logits = model(graph, feats, perturb)

        unlabel_idx = torch.cat((train_idx_, val_idx, test_idx))
        for _ in range(args.label_iters * int(args.use_labels)):
            logits = logits.detach()
            torch.cuda.empty_cache()
            feats[unlabel_idx, -output_dim:] = F.softmax(logits[unlabel_idx], dim=-1)
            logits = model(graph, feats, perturb)

        loss = loss_fn(logits[train_idx_], labels[train_idx_], model, args) / m
        if args.kd_mode == 'student':
            teacher_logits = torch.load(f'./output/teacher_{iter}.pt', map_location=device)
            loss = loss * (1 - args.kd_alpha) + kd_loss_fn(logits, teacher_logits, model, args) / m * args.kd_alpha
        total_loss = total_loss + loss.item()
        loss.backward()

        if args.flag:
            perturb_data = perturb[train_idx].detach() + args.train_step_size * torch.sign(perturb.grad[train_idx].detach())
            perturb.data[train_idx] = perturb_data.data
            perturb_data = perturb[~train_idx].detach() + args.untrain_step_size * torch.sign(perturb.grad[~train_idx].detach())
            perturb.data[~train_idx] = perturb_data.data
            perturb.grad[:] = 0

    optimizer.step()

    return total_loss - regularizer(model, args).item()

@torch.no_grad()
def evaluate(model, graph, labels, masks, device, args, evaluator):
    model.eval()
    
    train_idx, val_idx, test_idx = masks
    output_dim = torch.unique(labels).shape[0]
    feats = graph.ndata['feat']
    
    if args.use_labels:
        feats = add_labels(feats, labels, train_idx, device)
    
    logits = model(graph, feats)
    
    unlabel_idx = torch.cat((val_idx, test_idx))
    for _ in range(args.label_iters * int(args.use_labels)):
        feats[unlabel_idx, -output_dim:] = F.softmax(logits[unlabel_idx], dim=-1)
        logits = model(graph, feats)

    loss = loss_fn(logits[train_idx], labels[train_idx], model, args).item()
    acc = acc_fn(logits[train_idx], labels[train_idx], evaluator)
    val_loss = loss_fn(logits[val_idx], labels[val_idx], model, args).item()
    val_acc = acc_fn(logits[val_idx], labels[val_idx], evaluator)
    test_loss = loss_fn(logits[test_idx], labels[test_idx], model, args).item()
    test_acc = acc_fn(logits[test_idx], labels[test_idx], evaluator)
    return (loss, acc, val_loss, val_acc, test_loss, test_acc), logits.detach().cpu()


def run(model, graph, labels, masks, device, evaluator, args, iter):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    best_val_loss = 1e10
    
    for epoch in range(args.epochs):
        warmup_lr(optimizer, args.lr, epoch + 1, 20)
        loss = train(model, graph, labels, masks, device, args, optimizer, iter)
        metrics, logits = evaluate(model, graph, labels, masks, device, args, evaluator)
        loss, acc, val_loss, val_acc, test_loss, test_acc = metrics
        scheduler.step(loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            result = {
                'val_loss': val_loss,
                'val_acc': val_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'logits': logits,
            }
            
        if (epoch + 1) == args.epochs or (epoch + 1) % args.log_every == 0:
            print(f'Epoch {epoch+1:04d} | loss: {loss:.4f} | acc: {acc:.4f} | '
                  f'val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | '
                  f'test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}')
        
    # Save predictions
    if args.save_pred:
        os.makedirs('./output', exist_ok=True)
        torch.save(F.softmax(result['logits'], dim=1), f'./output/{args.kd_mode}_{iter}.pt')

    return result


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'SIR-GCN/GATv2 implementation on ogbn-arxiv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    argparser.add_argument('--seed', type=int, default=0, help='seed')

    argparser.add_argument('--model', type=str, default='SIR', help='model name', choices=['SIR', 'GAT']) 
    argparser.add_argument('--nhidden', type=int, default=256, help='number of hidden units')
    argparser.add_argument('--nlayers', type=int, default=1, help='number of graph convolution layers')
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
    
    argparser.add_argument('--add-self-loop', action='store_true', help='add self-loop to graph')
    argparser.add_argument('--add-reverse-edge', action='store_true', help='add reverse edge to graph')
    argparser.add_argument('--use-xrt-emb', action='store_true', help='use GIANT-XRT embeddings')
    argparser.add_argument('--use-labels', action='store_true', help='use label trick')
    argparser.add_argument('--label-iters', type=int, default=0, help='number of iterations for label reuse')
    argparser.add_argument('--mask-rate', type=float, default=1, help='mask rate for training')

    argparser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    argparser.add_argument('--wd', type=float, default=0, help='weight decay')
    argparser.add_argument('--l1', type=float, default=0, help='weight for L1 regularization')
    argparser.add_argument('--l2', type=float, default=0, help='weight for L2 regularization')
    argparser.add_argument('--factor', type=float, default=0.5, help='factor for learning rate decay')
    argparser.add_argument('--patience', type=int, default=10, help='patience for learning rate decay')
    
    argparser.add_argument('--kd-mode', type=str, default='teacher', help='training mode', choices=['teacher', 'student']) 
    argparser.add_argument('--kd-alpha', type=float, default=0.5, help='ratio of knowledge distillation loss')
    argparser.add_argument('--kd-temp', type=float, default=1, help='temperature of knowledge distillation')

    argparser.add_argument('--flag', action='store_true', help='use FLAG for training')
    argparser.add_argument('--m', type=int, default=5, help='mini batch size for FLAG')
    argparser.add_argument('--train-step-size', type=float, default=1e-5, help='step size for training nodes for FLAG')
    argparser.add_argument('--untrain-step-size', type=float, default=1e-5, help='step size for non training nodes for FLAG')

    argparser.add_argument('--nruns', type=int, default=10, help='number of runs')
    argparser.add_argument('--log-every', type=int, default=20, help='log every LOG_EVERY epochs')
    argparser.add_argument('--save-pred', action='store_true', help='save final predictions')
    args = argparser.parse_args()

    val_accs, test_accs = [], []
    for i in range(args.nruns):
        # Set seed
        set_seed(args.seed + i)

        # Load dataset
        device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
        graph, labels, masks, evaluator = load_dataset('ogbn-arxiv', device, args)
        input_dim = graph.ndata['feat'].shape[1]
        output_dim = torch.unique(labels).shape[0]

        # Load model
        Model = {'SIR': SIRModel, 'GAT': GATModel}
        _input_dim = input_dim + output_dim if args.use_labels else input_dim
        model = Model[args.model](_input_dim, args.nhidden, output_dim, args.nlayers, args.input_dropout, args.edge_dropout, args.dropout, args.norm,
                                  args.readout_layers, args.readout_dropout, args.jumping_knowledge, args.residual,
                                  resid_layers=args.resid_layers, resid_dropout=args.resid_dropout, 
                                  feat_dropout=args.feat_dropout, agg_type=args.agg_type, 
                                  num_heads=args.nheads, attn_dropout=args.attn_dropout).to(device)
        summary(model)

        # Training
        result = run(model, graph, labels, masks, device, evaluator, args, i)
        val_accs.append(result['val_acc'])
        test_accs.append(result['test_acc'])
    
    print(args)
    print(f'Runned {args.nruns} times')
    print(f'Val accuracy: {val_accs}')
    print(f'Test accuracy: {test_accs}')
    print(f'Average val accuracy: {np.mean(val_accs):.6f} ± {np.std(val_accs):.6f}')
    print(f'Average test accuracy: {np.mean(test_accs):.6f} ± {np.std(test_accs):.6f}')

# SIR-GCN
# Namespace(cpu=False, gpu=0, seed=0, model='SIR', nhidden=95, nlayers=3, input_dropout=0, edge_dropout=0, dropout=0.2, norm='bn', readout_layers=1, readout_dropout=0, jumping_knowledge=False, residual=True, resid_layers=0, resid_dropout=0, feat_dropout=0.2, agg_type='sym', nheads=1, attn_dropout=0, add_self_loop=True, add_reverse_edge=True, use_xrt_emb=False, use_labels=False, label_iters=0, mask_rate=1, epochs=1000, lr=0.01, wd=0.001, l1=0, l2=0, factor=0.5, patience=40, kd_mode='teacher', kd_alpha=0.5, kd_temp=1, flag=False, m=5, train_step_size=1e-05, untrain_step_size=1e-05, nruns=10, log_every=20, save_pred=False)
# Runned 10 times
# Val accuracy: [0.7376422027584818, 0.7343199436222692, 0.7343199436222692, 0.7325413604483372, 0.7333131984294775, 0.7382126917010637, 0.7349911070841303, 0.734823316218665, 0.7346555253531998, 0.7359307359307359]
# Test accuracy: [0.7274653827953007, 0.7272390593173261, 0.7257576692796741, 0.7233092607452215, 0.724235129518754, 0.7244408781350945, 0.7244820278583627, 0.7222187930786166, 0.7259839927576487, 0.7264160648519639]
# Average val accuracy: 0.735075 ± 0.001676
# Average test accuracy: 0.725155 ± 0.001617
