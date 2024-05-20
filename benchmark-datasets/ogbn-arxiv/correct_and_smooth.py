import dgl
import glob
import torch
import argparse
import numpy as np
from functools import partial
from dgl import function as fn
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


def load_dataset(name, device, args):
    dataset = DglNodePropPredDataset(name=name)
    evaluator = Evaluator(name=name)

    train_idx = dataset.get_idx_split()['train']
    val_idx = dataset.get_idx_split()['valid']
    test_idx = dataset.get_idx_split()['test']
    graph, labels = dataset[0]

    if args.add_reverse_edge:
        graph = dgl.to_bidirected(graph, copy_ndata=False)
    else:
        # Edge is from reference paper to new paper
        graph = dgl.reverse(graph, copy_ndata=False)

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


def label_spreading(graph, y0, nprop=10, alpha=0.1, use_sym=True, post_step=None):
    with graph.local_scope():
        y = y0
        degs = graph.in_degrees().float().clamp(min=1).to(graph.device)
        norm = torch.pow(degs, -0.5).unsqueeze(dim=1) if use_sym else 1
        agg_fn = fn.sum if use_sym else fn.mean

        for _ in range(nprop):
            graph.ndata.update({'y': y * norm})
            graph.update_all(fn.copy_u('y', 'm'), agg_fn('m', 'y'))
            y = graph.ndata['y'] * norm

            y = alpha * y + (1 - alpha) * y0

            if post_step is not None:
                y = post_step(y)

        return y


def fix_input(x, y, mask):
    x[mask] = y[mask]
    return x

def acc_fn(predictions, labels, evaluator):
    return evaluator.eval({'y_pred': predictions.argmax(dim=-1, keepdim=True), 'y_true': labels})['acc']


def evaluate(predictions, labels, masks, evaluator):
    train_idx, val_idx, test_idx = masks
    train_acc = acc_fn(predictions[train_idx], labels[train_idx], evaluator)
    val_acc = acc_fn(predictions[val_idx], labels[val_idx], evaluator)
    test_acc = acc_fn(predictions[test_idx], labels[test_idx], evaluator)
    return train_acc, val_acc, test_acc

def run(graph, predictions, labels, masks, device, evaluator, args, pred_file):
    train_idx, _, _ = masks
    nclasses = torch.unique(labels).shape[0]

    y = predictions.clone()
    orig_train_acc, orig_val_acc, orig_test_acc = evaluate(y, labels, masks, evaluator)

    print(f'Original val_acc: {orig_val_acc:.4f}')
    print(f'Original test_acc: {orig_test_acc:.4f}')
    
    # Correct step
    dy = torch.zeros(graph.number_of_nodes(), nclasses, device=device)
    dy[train_idx] = F.one_hot(labels[train_idx], nclasses).float().squeeze(1) - y[train_idx]
    smoothed_dy = label_spreading(graph, dy, nprop=args.nprop_c, alpha=args.alpha_c, 
                                    use_sym=args.use_sym, post_step=partial(fix_input, y=dy, mask=train_idx))
    y = y + args.alpha_c * smoothed_dy
    
    # Smooth step
    y[train_idx] = F.one_hot(labels[train_idx], nclasses).float().squeeze(1)
    smoothed_y = label_spreading(graph, y, nprop=args.nprop_s, alpha=args.alpha_s, 
                                 use_sym=args.use_sym, post_step=lambda x: x.clamp(0, 1))
    train_acc, val_acc, test_acc = evaluate(smoothed_y, labels, masks, evaluator)

    print(f'New val_acc: {val_acc:.4f}')
    print(f'New test_acc: {test_acc:.4f}')

    # Save predictions
    if args.save_pred:
        torch.save(smoothed_y, pred_file.replace('_', '_cs_'))

    return {
        'orig_val_acc': orig_val_acc,
        'orig_test_acc': orig_test_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'Correct & Smooth implementation on ogbn-arxiv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    argparser.add_argument('--add-self-loop', action='store_true', help='add self-loop to graph')
    argparser.add_argument('--add-reverse-edge', action='store_true', help='add reverse edge to graph')
    
    argparser.add_argument('--use-sym', action='store_true', help='use symmetric normalized adjacency matrix')
    argparser.add_argument('--alpha-c', type=float, default=0.8, help='correct step alpha')
    argparser.add_argument('--nprop-c', type=int, default=10, help='correct step number of iterations')
    argparser.add_argument('--alpha-s', type=float, default=0.8, help='smooth step alpha')
    argparser.add_argument('--nprop-s', type=int, default=10, help='smooth step number of iterations')
    
    argparser.add_argument('--pred-files', type=str, default='./output/*.pt', help='path to prediction files')
    argparser.add_argument('--save-pred', action='store_true', help='save smoothed predictions')
    args = argparser.parse_args()

    orig_val_accs, orig_test_accs, val_accs, test_accs = [], [], [], []
    for pred_file in glob.iglob(args.pred_files):
        print(f'loading: {pred_file}')

        # Load dataset
        device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
        graph, labels, masks, evaluator = load_dataset('ogbn-arxiv', device, args)

        # Load predictions
        predictions = torch.load(pred_file, map_location=device)
        if predictions.max() > 1 or predictions.min() < 0:
            print('Not standard probability')
            predictions = F.softmax(predictions, dim=-1)

        # Training
        result = run(graph, predictions, labels, masks, device, evaluator, args, pred_file)
        orig_val_accs.append(result['orig_val_acc'])
        orig_test_accs.append(result['orig_test_acc'])
        val_accs.append(result['val_acc'])
        test_accs.append(result['test_acc'])
    
    print(args)
    print(f'Runned {len(val_accs)} times')
    print(f'Val accuracy: {val_accs}')
    print(f'Test accuracy:', test_accs)
    print(f'Average original val accuracy: {np.mean(orig_val_accs):.6f} ± {np.std(orig_val_accs):.6f}')
    print(f'Average original test accuracy: {np.mean(orig_test_accs):.6f} ± {np.std(orig_test_accs):.6f}')
    print(f'Average val accuracy: {np.mean(val_accs):.6f} ± {np.std(val_accs):.6f}')
    print(f'Average test accuracy: {np.mean(test_accs):.6f} ± {np.std(test_accs):.6f}')

# GIANT-XRT + SIR-GCN + BoT + C&S
# Namespace(cpu=False, gpu=0, add_self_loop=True, add_reverse_edge=True, use_sym=True, alpha_c=0.8, nprop_c=10, alpha_s=0.8, nprop_s=10, pred_files='./output/*.pt', save_pred=False)
# Runned 10 times
# Val accuracy: [0.7684150474848149, 0.7705627705627706, 0.7692875599852345, 0.7678110003691399, 0.7690862109466761, 0.7687506292157454, 0.7686163965233732, 0.7693882345045135, 0.7690862109466761, 0.7689184200812108]
# Test accuracy: [0.7543978766742794, 0.7610024072588112, 0.7574223813344855, 0.7585128490010905, 0.7572372075797791, 0.7580190523218732, 0.7556735180955908, 0.7591918194350143, 0.7577310042589963, 0.7543773018126453]
# Average original val accuracy: 0.766529 ± 0.000835
# Average original test accuracy: 0.755137 ± 0.002029
# Average val accuracy: 0.768992 ± 0.000683
# Average test accuracy: 0.757357 ± 0.001976
