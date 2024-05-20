import os
import dgl
import torch
import random
import argparse
import numpy as np
from torch import nn
from torchinfo import summary
from dgl.data import WikiCSDataset

from model import SIRModel, GATModel

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


def load_dataset(device, args, iter):
    dataset = WikiCSDataset()
    graph = dataset[0]
    graph = dgl.to_bidirected(graph, copy_ndata=True) if args.add_reverse_edge else graph
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph)) if args.add_self_loop else graph
    graph = graph.to(device)
    labels = graph.ndata['label'].to(device)
    train_idx = graph.ndata['train_mask'][:, iter].to(device)
    val_idx = (graph.ndata['val_mask'] + graph.ndata['stopping_mask'])[:, iter].to(device)
    test_idx = graph.ndata['test_mask'].to(device)
    dataset.input_dim = graph.ndata['feat'].shape[1]
    
    return dataset, graph, labels, (train_idx.to(torch.bool), val_idx.to(torch.bool), test_idx.to(torch.bool))
    

def regularizer(model, args):
    l1 = torch.sum(torch.stack([torch.sum(torch.abs(param)) for param in model.parameters()]))
    l2 = torch.sum(torch.stack([torch.sum(torch.pow(param, 2)) for param in model.parameters()]))
    return (args.l1 * l1 + args.l2 * l2) * int(model.training)


def train(model, graph, labels, masks, args, optimizer):
    model.train()
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    train_idx, val_idx, test_idx = masks
    feats = graph.ndata['feat']
    logits = model(graph, feats)
    loss = loss_fn(logits[train_idx], labels[train_idx]) + regularizer(model, args)

    loss.backward()
    optimizer.step()
    
    return loss.item() - regularizer(model, args).item()

@torch.no_grad()
def evaluate(model, graph, labels, masks, args):
    model.eval()

    train_idx, val_idx, test_idx = masks
    feats = graph.ndata['feat']
    logits = model(graph, feats)
    
    loss = loss_fn(logits[train_idx], labels[train_idx]).item()
    acc = acc_fn(logits[train_idx], labels[train_idx]).item()
    val_loss = loss_fn(logits[val_idx], labels[val_idx]).item()
    val_acc = acc_fn(logits[val_idx], labels[val_idx]).item()
    test_loss = loss_fn(logits[test_idx], labels[test_idx]).item()
    test_acc = acc_fn(logits[test_idx], labels[test_idx]).item()

    return loss, acc, val_loss, val_acc, test_loss, test_acc


def run(model, graph, labels, masks, args, iter):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    best_val_loss = 1e10

    for epoch in range(args.epochs):
        warmup_lr(optimizer, args.lr, epoch + 1, 10)
        loss = train(model, graph, labels, masks, args, optimizer)
        loss, acc, val_loss, val_acc, test_loss, test_acc = evaluate(model, graph, labels, masks, args)
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
        'SIR-GCN/GATv2 implementation on WikiCSDataset',
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
    
    argparser.add_argument('--add-self-loop', action='store_true', help='add self-loop to graph')
    argparser.add_argument('--add-reverse-edge', action='store_true', help='add reverse edge to graph')

    argparser.add_argument('--epochs', type=int, default=500, help='number of epochs')
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
        for idx in range(20):
            # Set seed
            set_seed(args.seed + i)

            # Load dataset
            device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
            dataset, graph, labels, masks = load_dataset(device, args, idx)

            # Load model
            Model = {'SIR': SIRModel, 'GAT': GATModel}
            model = Model[args.model](dataset.input_dim, args.nhidden, dataset.num_classes, args.nlayers, args.input_dropout, args.edge_dropout, args.dropout, args.norm,
                                      args.readout_layers, args.readout_dropout, args.jumping_knowledge, args.residual,
                                      resid_layers=args.resid_layers, resid_dropout=args.resid_dropout, 
                                      feat_dropout=args.feat_dropout, agg_type=args.agg_type, 
                                      num_heads=args.nheads, attn_dropout=args.attn_dropout).to(device)
            summary(model)

            # Training
            result = run(model, graph, labels, masks, args, i)
            val_accs.append(result['val_acc'])
            test_accs.append(result['test_acc'])
    
    print(args)
    print(f'Runned {args.nruns} times')
    print(f'Val accuracy: {val_accs}')
    print(f'Test accuracy: {test_accs}')
    print(f'Average val accuracy: {np.mean(val_accs):.6f} ± {np.std(val_accs):.6f}')
    print(f'Average test accuracy: {np.mean(test_accs):.6f} ± {np.std(test_accs):.6f}')

# SIR-GCN
# Namespace(cpu=False, gpu=0, seed=0, model='SIR', nhidden=60, nlayers=4, input_dropout=0.3, edge_dropout=0.0, dropout=0.1, norm='bn', readout_layers=1, readout_dropout=0.0, jumping_knowledge=False, residual=True, resid_layers=1, resid_dropout=0.2, feat_dropout=0.0, agg_type='mean', nheads=1, attn_dropout=0, add_self_loop=False, add_reverse_edge=False, epochs=200, lr=0.001, wd=0, l1=1e-05, l2=1e-05, factor=0.5, patience=10, nruns=10, log_every=20)
# Runned 10 times
# Val accuracy: [0.7925673127174377, 0.7904816269874573, 0.806598424911499, 0.7944633960723877, 0.7923777103424072, 0.7954114675521851, 0.7872582674026489, 0.786120593547821, 0.7971179485321045, 0.801478922367096, 0.7956010699272156, 0.801478922367096, 0.790291965007782, 0.7919985055923462, 0.7952218651771545, 0.8026165962219238, 0.7874478697776794, 0.7988244295120239, 0.7957906723022461, 0.8005309104919434, 0.7954114675521851, 0.7895335555076599, 0.8052711486816406, 0.7963594794273376, 0.7878270745277405, 0.7986348271369934, 0.7937049865722656, 0.7821387648582458, 0.7952218651771545, 0.7948426008224487, 0.7887751460075378, 0.7918088436126709, 0.797686755657196, 0.7916192412376404, 0.7925673127174377, 0.7937049865722656, 0.7864997982978821, 0.7982555627822876, 0.7846037149429321, 0.8003413081169128, 0.7872582674026489, 0.7874478697776794, 0.8039438724517822, 0.7828972339630127, 0.7847933173179626, 0.7988244295120239, 0.78574138879776, 0.7844141125679016, 0.7827076315879822, 0.7849829196929932, 0.7882062792778015, 0.7823284268379211, 0.7906712293624878, 0.7825180292129517, 0.7893439531326294, 0.7910504341125488, 0.7921881079673767, 0.8007205128669739, 0.7923777103424072, 0.7893439531326294, 0.7923777103424072, 0.7874478697776794, 0.8037542700767517, 0.7948426008224487, 0.7739855647087097, 0.7944633960723877, 0.7847933173179626, 0.7952218651771545, 0.7882062792778015, 0.7961698770523071, 0.7954114675521851, 0.7893439531326294, 0.797307550907135, 0.7957906723022461, 0.7916192412376404, 0.8037542700767517, 0.7944633960723877, 0.8031854629516602, 0.797686755657196, 0.8033750653266907, 0.7974971532821655, 0.7863101959228516, 0.8069776296615601, 0.790291965007782, 0.7781569957733154, 0.8050815463066101, 0.776071310043335, 0.790291965007782, 0.7885854840278625, 0.7925673127174377, 0.7878270745277405, 0.7919985055923462, 0.7901023626327515, 0.7931361198425293, 0.7942737936973572, 0.8028061985969543, 0.7859309911727905, 0.7919985055923462, 0.7846037149429321, 0.790291965007782, 0.7944633960723877, 0.7887751460075378, 0.8043230772018433, 0.7864997982978821, 0.78574138879776, 0.7927569150924683, 0.7855517864227295, 0.7912400364875793, 0.7957906723022461, 0.796928346157074, 0.7891543507575989, 0.7859309911727905, 0.8005309104919434, 0.797686755657196, 0.7921881079673767, 0.7952218651771545, 0.7918088436126709, 0.7878270745277405, 0.7971179485321045, 0.7908608317375183, 0.7893439531326294, 0.7842245101928711, 0.8043230772018433, 0.7954114675521851, 0.7908608317375183, 0.7937049865722656, 0.788016676902771, 0.7846037149429321, 0.7874478697776794, 0.7952218651771545, 0.7885854840278625, 0.78763747215271, 0.797686755657196, 0.7984452247619629, 0.7929465174674988, 0.7929465174674988, 0.7866894006729126, 0.7946529984474182, 0.7971179485321045, 0.8020477890968323, 0.7929465174674988, 0.7840349078178406, 0.8005309104919434, 0.7777777910232544, 0.7789154052734375, 0.7885854840278625, 0.7827076315879822, 0.7834660410881042, 0.7933257222175598, 0.7923777103424072, 0.7863101959228516, 0.78574138879776, 0.7954114675521851, 0.7982555627822876, 0.7874478697776794, 0.797686755657196, 0.7897231578826904, 0.7938945889472961, 0.7931361198425293, 0.7971179485321045, 0.788395881652832, 0.7792946696281433, 0.7938945889472961, 0.7904816269874573, 0.7838452458381653, 0.7978763580322266, 0.776450514793396, 0.7864997982978821, 0.7916192412376404, 0.7912400364875793, 0.7849829196929932, 0.7847933173179626, 0.7904816269874573, 0.7927569150924683, 0.7914296388626099, 0.7895335555076599, 0.7849829196929932, 0.786120593547821, 0.7853621244430542, 0.7950322031974792, 0.7847933173179626, 0.7720894813537598, 0.8001517057418823, 0.7906712293624878, 0.7874478697776794, 0.7952218651771545, 0.7847933173179626, 0.7813803553581238, 0.7927569150924683, 0.7914296388626099, 0.7980659604072571, 0.7904816269874573, 0.7901023626327515, 0.797307550907135, 0.7901023626327515, 0.8018581867218018, 0.7753128409385681, 0.7901023626327515, 0.7828972339630127, 0.7885854840278625]
# Test accuracy: [0.7822815179824829, 0.7754403948783875, 0.7874123454093933, 0.7769796252250671, 0.7863861918449402, 0.7882674932479858, 0.7793740034103394, 0.7769796252250671, 0.7814263701438904, 0.7916880249977112, 0.7846758961677551, 0.7793740034103394, 0.7766375541687012, 0.7809132933616638, 0.789293646812439, 0.7922011017799377, 0.7696254253387451, 0.7913459539413452, 0.7754403948783875, 0.7846758961677551, 0.7846758961677551, 0.7754403948783875, 0.788609504699707, 0.7798870801925659, 0.7735590934753418, 0.7932272553443909, 0.7862151265144348, 0.7735590934753418, 0.7887805700302124, 0.7809132933616638, 0.7865571975708008, 0.7790319919586182, 0.7827945947647095, 0.7827945947647095, 0.7797160744667053, 0.7853599786758423, 0.7730460166931152, 0.7891225814819336, 0.7694544196128845, 0.7838207483291626, 0.7780057787895203, 0.7713357210159302, 0.788951575756073, 0.7694544196128845, 0.7747562527656555, 0.7904908061027527, 0.7793740034103394, 0.7773216962814331, 0.7708226442337036, 0.7674020528793335, 0.7824525237083435, 0.7716777920722961, 0.773901104927063, 0.7720198035240173, 0.773901104927063, 0.7776637673377991, 0.7750983238220215, 0.7851889729499817, 0.7754403948783875, 0.7793740034103394, 0.7879254221916199, 0.7737300992012024, 0.7851889729499817, 0.7732170224189758, 0.76124507188797, 0.7882674932479858, 0.7785188555717468, 0.7817684412002563, 0.7785188555717468, 0.7857020497322083, 0.7880964279174805, 0.7807422280311584, 0.7857020497322083, 0.7915170192718506, 0.781597375869751, 0.7956216931343079, 0.7783478498458862, 0.7894646525382996, 0.7788609266281128, 0.7843338251113892, 0.7865571975708008, 0.7709936499595642, 0.7884384989738464, 0.7744142413139343, 0.7658628225326538, 0.7944244742393494, 0.774243175983429, 0.7827945947647095, 0.7780057787895203, 0.7810842990875244, 0.7841628193855286, 0.7826235294342041, 0.781597375869751, 0.7792029976844788, 0.7809132933616638, 0.791003942489624, 0.7802291512489319, 0.7841628193855286, 0.7737300992012024, 0.7809132933616638, 0.7882674932479858, 0.7677441239356995, 0.7916880249977112, 0.7708226442337036, 0.7780057787895203, 0.7838207483291626, 0.7819394469261169, 0.7804002165794373, 0.7853599786758423, 0.7884384989738464, 0.7851889729499817, 0.783307671546936, 0.7877544164657593, 0.7906618714332581, 0.7819394469261169, 0.7857020497322083, 0.7831366062164307, 0.7822815179824829, 0.7774927020072937, 0.783649742603302, 0.7848469018936157, 0.7737300992012024, 0.7865571975708008, 0.7783478498458862, 0.7730460166931152, 0.7839917540550232, 0.7838207483291626, 0.7694544196128845, 0.7802291512489319, 0.7904908061027527, 0.7841628193855286, 0.7778347730636597, 0.7845048308372498, 0.7870702743530273, 0.7841628193855286, 0.7821104526519775, 0.7776637673377991, 0.7863861918449402, 0.7795450687408447, 0.7870702743530273, 0.7874123454093933, 0.7721908688545227, 0.7841628193855286, 0.7627843022346497, 0.7691123485565186, 0.7804002165794373, 0.7776637673377991, 0.7804002165794373, 0.7819394469261169, 0.7839917540550232, 0.7790319919586182, 0.783307671546936, 0.7795450687408447, 0.7814263701438904, 0.7757824063301086, 0.7845048308372498, 0.774243175983429, 0.7805712223052979, 0.7778347730636597, 0.7860441207885742, 0.7870702743530273, 0.7696254253387451, 0.7839917540550232, 0.7727039456367493, 0.7723618745803833, 0.788951575756073, 0.7732170224189758, 0.7822815179824829, 0.7845048308372498, 0.7865571975708008, 0.7838207483291626, 0.775611400604248, 0.7810842990875244, 0.7790319919586182, 0.7792029976844788, 0.7824525237083435, 0.7691123485565186, 0.7826235294342041, 0.7740721702575684, 0.7713357210159302, 0.7800581455230713, 0.762955367565155, 0.7872412800788879, 0.7687702775001526, 0.7778347730636597, 0.7882674932479858, 0.7768086194992065, 0.7800581455230713, 0.7850179672241211, 0.7792029976844788, 0.7922011017799377, 0.7788609266281128, 0.7776637673377991, 0.7824525237083435, 0.7786899209022522, 0.7899777293205261, 0.7639815211296082, 0.7771506309509277, 0.7740721702575684, 0.773901104927063]
# Average val accuracy: 0.791553 ± 0.006560
# Average test accuracy: 0.780575 ± 0.006644
