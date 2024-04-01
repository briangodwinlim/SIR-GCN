# SIR-GCN/GraphSAGE/GATv2/GIN implementation on DictionaryLookup

## Experiments

### SIR-GCN (nodes = $n$)

```
python train.py --model SIR --nhidden $((4 * n)) --nlayers 1 --nodes $n --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GraphSAGE (nodes = $n$)

```
python train.py --model SAGE --nhidden $((4 * n)) --nlayers 1 --nodes $n --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GAT (nodes = $n$)

```
python train.py --model GAT --nhidden $((4 * n)) --nlayers 1 --nheads 1 --nodes $n --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GIN (nodes = $n$)

```
python train.py --model GIN --nhidden $((4 * n)) --nlayers 1 --nlayers-mlp 2 --nodes $n --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

## Summary

|         Model         |    Test Accuracy    | Parameters |
| :--------------------: | :------------------: | :--------: |
|  SIR-GCN (nodes = 10)  | 1.000000 ± 0.000000 |   9,480   |
|  SIR-GCN (nodes = 20)  | 1.000000 ± 0.000000 |   37,360   |
|  SIR-GCN (nodes = 30)  | 1.000000 ± 0.000000 |   83,640   |
|  SIR-GCN (nodes = 40)  | 1.000000 ± 0.000000 |  148,320  |
|  SIR-GCN (nodes = 50)  | 1.000000 ± 0.000000 |  231,400  |
| GraphSAGE (nodes = 10) | 0.099100 ± 0.004160 |   4,520   |
| GraphSAGE (nodes = 20) | 0.050740 ± 0.001414 |   17,840   |
| GraphSAGE (nodes = 30) | 0.033370 ± 0.001146 |   39,960   |
| GraphSAGE (nodes = 40) | 0.024878 ± 0.000545 |   70,880   |
| GraphSAGE (nodes = 50) | 0.020156 ± 0.000679 |  110,600  |
|    GAT (nodes = 10)    | 0.989930 ± 0.030210 |   2,960   |
|    GAT (nodes = 20)    | 0.884880 ± 0.180227 |   11,520   |
|    GAT (nodes = 30)    | 0.739013 ± 0.283080 |   25,680   |
|    GAT (nodes = 40)    | 0.554853 ± 0.369063 |   45,440   |
|    GAT (nodes = 50)    | 0.598708 ± 0.402419 |   70,800   |
|    GIN (nodes = 10)    | 0.775870 ± 0.069771 |   4,560   |
|    GIN (nodes = 20)    | 0.290820 ± 0.025707 |   17,920   |
|    GIN (nodes = 30)    | 0.119097 ± 0.030321 |   40,080   |
|    GIN (nodes = 40)    | 0.028970 ± 0.010733 |   71,040   |
|    GIN (nodes = 50)    | 0.021502 ± 0.005474 |  110,800  |
