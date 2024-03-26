# SIR-GCN/GraphSAGE/GATv2/GIN implementation on GraphHeterophily dataset

## Experiments

### SIR-GCN (classes = $c$)

```
python train.py --model SIR --nhidden $((10 * c)) --nlayers 1 --nlayers-mlp 1 --nodes 50 --classes $c --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GraphSAGE (classes = $c$)

```
python train.py --model SAGE --nhidden $((10 * c)) --nlayers 1 --nodes 50 --classes $c --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GAT (classes = $c$)

```
python train.py --model GAT --nhidden $((10 * c)) --nlayers 1 --nheads 1 --nodes 50 --classes $c --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GIN (classes = $c$)

```
python train.py --model GIN --nhidden $((10 * c)) --nlayers 1 --nlayers-mlp 1 --nodes 50 --classes $c --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

## Summary

|          Model          |          Test MSE          | Parameters |
| :----------------------: | :-------------------------: | :--------: |
|  SIR-GCN (classes = 2)  |    0.000766 ± 0.000065    |    480    |
|  SIR-GCN (classes = 4)  |    0.084622 ± 0.228524    |   1,840   |
|  SIR-GCN (classes = 6)  |    0.208712 ± 0.440330    |   4,080   |
|  SIR-GCN (classes = 8)  |    0.283931 ± 0.292037    |   7,200   |
|  SIR-GCN (classes = 10)  |    0.190712 ± 0.194299    |   11,200   |
| GraphSAGE (classes = 2) | 23303.538866 ± 1275.888243 |    860    |
| GraphSAGE (classes = 4) | 52004.280178 ± 2890.689899 |   3,400   |
| GraphSAGE (classes = 6) | 64113.462877 ± 3452.275876 |   7,620   |
| GraphSAGE (classes = 8) | 70707.106702 ± 3736.558458 |   13,520   |
| GraphSAGE (classes = 10) | 74726.936288 ± 4045.603006 |   21,100   |
|    GAT (classes = 2)    | 22566.301221 ± 1269.173991 |    480    |
|    GAT (classes = 4)    | 44433.729514 ± 2691.017909 |   1,840   |
|    GAT (classes = 6)    | 49548.629688 ± 2760.261843 |   4,080   |
|    GAT (classes = 8)    | 50225.720372 ± 3415.420715 |   7,200   |
|    GAT (classes = 10)    | 49742.119637 ± 2726.633425 |   11,200   |
|    GIN (classes = 2)    |    40.345143 ± 2.655985    |    480    |
|    GIN (classes = 4)    |    38.158401 ± 1.555557    |   1,840   |
|    GIN (classes = 6)    |    33.317112 ± 2.110285    |   4,080   |
|    GIN (classes = 8)    |    31.584152 ± 1.422833    |   7,200   |
|    GIN (classes = 10)    |    30.090450 ± 1.450659    |   11,200   |
