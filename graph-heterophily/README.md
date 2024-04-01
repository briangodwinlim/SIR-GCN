# sSIR-GCN/GraphSAGE/GATv2/GIN implementation on GraphHeterophily

## Experiments

### SIR-GCN (classes = $c$)

```
python train.py --model SIR --nhidden $((10 * c)) --nlayers 1 --nodes 50 --classes $c --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
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
|  SIR-GCN (classes = 2)  |    0.000875 ± 0.000076    |   1,320   |
|  SIR-GCN (classes = 4)  |    0.004258 ± 0.005203    |   5,120   |
|  SIR-GCN (classes = 6)  |    0.005451 ± 0.004927    |   11,400   |
|  SIR-GCN (classes = 8)  |    0.080144 ± 0.096676    |   20,160   |
|  SIR-GCN (classes = 10)  |    0.089362 ± 0.134417    |   31,400   |
| GraphSAGE (classes = 2) | 23214.047926 ± 1262.416097 |    880    |
| GraphSAGE (classes = 4) | 51892.594971 ± 2908.854789 |   3,440   |
| GraphSAGE (classes = 6) | 64135.079252 ± 3426.201783 |   7,680   |
| GraphSAGE (classes = 8) | 70708.779051 ± 3738.015090 |   13,600   |
| GraphSAGE (classes = 10) | 74751.465270 ± 4079.899968 |   21,200   |
|    GAT (classes = 2)    | 22319.938380 ± 1288.360150 |    500    |
|    GAT (classes = 4)    | 45036.694787 ± 2941.190224 |   1,880   |
|    GAT (classes = 6)    | 50112.055970 ± 3063.393533 |   4,140   |
|    GAT (classes = 8)    | 49965.998744 ± 3253.327400 |   7,280   |
|    GAT (classes = 10)    | 49938.593322 ± 3496.310929 |   11,300   |
|    GIN (classes = 2)    |    40.021049 ± 3.434329    |    480    |
|    GIN (classes = 4)    |    37.226528 ± 1.421944    |   1,840   |
|    GIN (classes = 6)    |    34.471370 ± 1.920446    |   4,080   |
|    GIN (classes = 8)    |    31.585826 ± 1.421587    |   7,200   |
|    GIN (classes = 10)    |    30.423154 ± 1.729089    |   11,200   |
