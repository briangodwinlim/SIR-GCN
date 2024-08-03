# SIR-GCN/GCN/GraphSAGE/GATv2/GIN/PNA implementation on GraphHeterophily

## Experiments

### SIR-GCN (classes = $c$)

```
python train.py --model SIR --nhidden $((10 * c)) --nlayers 1 --nodes 50 --classes $c --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GCN (classes = $c$)

```
python train.py --model GCN --nhidden $((10 * c)) --nlayers 1 --nodes 50 --classes $c --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
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

### PNA (classes = $c$)

```
python train.py --model PNA --nhidden $((10 * c)) --nlayers 1 --nodes 50 --classes $c --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

## Summary

|          Model          |          Test MSE          | Parameters |
| :----------------------: | :-------------------------: | :--------: |
|  SIR-GCN (classes = 2)  |    0.000875 ± 0.000076    |   1,320   |
|  SIR-GCN (classes = 4)  |    0.004258 ± 0.005203    |   5,120   |
|  SIR-GCN (classes = 6)  |    1.495102 ± 4.428228    |   11,400   |
|  SIR-GCN (classes = 8)  |    0.038115 ± 0.067555    |   20,160   |
|  SIR-GCN (classes = 10)  |    0.089362 ± 0.134417    |   31,400   |
|    GCN (classes = 2)    | 22749.102445 ± 1242.138932 |    480    |
|    GCN (classes = 4)    | 50806.774496 ± 2827.512044 |   1,840   |
|    GCN (classes = 6)    | 62632.997681 ± 3491.122026 |   4,080   |
|    GCN (classes = 8)    | 68965.326678 ± 3783.594837 |   7,200   |
|    GCN (classes = 10)    | 72985.544544 ± 4025.255478 |   11,200   |
| GraphSAGE (classes = 2) | 22962.426364 ± 1215.461141 |   1,300   |
| GraphSAGE (classes = 4) | 36853.816634 ± 2329.579652 |   5,080   |
| GraphSAGE (classes = 6) | 30551.579969 ± 1574.382135 |   11,340   |
| GraphSAGE (classes = 8) | 21886.120242 ± 1895.672093 |   20,080   |
| GraphSAGE (classes = 10) | 16529.294392 ± 1589.266900 |   31,300   |
|    GAT (classes = 2)    | 22329.455734 ± 1306.562863 |    500    |
|    GAT (classes = 4)    | 44971.537849 ± 2833.971397 |   1,880   |
|    GAT (classes = 6)    | 49939.875793 ± 2942.065555 |   4,140   |
|    GAT (classes = 8)    | 50063.074091 ± 3407.345106 |   7,280   |
|    GAT (classes = 10)    | 49661.203828 ± 3488.288494 |   11,300   |
|    GIN (classes = 2)    |    39.620488 ± 2.060092    |    480    |
|    GIN (classes = 4)    |    37.193070 ± 1.381894    |   1,840   |
|    GIN (classes = 6)    |    34.649004 ± 1.501609    |   4,080   |
|    GIN (classes = 8)    |    32.424090 ± 1.840737    |   7,200   |
|    GIN (classes = 10)    |    30.091226 ± 1.429145    |   11,200   |
|    PNA (classes = 2)    |   172.147297 ± 97.817594   |   2,960   |
|    PNA (classes = 4)    |   224.827848 ± 85.803848   |   11,600   |
|    PNA (classes = 6)    |  249.990954 ± 108.557330  |   25,920   |
|    PNA (classes = 8)    |   251.487032 ± 98.836990   |   45,920   |
|    PNA (classes = 10)    |   195.721559 ± 36.647286   |   71,600   |
