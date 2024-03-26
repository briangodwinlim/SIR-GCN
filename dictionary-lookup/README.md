# SIR-GCN/GraphSAGE/GATv2/GIN implementation on DictionaryLookup dataset

## Experiments

### SIR-GCN (nodes = $n$)

```
python train.py --model SIR --nhidden $((5 * n)) --nlayers 1 --nlayers-mlp 2 --nodes $n --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GraphSAGE (nodes = $n$)

```
python train.py --model SAGE --nhidden $((5 * n)) --nlayers 1 --nodes $n --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GAT (nodes = $n$)

```
python train.py --model GAT --nhidden $((5 * n)) --nlayers 1 --nheads 1 --nodes $n --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GIN (nodes = $n$)

```
python train.py --model GIN --nhidden $((5 * n)) --nlayers 1 --nlayers-mlp 2 --nodes $n --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

## Summary

|         Model         |    Test Accuracy    | Parameters |
| :--------------------: | :------------------: | :--------: |
|  SIR-GCN (nodes = 10)  | 1.000000 ± 0.000000 |   6,700   |
|  SIR-GCN (nodes = 20)  | 1.000000 ± 0.000000 |   26,400   |
|  SIR-GCN (nodes = 30)  | 1.000000 ± 0.000000 |   59,100   |
|  SIR-GCN (nodes = 40)  | 1.000000 ± 0.000000 |  104,800  |
|  SIR-GCN (nodes = 50)  | 1.000000 ± 0.000000 |  163,500  |
| GraphSAGE (nodes = 10) | 0.100280 ± 0.004878 |   6,600   |
| GraphSAGE (nodes = 20) | 0.049520 ± 0.001852 |   26,200   |
| GraphSAGE (nodes = 30) | 0.033677 ± 0.001169 |   58,800   |
| GraphSAGE (nodes = 40) | 0.024700 ± 0.000638 |  104,400  |
| GraphSAGE (nodes = 50) | 0.020258 ± 0.000397 |  163,000  |
|    GAT (nodes = 10)    | 0.999960 ± 0.000120 |   4,150   |
|    GAT (nodes = 20)    | 0.979930 ± 0.060210 |   16,300   |
|    GAT (nodes = 30)    | 0.834230 ± 0.255738 |   36,450   |
|    GAT (nodes = 40)    | 0.917453 ± 0.247642 |   64,600   |
|    GAT (nodes = 50)    | 0.761626 ± 0.364743 |  100,750  |
|    GIN (nodes = 10)    | 0.868200 ± 0.073406 |   6,700   |
|    GIN (nodes = 20)    | 0.336830 ± 0.029015 |   26,400   |
|    GIN (nodes = 30)    | 0.136640 ± 0.012435 |   59,100   |
|    GIN (nodes = 40)    | 0.033505 ± 0.016597 |  104,800  |
|    GIN (nodes = 50)    | 0.020216 ± 0.000661 |  163,500  |
