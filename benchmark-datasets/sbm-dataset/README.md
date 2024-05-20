# SIR-GCN/GATv2 implementation on SBMDataset

## Experiments

### SIR-GCN (PATTERN)

```
python train.py --nhidden 80 --nlayers 4 --input-dropout 0 --edge-dropout 0 --dropout 0 --norm bn --readout-layers 1 --readout-dropout 0 --residual --resid-layers 1 --resid-dropout 0 --feat-dropout 0 --agg-type sym --dataset PATTERN --epochs 200 --batch-size 128 --lr 1e-3 --l1 1e-7 --l2 1e-7 --factor 0.5 --patience 10
```

### SIR-GCN (CLUSTER)

```
python train.py --nhidden 80 --nlayers 4 --input-dropout 0 --edge-dropout 0 --dropout 0 --norm bn --readout-layers 1 --readout-dropout 0 --residual --resid-layers 1 --resid-dropout 0 --feat-dropout 0 --agg-type sym --dataset CLUSTER --epochs 200 --batch-size 128 --lr 1e-3 --l1 1e-7 --l2 1e-7 --factor 0.5 --patience 10
```

## Summary

|       Model       |    Test Accuracy    | Parameters |
| :---------------: | :------------------: | :--------: |
| SIR-GCN (PATTERN) | 0.857542 ± 0.000263 |  104,722  |
| SIR-GCN (CLUSTER) | 0.633513 ± 0.001854 |  105,366  |
