# SIR-GCN/GATv2 implementation on WikiCSDataset

## Experiments

### SIR-GCN

```
python train.py --nhidden 60 --nlayers 4 --input-dropout 0.3 --edge-dropout 0 --dropout 0.1 --norm bn --readout-layers 1 --readout-dropout 0 --residual --resid-layers 1 --resid-dropout 0.2 --feat-dropout 0 --agg-type mean --epochs 200 --lr 1e-3 --l1 1e-5 --l2 1e-5 --factor 0.5 --patience 10
```

## Summary

|  Model  |    Test Accuracy    | Parameters |
| :-----: | :------------------: | :--------: |
| SIR-GCN | 0.780575 ± 0.006644 |   102,850   |
