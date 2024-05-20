# SIR-GCN/GIN implementation on ZINCDataset

## Experiments

### SIR-GCN

```
python train.py --nhidden 75 --nlayers 4 --input-dropout 0 --edge-dropout 0 --dropout 0 --norm bn --readout-layers 2 --readout-dropout 0 --readout-pooling sum --residual --resid-layers 1 --resid-dropout 0 --feat-dropout 0 --agg-type sym --epochs 500 --batch-size 128 --lr 1e-3 --l1 1e-7 --l2 1e-7 --factor 0.5 --patience 10
```

## Summary

|  Model  |      Test MAE      | Parameters |
| :-----: | :------------------: | :--------: |
| SIR-GCN | 0.278175 ± 0.024087 |   99,676   |
