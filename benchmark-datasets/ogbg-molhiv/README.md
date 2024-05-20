# SIR-GCN/GIN implementation on ogbg-molhiv

## Preparations

### Molecular Fingerprints

Run the script below to generate molecular fingerprints (not used).

```
python fingerprint.py --morgan --maccs --rdkit --save
```

## Experiments

### SIR-GCN

```
python train.py --nhidden 300 --nlayers 1 --input-dropout 0 --edge-dropout 0 --dropout 0.4 --norm bn --readout-layers 1 --readout-dropout 0 --readout-pooling mean --residual --resid-layers 0 --resid-dropout 0.1 --feat-dropout 0 --agg-type sum --epochs 200 --batch-size 128 --lr 1e-3 --l1 1e-7 --l2 1e-7 --factor 0.5 --patience 20
```

### SIR-GCN + GraphNorm

```
python train.py --nhidden 300 --nlayers 1 --input-dropout 0 --edge-dropout 0 --dropout 0.4 --norm gn --readout-layers 1 --readout-dropout 0 --readout-pooling mean --residual --resid-layers 0 --resid-dropout 0.1 --feat-dropout 0 --agg-type sum --epochs 200 --batch-size 128 --lr 1e-3 --l1 1e-7 --l2 1e-7 --factor 0.5 --patience 20
```

## Summary

|        Model        |     Test ROC-AUC     | Parameters |
| :-----------------: | :------------------: | :--------: |
|       SIR-GCN       | 0.772064 ± 0.010995 |  327,901  |
| SIR-GCN + GraphNorm | 0.798126 ± 0.006157 |  328,201  |
