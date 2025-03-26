# SIR-GCN/GIN implementation on ogbg-molhiv

## Preparations

### Molecular Fingerprints

Run the script below to generate molecular fingerprints (not used).

```
python fingerprint.py --morgan --maccs --rdkit --save
```

## Experiments

### SIR-GCN (100k)

```
python train.py --nhidden 80 --nlayers 4 --input-dropout 0.2 --norm bn --readout-pooling mean --residual --feat-dropout 0.2 --agg-type max --epochs 100 --batch-size 64 --lr 1e-3 --wd 1e-4 --factor 0.5 --patience 10
```

## Summary

|     Model     |     Test ROC-AUC     | Parameters |
| :------------: | :------------------: | :--------: |
| SIR-GCN (100k) | 0.776309 ± 0.008434 |   96,521   |
