# SIR-GCN/GIN implementation on SuperPixelDataset

## Experiments

### SIR-GCN (MNIST)

```
python train.py --nhidden 80 --nlayers 4 --input-dropout 0 --edge-dropout 0 --dropout 0 --norm bn --readout-layers 3 --readout-dropout 0 --readout-pooling mean --residual --resid-layers 1 --resid-dropout 0.2 --feat-dropout 0.1 --agg-type max --dataset MNIST --epochs 200 --batch-size 128 --lr 1e-3 --l1 1e-6 --l2 1e-6 --factor 0.5 --patience 10
```

### SIR-GCN (CIFAR10)

```
python train.py --nhidden 80 --nlayers 4 --input-dropout 0 --edge-dropout 0 --dropout 0 --norm bn --readout-layers 3 --readout-dropout 0 --readout-pooling mean --residual --resid-layers 1 --resid-dropout 0.2 --feat-dropout 0.1 --agg-type max --dataset CIFAR10 --epochs 200 --batch-size 128 --lr 1e-3 --l1 1e-6 --l2 1e-6 --factor 0.5 --patience 10
```

## Summary

|       Model       |    Test Accuracy    | Parameters |
| :---------------: | :------------------: | :--------: |
|  SIR-GCN (MNIST)  | 0.979030 ± 0.000806 |   99,610   |
| SIR-GCN (CIFAR10) | 0.719800 ± 0.003979 |  100,090  |
