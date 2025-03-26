# SIR-GCN implementation on HeterophilousGraphs

## Experiments

### SIR-GCN (roman-empire)

```
python train.py --use-amp --nhidden 512 --nlayers 5 --input-dropout 0.2 --dropout 0.2 --norm ln --residual --feat-dropout 0.2 --agg-type max --dataset roman-empire --epochs 1000 --lr 3e-5 --patience 1000 
```

### SIR-GCN (amazon-ratings)

```
python train.py --use-amp --nhidden 256 --nlayers 3 --input-dropout 0.2 --dropout 0.2 --norm bn --residual --feat-dropout 0.2 --agg-type max --dataset amazon-ratings --epochs 1000 --lr 3e-5 --patience 1000 
```

### SIR-GCN (minesweeper)

```
python train.py --use-amp --nhidden 256 --nlayers 5 --input-dropout 0.2 --dropout 0.2 --norm bn --residual --feat-dropout 0.2 --agg-type sym --dataset minesweeper --epochs 1000 --lr 3e-5 --patience 1000 
```

### SIR-GCN (tolokers)

```
python train.py --use-amp --nhidden 256 --nlayers 5 --input-dropout 0.2 --dropout 0.2 --norm ln --residual --feat-dropout 0.2 --agg-type sym --dataset tolokers --epochs 1000 --lr 3e-5 --patience 1000 
```

### SIR-GCN (questions)

```
python train.py --use-amp --nhidden 512 --nlayers 3 --input-dropout 0.2 --dropout 0.2 --norm ln --residual --feat-dropout 0.2 --agg-type sym --dataset questions --epochs 1000 --lr 3e-5 --patience 1000 
```

## Summary

|          Model          |     Test Metric     | Parameters |
| :----------------------: | :-------------------: | :--------: |
|  SIR-GCN (roman-empire)  | 0.876721 ± 0.002819 | 5,422,610 |
| SIR-GCN (amazon-ratings) | 0.467336 ± 0.006125 |  869,893  |
|  SIR-GCN (minesweeper)  | 0.941171 ± 0.004241 | 1,321,217 |
|    SIR-GCN (tolokers)    | 0.828513 ± 0.007204 | 1,321,985 |
|   SIR-GCN (questions)   | 0.753335 ± 0.013396 | 3,311,105 |
