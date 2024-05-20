# SIR-GCN/GATv2 implementation on ogbn-arxiv

## Preparations

### GIANT-XRT Embeddings

Follow the instructions at the [GIANT-XRT repository](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt).

```
wget https://archive.org/download/pecos-dataset/giant-xrt/ogbn-arxiv.tar.gz
tar -zxvf ogbn-arxiv.tar.gz
mv ogbn-arxiv dataset/ogbn_arxiv_xrt
rm -r ogbn-arxiv.tar.gz
```

## Experiments

### GIANT-XRT + SIR-GCN

```
python train.py --nhidden 256 --nlayers 1 --input-dropout 0.3 --edge-dropout 0.2 --dropout 0.4 --norm bn --readout-layers 1 --readout-dropout 0 --residual --resid-layers 1 --resid-dropout 0.6 --feat-dropout 0.5 --agg-type sym --add-self-loop --add-reverse-edge --use-xrt-emb --epochs 500 --lr 0.01 --l1 1e-6 --l2 1e-6 --factor 0.5 --patience 50 --save-pred
```

### GIANT-XRT + SIR-GCN + BoT + C&S

```
python train.py --nhidden 256 --nlayers 1 --input-dropout 0.3 --edge-dropout 0.2 --dropout 0.4 --norm bn --readout-layers 1 --readout-dropout 0 --residual --resid-layers 1 --resid-dropout 0.6 --feat-dropout 0.5 --agg-type sym --add-self-loop --add-reverse-edge --use-xrt-emb --use-labels --label-iters 3 --mask-rate 0.8 --epochs 500 --lr 0.01 --l1 1e-6 --l2 1e-6 --factor 0.5 --patience 50 --save-pred
python correct_and_smooth.py --add-self-loop --add-reverse-edge --use-sym
```

## Summary

|              Model              |    Test Accuracy    | Parameters |
| :-----------------------------: | :------------------: | :--------: |
|       GIANT-XRT + SIR-GCN       | 0.752515 ± 0.000908 |  667,176  |
| GIANT-XRT + SIR-GCN + BoT + C&S | 0.757357 ± 0.001976 |  697,896  |
