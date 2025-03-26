# SIR-GCN/GATv2 implementation on ogbn-arxiv

## Preparations

### GIANT-XRT Embeddings

Follow the instructions at the [GIANT-XRT repository](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt) (not used).

```
wget https://archive.org/download/pecos-dataset/giant-xrt/ogbn-arxiv.tar.gz
tar -zxvf ogbn-arxiv.tar.gz
mv ogbn-arxiv dataset/ogbn_arxiv_xrt
rm -r ogbn-arxiv.tar.gz
```

## Experiments

### SIR-GCN (100k)

```
python train.py --nhidden 95 --nlayers 3 --dropout 0.2 --norm bn --residual --feat-dropout 0.2 --agg-type sym --add-self-loop --add-reverse-edge --epochs 1000 --lr 1e-2 --wd 1e-3 --factor 0.5 --patience 40
```

## Summary

|     Model     |    Test Accuracy    | Parameters |
| :------------: | :------------------: | :--------: |
| SIR-GCN (100k) | 0.725155 ± 0.001617 |   98,745   |
