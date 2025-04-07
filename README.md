# Soft-Isomorphic Relational Graph Convolution Network (SIR-GCN)

This is the official repository for the paper [Contextualized Messages Boost Graph Representations](https://arxiv.org/abs/2403.12529) ([TMLR, 2025](https://openreview.net/forum?id=sXr1fRjs1N)).

## Method

SIR-GCN emphasizes the anisotropic and dynamic (i.e., contextualized) transformation of neighborhood features to control and guide hash (aggregation) collisions when the space of node feature is uncountable. The model may be expressed as

$$\boldsymbol{h_u^*} = \sum_{v \in \mathcal{N}(u)} \boldsymbol{W_R} ~ \sigma\left(\boldsymbol{W_Q} \boldsymbol{h_u} + \boldsymbol{W_K} \boldsymbol{h_v}\right),$$

where $\sigma$ is a non-linear activation function. Leveraging linearity, the model has a computational complexity of 

$$\mathcal{O}\left(\left|\mathcal{V}\right| \times d_{\text{hidden}} \times d_{\text{in}} + \left|\mathcal{E}\right| \times d_{\text{hidden}} + \left|\mathcal{V}\right| \times d_{\text{out}} \times d_{\text{hidden}}\right).$$

## Experiments

All experiments are conducted on a single Nvidia Quadro RTX 6000 (24GB) card using the [Deep Graph Library (DGL)](https://www.dgl.ai/) with [PyTorch](https://pytorch.org/) backend.
