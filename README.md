# Soft-Isomorphic Relational Graph Convolution Network (SIR-GCN)

This is the official repository of the paper [Contextualized Messages Boost Graph Representations](https://arxiv.org/abs/2403.12529).

## Method

SIR-GCN emphasizes the non-linear and contextualized transformation of the neighborhood features to control and guide hash collisions when the space of node feature is uncountable. The model may be expressed as

$$\boldsymbol{h_u^*} = \sum_{v \in \mathcal{N}(u)} \boldsymbol{W_R} \cdot \sigma\left(\boldsymbol{W_Q} \boldsymbol{h_u} + \boldsymbol{W_K} \boldsymbol{h_v}\right),$$

where $\sigma$ is a non-linear activation function.

## Experiments

All experiments are conducted on a single Quadro RTX 6000 (24GB) card using the [Deep Graph Library (DGL)](https://www.dgl.ai/). 

Implementation for the following datasets are available

- DictionaryLookup ([Brody et al.](https://arxiv.org/abs/2105.14491))
- GraphHeterophily
