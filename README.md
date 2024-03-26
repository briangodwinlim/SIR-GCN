# Soft-Isomorphic Relational Graph Convolution Network (SIR-GCN)

This is the official repository of the paper [Contextualized Messages Boost Graph Representations](https://arxiv.org/abs/2403.12529).

## Method

SIR-GCN emphasizes the non-linear and contextualized transformation of the neighborhood features to control and guide hash collisions when the space of node feature is uncountable. The model may be expressed as

$$\boldsymbol{h_u^*} = \sigma\left(\sum_{v \in \mathcal{N}(u)} \text{MLP}_A\left(\text{MLP}_K\left(\boldsymbol{h_v}\right) + \text{MLP}_Q\left(\boldsymbol{h_u}\right)\right)\right),$$

where $\sigma$ is a non-linear activation function, $\text{MLP}_Q$ and $\text{MLP}_K$ serve as pre-processing steps for the features of query and key nodes, respectively, to ensure asymmetry and $\text{MLP}_A$ is the relational mechanism.

## Experiments

All experiments are conducted on a single Quadro RTX 6000 (24GB) card using the [Deep Graph Library (DGL)](https://www.dgl.ai/). 

Implementation for the following datasets are available

- DictionaryLookup ([Brody et al.](https://arxiv.org/abs/2105.14491))
- GraphHeterophily
