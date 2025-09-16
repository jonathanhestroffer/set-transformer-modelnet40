# Set Transformer

This project implements the Set Transformer model developed by [Leet et al., 2019](https://arxiv.org/pdf/1810.00825) to perform the task of point cloud classification using the [ModelNet40 dataset](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset).

# Introduction

The majority of learning tasks tackled by deep learning are *instance based* in that they involve learning fixed dimensional representations of the data for the applicable downstream task. For some applications however multiple instances of data are provided as input. Learning representations of such *set-structured* data requires models that are (1) *permutation invariant*  - the output of the model should not change upon any permutation of elements in the set. And (2) able to process input sets of any size.

To achieve this, Leet et al. 2019 designed a set-input deep neural network architecture called the *Set Transformer* which uses self-attention without positional encoding to both encode pairwise- or higher-order interactions between set elements and to perform feature aggregation/pooling.

# Background and Module Notation

## Multihead Attention Blocks (MABs)

Recalling the standard multi-head, scaled dot-product attention mechanism introduced by [Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762),

$$ \text{MultiHead}(Q,K,V) = \text{Concat} (\text{head}_\text{i},...,\text{head}_\text{h})W^O $$

$$ \text{head}_\text{i} = \text{Attention}(Q_i,K_i,V_i) $$

$$ \text{Attention}(Q_i,K_i,V_i) = \text{softmax}(\dfrac{Q_iK_i^T}{\sqrt{d_k}})V_i $$

$$ Q_i = XW_i^Q \ \text{and} \ K_i = YW_i^Q \ \text{and} \ V_i = YW_i^Q $$

for simplicity,

$$ X\in\mathbb{R}^{n\times d} \ \text{and} \ Y\in\mathbb{R}^{m\times d} $$

$$ W_i^Q,W_i^K,W_i^V\in\mathbb{R}^{d\times(\frac{d}{h})} \ \text{and} \ W^O\in\mathbb{R}^{d \times d}$$

The authors define the **Multihead Attention Block (MAB)** as,

$$ \text{MAB}(X,Y) = \text{LayerNorm}(H + \text{rFF}(H))$$
$$ H = \text{LayerNorm}(X + \text{MultiHead}(X,Y,Y))$$

where $\text{LayerNorm}$ and $\text{rFF}$ represent layer normalization [Ba et al., 2016](https://arxiv.org/abs/1607.06450) and a row-wise feedforward network respectively.

### Self-Attention:

$(X=Y)$  $\rightarrow\text{MultiHead}(X,X,X)$

Your queries, keys, and values all come from the same single input set/sequence.

### Cross-Attention:

$(X \neq Y)$  $\rightarrow\text{MultiHead}(X,Y,Y)$

Your queries come from $X$, your keys and values come from $Y$.

## Set Attention Blocks (SABs)

Full self-attention defined simply as,

$$ \text{SAB}(X) := \text{MAB}(X,X) $$

## Induced Set Attention Blocks (ISABs)

The authors note a potential problem with using SABs is the quadratic time complexity $\mathcal{O}(n^2)$, which can be too expensive for large sets $(n\gg1)$. As a solution, the authors propose the **Induced Set Attention Block (ISAB)** whereby $m$ $d$-dimensional vectors $I\in\mathbb{R}^{m\times d}$, called *inducing points*, are introduced as learnable parameters.

$$ \text{ISAB}_m(X)  = \text{MAB}(X,H) \in\mathbb{R}^{n\times d}, $$
$$ H = \text{MAB}(I,X) \in\mathbb{R}^{m\times d} $$

Here, the inducing points $I$ attend to the input sequence $X$ producing $H$, acting as a low-rank projection capturing the relevant features for the final task. The input $X$ then attends to the lower dimensional $H$. The time complexity of ISABs is then $\mathcal{O}(nm)$.

## Pooling by Multihead Attention (PMA)

The most common permutation invariant feature aggregation methods include computing dimension-wise averages or maxima of feature vectors (i.e., mean and max pooling). The authors instead propose generating learnable sets of pooled feature vectors via multihead attention. 

To do so, we introduce $k$ $d$-dimensional seed vectors $S\in\mathbb{R}^{k\times d}$. **Pooling by Multihead Attention (PMA)** is then defined as,

$$ \text{PMA}_k(Z) = \text{MAB}(S, \text{rFF}(Z))\in\mathbb{R}^{k\times d} $$

where $Z\in\mathbb{R}^{n\times d}$ represent the set of features constructed from an encoder. In most cases just one seed vector is used $(k=1)$.

## Overall Architecture

Set Transformer has the following basic architecture:

$ \text{Encoder}(X) = \text{SAB}(\text{SAB}(X)) $

or

$ \text{Encoder}(X) = \text{ISAB}_m(\text{ISAB}_m(X)) $

and 

$ \text{Decoder}(Z) = \text{rFF}(\text{SAB}(\text{PMA}_k(Z))) $