# zk cake

prove the execution of resnets w/ recursive snarks 

## Circuit Design
For an L-layer CNN. Bulk of the encoding done by the Backbone, where layers are verified with recursive SNARKs using Nova. Head & Tail layers are verified with single circuits that have model parameters directly built in. 

### Head Circuit `[layer 1]`
**Public Inputs**
1. $v_0 = H(a_0)$: Hash of input image 

**Public Outputs**
1. $v_1 = H(a_1)$: Hash of the activations produced by evaluating current layer

**Private Inputs**
1. $a_0$: Input image `[height x width x nChannels]`

**Logic**
1. Check that $H(a_0) = v_0$
1. Convolve filters stored in circuit ($W_1$ and $b_1$) over $a_0$ to produce $a_1$
1. Compute $v_1 = H(a_1)$

### Backbone Circuit `[layer 2, L)`
**Public Inputs**
1. $p_{n - 1} = H(H(H(W_2 || b_2) || W_3 || b_3) ... || W_{n - 1} || b_{n - 1}))$: Accumulated parameter hash
1. $v_{n - 1} = H(a_{n - 1})$: Hash of the activations (output) of the previous layer

**Public Outputs**
1. $p_n = H(p_n || H(W_n) || H(b_n))$: Updated running parameter hash 
1. $v_n = H(a_n)$: Hash of the activations produced by evaluating current layer

**Private Inputs**
1. $W_n$: Filters for convolution `[kernelSize x kernelSize x nChannels x nFilters]`
1. $b_n$: Bias vector `[nFilters]`
1. $a_{n-1}$: Input volume `[height x width x nChannels]`

**Logic**
1. Check that $H(a_{n-1}) = v_{n-1}$
1. Convolve $W_n$ and $b_n$ over $a_{n-1}$ to produce $a_n$
1. Compute $v_n = H(a_n)$
1. Update running parameter hash to $p_n$

### Tail Circuit `layer L`
**Public Inputs**
1. $v_{L - 1} = H(a_{L - 1})$: Hash of the activations (output) of last backbone layer 

**Public Outputs**
1. $v_L = H(a_L)$: Hash of the activations produced by evaluating current layer

**Private Inputs**
1. $W_L$: Matrix transformation [kernelSize x kernelSize x nChannels x nFilters]
1. $b_n$: Bias vector [nFilters]
1. $x_n$: Input volume [height x width x nChannels]

**Logic**
1. Check that $H(x) = v_n$
1. Convolve $W_n$ and $b_n$ over $x_n$ to produce $a_n$
1. Compute $v_{n + 1} = H(a_n)$
1. Update running parameter hash $p_{n + 1}$
