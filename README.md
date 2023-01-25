# Zator

Proving the execution of arbitrarily deep neural networks with recursive SNARKs. 

**ETA for fleshing out README: Feb 6**

## Motivation
With the increasing use of deep learning in important decision-making processes, such as healthcare and finance, it is crucial that these models can be verified and trusted. In this project, we explored using zero knowledge succinct non-interactive arguments of knowledge (zk-SNARKs) to provide verfication on the inference for an arbitrarily deep neural network. Specifically, we trained a 512 layer MNIST model for digit classification and verified the inference execution trace using recruse SNARKs. 

There has been tremendous progress in the past year toward verifying neural network inference using SNARKs. Along this line of research, notable projects such as [EZKL](https://github.com/zkonduit/ezkl) and work by [D. Kang et al](https://arxiv.org/pdf/2210.08674.pdf) have been able to leverage properties of the Halo2 proving system to snark models as complex as MobileNetv2 with 50 layers. 

The primary constraint preventing these efforts from expanding to even deeper models is the fact that they attempt to fit the entire computation trace into a single circuit. With Zator, we wanted to explore verifying one layer at a time using recursive SNARKs, a class of SNARKs that enables an N-step (in our case, N-layer) repeated computation to be verified incrementally. We leverage a recent construction called [Nova](https://github.com/microsoft/Nova) that uses a folding scheme to reduce N instances of repeated computation into a single instance that can be verified at the cost of a single step. We looked to utilize the remarkably light recursive overhead of folding (10k constraints per step) to SNARK a network with 512 layers, which is as deep or deeper than the majority of production AI models today. 

## Why SNARK a 512-Layer network now?
Advancements in zero knowledge proofs, such as [Microsoft's Nova](https://github.com/microsoft/Nova) prover system and [Nova Scotia](https://github.com/nalinbhardwaj/Nova-Scotia), allow the generation of recursive SNARKs using circom. Recursive SNARKs are particularly useful in the context of verifying complex  computations, such as those performed by a homogeneous neural network where many intermediary layers are identical, because they allow for the efficient verification of large and repetitive circuits.

The "recursive" in recursive SNARKs refers to the ability to use the proof for one statement as the input for the next proof. This allows for a series of statements to be proven using a single proof, reducing the overall size and complexity of the proof and making the verification process more efficient.

## SNARKing an arbitrary depth neural network
We spent the lask week hacking on [Zator](https://github.com/lyronctk/zator), a framework for verifying the computation trace for an arbitrary depth neural network. Our final design composes the [Nova](https://eprint.iacr.org/2021/370) and [Spartan](https://eprint.iacr.org/2019/550) proving systems. Nova is a recent construction with remarkably low recursive overhead and minimal cryptographic assumptions. Rather than creating a separate SNARK proof at each of the N steps of recursion, Nova works by instantiating a single "relaxed" R1CS instance at the beginning of the computation and "folding" it N times. Folding is a fast operation primarily made up of MSMs (rather than expensive FFTs) that results in a single instance that, when satisfied, proves the execution of all N steps. Upshot: expensive SNARK machinery only needs to be invoked to prove this *single* instance. We feed this folded instance into Spartan to create a final succinct proof.

The recursive structure for the network exists in the repetitive backbone. Head and tail layers are are proved separately since they cannot be homogenous with the rest of the network (i.e. the head needs to project the input image into the space we work in and the tail needs to transform into output probabilities).

## Circuit Design
![Untitled-2023-01-10-1700](https://user-images.githubusercontent.com/97858468/212182755-d0ceca49-71f3-4ec8-b627-46da56fd7261.svg)

We split the network into three parts: a head, backbone, and a tail. We assume an L-layer CNN. The backbone layers are verified with recursive SNARKs using Nova. Head & Tail layers are verified with single circuits that have model parameters directly built in. 

### **Head Circuit** - `[layer 1]`
We denote the first layer of our neural net as the head layer. The head accepts a 28x28 image from the MNIST database and outputs activations. The corresponding head layer circuit accepts the input image as a 28x28 matrix as a private signal, along with the weights matrix and bias vector as additional private signals. The head layer circuit outputs a hash of the activations, denoted in our diagram by $v_1$.
#### Public Inputs
1. $v_0 = H(a_0)$: Hash of input image 

#### Public Outputs
1. $v_1 = H(a_1)$: Hash of the activations produced by evaluating current layer

#### Private Inputs
1. $a_0$: Input image `[imgHeight x imgWidth x nChannels]`

#### Logic
1. Check that $H(a_0) = v_0$
1. Convolve filters stored in circuit ($W_1$ / $b_1$) over $a_0$ to produce $a_1$
1. Compute $v_1 = H(a_1)$


### **Backbone Circuit** - `[layer 2, L)`
Intermediary layers of our neural net have a corresponding backbone circuit that proves the execution of that specific layer. It ingests the activations of the previous layer ($a_{n-1}$), a weight matrix ($W_n$), and a bias vector ($b_n$) as private signals. Two steps are taken in order to prove correct execution of the layer. First, the circuit accepts the hash of the previous activations (i.e $v_{n-1}$) as a public signal. It hashes the passed in activations, and verifies the two values match (i.e it verifies that $hash(a_{n-1}) == v_{n-1}$). Secondly, a "running hash" denoted $p_n$ is a public signal to the circuit. This running hash is defined as $p_n = H(p_{n-1} || H(W_n) || H(b_n))$ and creates a running chain of commitments to weights and bias inputs. This ensures that all layers in the proof are part of the same execution trace. As public outputs, the circuit produces $p_n$ and $v_n$.
#### Public Inputs
1. $p_{n - 1} = H(H(H(W_2 || b_2) || W_3 || b_3) ... || W_{n - 1} || b_{n - 1}))$: Accumulated parameter hash
1. $v_{n - 1} = H(a_{n - 1})$: Hash of the activations (output) of the previous layer

#### Public Outputs
1. $p_n = H(p_n || H(W_n) || H(b_n))$: Updated running parameter hash 
1. $v_n = H(a_n)$: Hash of the activations produced by evaluating current layer

#### Private Inputs
1. $W_n$: Filters for convolution `[kernelSize x kernelSize x nChannels x nFilters]`
1. $b_n$: Bias vector `[nFilters]`
1. $a_{n-1}$: Input volume `[imgHeight x imgWidth x nChannels]`

#### Logic
1. Check that $H(a_{n-1}) = v_{n-1}$
1. Convolve $W_n$ / $b_n$ over $a_{n-1}$ to produce $a_n$
1. Compute $v_n = H(a_n)$
1. Update running parameter hash to $p_n$

### **Tail Circuit** - `layer L`
The final layer in our neural network (the tail layer) corresponds to our tail circuit. Note that $L$ denotes the final layer number (512 in our case). The tail accepts as public input $v_{L-1}$ (the hash of the activations of the previous layer), along with private inputs $W_L$, $b_L$, and $a_{L-1}$. Similar to our backbone circuits, it hashes the passed in activations, and verifies the two values match (i.e it verifies that $hash(a_{L-1}) == v_{L-1}$). As an output, it produces $v_L$, a hash of the activations produced.
#### Public Inputs
1. $v_{L - 1} = H(a_{L - 1})$: Hash of the activations (output) of last backbone layer 

#### Public Outputs
1. $v_L = H(a_L)$: Hash of the activations produced by evaluating current layer

#### Private Inputs
1. $W_L$: Matrix transformation `[(imgHeight * imgWidth) x nClasses]`
1. $b_n$: Bias vector `[nClasses]`
1. $x_n$: Input volume `[imgHeight x imgWidth x nFilters]`

#### Logic
1. Check that $H(a_{L-1}) = v_{L-1}$
1. Convolve $W_L$ / $b_L$ over $a_{L-1}$ to produce $a_L$
1. Compute $v_L = H(a_L)$

The backbone (intermediary layers) have a homogeneous structure, which allows us to leverage Nova to bundle up the proofs for all these layers into a single proof through the folding mechanism described above. 

In the end, we have 3 total proofs: 1 for the head layer, 1 for all the backbone layers, and 1 for the final tail layer. A verifier would check the validity of all 3 proofs and trace through their public outputs to ensure that all 3 proofs were part of the same execution trace. 

Rather than verify 3 proofs, it is possible to have a single proof that verifies the entire execution trace. Due to the details of folding, Nova requires that all circuits folded into a single relaxed R1CS are homogenous. Since the head & tail layers are necessarily different than intmediary layers, we can't put head, backbone, & tail layers through Nova off the bat. It is possible to support hetrogenous neural nets through a multiplexing architecture, where we simply denote with a signal which layer we are "turning on". This is too heavy with our current implementation. Note that [SuperNova](https://eprint.iacr.org/2022/1758), the successor to Nova, will support hetrogenous architectures more natively.[^1]

[^1]: While the Supernova spec paper has been published, the implementation will be published in the future

## ZK-friendly neural network 
Our network consisted of only convolution layers for the backbone. Why only convolutions? 

1. Homogenous architectures (e.g. fully convolutional, fully linear) lend themselves to SNARK recursion as mentioned early (Nova requires a homogeneous architecture in order to perform folding)
2. Linear layers have many, many more weights than a convolutional layer. More weights = more constraints = large proving time. For example, a dense layer for a 28x28 MNIST image has a [784 x 784] weight matrix, which requires ~350m constraints to hash when using 220 rotations on MiMC. 

Our current neural net is 512 layers, but with more compute we could scale up to any number of layers. Our model works on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database), a standard for benchmarking models.

![](https://i.imgur.com/T9H1T1Q.png)

The weights and bias matrices of the head and tail circuits are hardcoded into the circuits themselves. The backbone circuit was written in circom and translated to Nova using Nova Scotia. At the end of an inference, Zator returns three proofs: two PLONK proofs, each for the head and tail, and a third [Spartan](https://eprint.iacr.org/2019/550) proof generated by Nova and recursive SNARKs.

## Benchmarks 
* [TODO]: table with layers & proving time
* [TODO]: disclaimer that each layer is small, POC, many optimizations carried out in similar projects eg. https://arxiv.org/pdf/2210.08674.pdf
    * in particular, ideally have above table for ImageNet as well

## Challenges and learnings 

Unlike most other week long hacks, we don't really have a final demo to share. However, through building Zator, we learned a lot of nuances about using the nascent ZKML toolchain that we'd like to share. 

* SNARKs
    * By default, Circom utilizes the BN-128 elliptic curve, and many of the libraries in the circom ecosystem are built with the assumption that BN-128 is used. However, Nova uses the Vesta curve (part of the family of [pasta curves](https://electriccoin.co/blog/the-pasta-curves-for-halo-2-and-beyond/)). Thus, we had to fork multiple libraries and make modifications to support Vesta. An example is [Dr. Cathie's library of ML circuits](https://github.com/socathie/circomlib-ml), and [our fork](https://github.com/verumlotus/circomlib-ml-vesta) supporting Vesta
    * Circom interopates with Groth16 and PLONK out of the box. Nova, however, uses the Spartan proving system. Currenty our architecture consists of 2 PLONK proofs for the head & tail layer, and a Spartan Proof for the backbone layers (that are folded into one proof). In the future, we intend to use a single proving system (Spartan) for all of our circuits
    * Depending on the architecture of your neural network, the compilation time of your circuit can grow rapidly. With a dense layer architecture, we had ~350M constraints and switched to convolutional layers to keep the number of constraints reasonable
    * We noticed that commitments to input data (e.g hashing the weights matrix) were a bulk of our circuit constraints
    * Since Circom uses an R1CS constraint system, we were unable to write custom gates. This means that operations such as divsions over a finite field and non-linearities (e.g ReLU) lead to many constraints
    * While we used WASM to generate our witness, C++ witness generation can be parallelized and an order of magnitude faster
* Modeling Neural Networks in Circom
    * Negative numbers are commonly used in neural nets, but do not exist over finite fields. To deal with this, we split the field into two halves. With all operations taken over a prime $p$, all values less than $floor(p/2)$ were treated as positive, and all values above this were treated as negative. The intuition is that negative numbers will wrap around to become a very large number in the finite field (e.g -1 maps to $p - 1$)
    * Floating point numbers are used in neural nets, but we are only able to work with integers within Circom. We quantized all of our values to integers by multiplying by a scale factor and then floor dividing all values. The accuracy of our model suffered due to integer only weights. There is a growing body of literature on quantization aware training that can help maintain model performance.[^2] 
    * The weights of a fully connected net can be too expensive (leads to too many constraints)
* Nova
    * Folding is quite efficient in Nova (about ~45 seconds/layer), but generating a proof in Nova requires generating public parameters. In our experience this take upwards of 2 hours, and domintaes the proving time. Serializing the proving parameters to allow for quicker iterations would be a significant improvement, and a contribution that we are exploring as a team. 
    * In Nova, the generation and verification of a proof are bundled together, meaning that proofs are never exposed to the user. We made modifications that allowed us to log the in-memory proof to a file in JSON format, but native support for this would allow for quicker iteration

[^2]: This is an active area of research. See [here](https://arxiv.org/abs/1712.05877), [here](https://www.tensorflow.org/model_optimization/guide/quantization/training), and [here](https://developer.nvidia.com/blog/improving-int8-accuracy-using-quantization-aware-training-and-tao-toolkit/) for some examples.

## Example proofs

Below are example proofs for a run with 512 layers.

- [Head](https://gist.github.com/varunshenoy/945fe6231b9a077160a0ae2360b854ab#file-head_layer_proof-json)
- Backbone
- [Tail](https://gist.github.com/varunshenoy/945fe6231b9a077160a0ae2360b854ab#file-tail_layer_proof-json)

## Contributing 
- [x] Head and Tail SNARK
- [x] Recursive SNARK Backbone
- [x] Integer quantization at arbitrary depth
- [ ] next...

## Disclaimer & Credits
This project was built out of interest and has not been thoroughly audited or battle-tested. We had help along the way and would like to thank: 
* [Nalin](https://nibnalin.me/) for his ZK wizardry 
* [Hack Lodge](https://hacklodge.org/) for the mentorship, friends, & support
* [Srinath](http://srinathsetty.net/) for his work on Nova  
* [Dr. Cathie](https://twitter.com/drCathieSo_eth) for her helpful ML circuits library
