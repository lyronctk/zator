# zk cake

prove the execution of resnets w/ recursive snarks 

## to do 
1. figure out quantization scheme of circomlib-ml
1. train a quantized three layer FFNN on MNIST 
1. circom circuit that is one matmul + relu combo 
1. recurse w/ nova scotia 

## circuit 
PUBLIC INPUTS 
1. $g_n$: $H(x)$, hash of the initial input 
1. $p_n$: $H(H(H(A_1 || b_1) || A_2 || b_1) ... || A_{n - 1} || b_{n - 1}))$, accumulated parameter hash
1. $v_n$: $H(a_{n - 1})$, hash of the activations of the previous layer, is equal to $g_n$ for layer 1 

PUBLIC OUTPUTS (symmetric with inputs)
1. $g_{n + 1}$: $g_n$, hash of the initial input
1. $p_{n + 1}$: $H(p_n || A_n || b_n)$, updated running parameter hash 
1. $v_{n + 1}$: $H(a_n)$, hash of the activations produced by evaluating current layer

PRIVATE INPUTS 
1. $A$: Matrix transformation
1. $b$: Bias vector 
1. $x$: Input vector

LOGIC
1. Check that $H(x) = v_n$
1. Compute $a_n = RELU(Ax + b)$ 
1. Compute $v_{n + 1} = H(a_n)$
1. Update running parameter hash $p_{n + 1}$
1. Keep forwarding $g_n$
