pragma circom 2.1.1;
include "../../node_modules/circomlib-ml/circuits/Dense.circom";
include "../../node_modules/circomlib-ml/circuits/Poly.circom";
include "../../node_modules/circomlib-ml/circuits/circomlib/mimc.circom";
include "./mimcsponge.circom";
include "utils.circom";

// Template to run ReLu on Dense Layer outputs
template RecursiveDenseLayer(nInputs, nOutputs) {

    signal input step_in[3];
    signal input in[nInputs];
    signal input weights[nInputs][nOutputs];
    signal input bias[nOutputs];
    signal activations[nOutputs];
    signal weights_matrix_hash;
    signal bias_vector_hash;
    signal output step_out[3];
    // NOTE: This is assuming a matrix of size (784 x 784)
    var matrix_dimension = 784;

    // Forward the hash of initial parameters
    step_out[0] <== step_in[0];

    // 1. Check that H(x) = v_n
    component mimc_previous_activations = MiMCSponge(nInputs, 220, 1);
    mimc_previous_activations.ins <== in;
    mimc_previous_activations.k <== 0;
    step_in[2] === mimc_previous_activations.outs[0];

    // 2. Compute activations = Relu(Ax + b) (Dense layer output is Ax + b)
    component dense = Dense(nInputs, nOutputs);
    dense.in <== in;
    dense.weights <== weights;
    dense.bias <== bias;

    component poly[nOutputs];
    // Now ReLu all of our outputs
    for (var i = 0; i < nOutputs; i++) {
        poly[i] = Poly(10**6);
        poly[i].in <== dense.out[i];
        activations[i] <== poly[i].out;
    }

    // 3. Update running hash parameter p_{n+1}
    component mimc_weights_matrix = MimcHashMatrix(matrix_dimension);
    mimc_weights_matrix.matrix <== weights;
    weights_matrix_hash <== mimc_weights_matrix.hash;

    component mimc_bias_vector = MiMCSponge(nOutputs, 220, 1);
    mimc_bias_vector.ins <== bias;
    mimc_bias_vector.k <== 0;
    bias_vector_hash <== mimc_bias_vector.outs[0];
    
    // Now p_{n+1} = Hash(p_n, Hash(Weights matrix), hash(bias vector))
    component pn_compositive_mimc = MiMCSponge(3, 220, 1);
    pn_compositive_mimc.ins[0] <== step_in[1];
    pn_compositive_mimc.ins[1] <== weights_matrix_hash;
    pn_compositive_mimc.ins[2] <== bias_vector_hash;
    pn_compositive_mimc.k <== 0;
    step_out[1] <== pn_compositive_mimc.outs[0];

    // 4. Compute v_{n+1} = H(Relu(Ax + b))
    component mimc_hash_activations = MiMCSponge(nOutputs, 220, 1);
    mimc_hash_activations.ins <== activations;
    mimc_hash_activations.k <== 0;
    step_out[2] <== mimc_hash_activations.outs[0];

}

component main { public [step_in] } = RecursiveDenseLayer(784, 784);
