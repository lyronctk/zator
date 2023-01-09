pragma circom 2.1.1;
include "../../node_modules/circomlib-ml/circuits/Dense.circom";
include "../../node_modules/circomlib-ml/circuits/Poly.circom";

// Template to run ReLu on Dense Layer outputs

template dense_layer(nInputs, nOutputs) {
    signal input in[nInputs];
    signal input weights[nInputs][nOutputs];
    signal input bias[nOutputs];
    signal output out[nOutputs];

    component dense = Dense(nInputs, nOutputs);
    dense.in <== in;
    dense.weights <== weights;
    dense.bias <== bias;

    component poly[nOutputs];
    // Now ReLu all of our outputs
    for (var i = 0; i < nOutputs; i++) {
        poly[i] = Poly(10**6);
        poly[i].in <== dense.out[i];
        out[i] <== poly[i].out;
    }
}

component main = dense_layer(784, 512);
