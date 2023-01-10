pragma circom 2.1.1;

template TailLayer() {
    var inRows = 3136;
    var outCols = 10;

    signal input prevHash;
    signal input in[inRows];

    var activations[outCols] = [-28639993667, -34677528381, -21898008346, -19130430221, -40692321777, -30850053787, -43104949951, -6874659061, -25667932510, -25357667922];
    var bias[outCols] = [-16762316, 79053082, -54210544, -95354311, 56738410, -37802950, 4387544, 4411432, -51621947, -40832181];
    var weights[2][3] = [[5, 10, 10], [1010, 10, 0]];

    signal output finalHash;

    // LOGIC
    // 1. Check hash of activation from prev layer matches hash of input to this layer
    component mimc_previous_activations = MiMCSponge(inRows, 220, 1);
    mimc_previous_activations.ins <== in;
    mimc_previous_activations.k <== 0;
    prevHash === mimc_previous_activations.outs[0];

    // 2. Compute product Ax + b
    component dense = Dense(inRows, outCols);
    dense.in <== in;
    dense.weights <== weights;
    dense.bias <== bias;

    component result[outCols];
    for (var i = 0; i < nOutputs; i++) {
        result[i] <== dense.out[i];
    }

    // 3. Compute hash of result
    component mimc_hash_activations = MiMCSponge(outCols, 220, 1);
    mimc_hash_activations.ins <== result;
    mimc_hash_activations.k <== 0;
    finalHash <== mimc_hash_activations.outs[0];
}