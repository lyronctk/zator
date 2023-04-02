pragma circom 2.1.1;
include "./node_modules/circomlib-ml-vesta/circuits/ReLU.circom";
// include "./node_modules/circomlib-ml/circuits/circomlib/mimc.circom";
include "./node_modules/circomlib-ml-vesta/circuits/Conv2D.circom";
include "./utils/Division.circom";
include "./utils/mimcsponge.circom";
include "./utils/utils.circom";

// Template for an intermediary (backbone) Convnet layer
template Backbone(nRows, nCols, nChannels, nFilters, kernelSize, strides, padding) {
    var paddedRows = nRows + padding * 2;
    var paddedCols = nCols + padding * 2;

    // [running of hash outputted by the previous layer, hash of the activations of the previous layer]
    signal input step_in[2];

    // Activations outputted by the previous layer (note that it is padded)
    signal input a_prev[paddedRows][paddedCols][nChannels];
    // Weights for current layer
    signal input W[kernelSize][kernelSize][nChannels][nFilters];
    // Bias vector
    signal input b[nFilters];
    var convLayerOutputRows = (paddedRows-kernelSize)\strides+1;
    var convLayerOutputCols = (paddedCols-kernelSize)\strides+1;
    var convLayerOutputDepth = nFilters;
    signal activations[convLayerOutputRows][convLayerOutputCols][convLayerOutputDepth];
    signal weights_matrix_hash;
    signal bias_vector_hash;
    var scaleFactor = 10**16;
    // [running hash for curent layer, hash of activations for current layer]
    signal output step_out[2];

    // 1. Check that H(a_prev) = v_{a-1}
    //    v_{a-1} is H(a_{n-1})
    component mimc_previous_activations = MimcHashMatrix3D(convLayerOutputRows, convLayerOutputCols, nChannels);
    for (var i = 0; i < nRows; i++)
        for (var j = 0; j < nCols; j++)
            mimc_previous_activations.matrix[i][j] <== a_prev[i + padding][j + padding];
    step_in[1] === mimc_previous_activations.hash;

    // 2. Generate Convolutional Network Output, Relu elements of 3D Matrix, and 
    // place the output into a flattened activations vector
    component convLayer = Conv2D(paddedRows, paddedCols, nChannels, nFilters, kernelSize, strides);
    convLayer.in <== a_prev;
    convLayer.weights <== W;
    convLayer.bias <== b;

    component divs[convLayerOutputRows][convLayerOutputCols][convLayerOutputDepth];
    component relu[convLayerOutputRows][convLayerOutputCols][convLayerOutputDepth];
    // Now Relu all of the elements in the 3D Matrix output of our Conv2D Layer
    // The relu'd outputs are stored in a flattened activations vector
    for (var row = 0; row < convLayerOutputRows; row++) {
        for (var col = 0; col < convLayerOutputCols; col++) {
            for (var depth = 0; depth < convLayerOutputDepth; depth++) {
                relu[row][col][depth] = ReLU();
                relu[row][col][depth].in <== convLayer.out[row][col][depth];
                // Floor divide by the scale factor
                divs[row][col][depth] = Division(scaleFactor);
                divs[row][col][depth].dividend <== relu[row][col][depth].out;
                activations[row][col][depth] <== divs[row][col][depth].quotient;

            }
        }
    }

    // 3. Update running hash parameter p_{n+1}
    component mimc_weights_matrix = MimcHashMatrix4D(kernelSize, kernelSize, nChannels, nFilters);
    mimc_weights_matrix.matrix <== W;
    weights_matrix_hash <== mimc_weights_matrix.hash;

    component mimc_bias_vector = MiMCSponge(nFilters, 91, 1);
    mimc_bias_vector.ins <== b;
    mimc_bias_vector.k <== 0;
    bias_vector_hash <== mimc_bias_vector.outs[0];
    
    // Now p_{n+1} = Hash(p_n, Hash(Weights matrix), hash(bias vector))
    component pn_compositive_mimc = MiMCSponge(3, 91, 1);
    pn_compositive_mimc.ins[0] <== step_in[0];
    pn_compositive_mimc.ins[1] <== weights_matrix_hash;
    pn_compositive_mimc.ins[2] <== bias_vector_hash;
    pn_compositive_mimc.k <== 0;
    step_out[0] <== pn_compositive_mimc.outs[0];

    // 4. Compute v_{n+1} = hash of the activations for this layer
    component mimc_hash_activations = MimcHashMatrix3D(convLayerOutputRows, convLayerOutputCols, convLayerOutputDepth);
    mimc_hash_activations.matrix <== activations;
    step_out[1] <== mimc_hash_activations.hash;
}
