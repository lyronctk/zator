pragma circom 2.1.1;
include "../../node_modules/circomlib-ml/circuits/ReLU.circom";
include "../../node_modules/circomlib-ml/circuits/circomlib/mimc.circom";
include "../../node_modules/circomlib-ml/circuits/Conv2D.circom";
include "mimcsponge.circom";
include "utils.circom";

template HeadLayer(nRows, nCols, nChannels, nFilters, kernelSize, strides) {
    signal input in_hash;
    signal input x[nRows][nCols][nChannels];
    var convLayerOutputRows = (nRows-kernelSize)\strides+1;
    var convLayerOutputCols = (nCols-kernelSize)\strides+1;
    var convLayerOutputDepth = nFilters;
    var convLayerOutputNumElements = convLayerOutputRows * convLayerOutputCols * convLayerOutputDepth;
    signal activations[convLayerOutputNumElements];
    signal output out;

    // 1. Verify that H(in) == in_hash
    component mimc_previous_activations = MimcHashMatrix3D(nRows, nCols, nChannels);
    mimc_previous_activations.matrix <== x;
    in_hash === mimc_previous_activations.hash;

    var W[kernelSize][kernelSize][nChannels][nFilters] = [
      [
        [
          [
            110095859,
            1968937
          ]
        ],
        [
          [
            163379773,
            143442363
          ]
        ],
        [
          [
            -85406423,
            -165240899
          ]
        ]
      ],
      [
        [
          [
            -123848811,
            318512261
          ]
        ],
        [
          [
            -203423515,
            153684407
          ]
        ],
        [
          [
            476588398,
            -46351999
          ]
        ]
      ],
      [
        [
          [
            17474651,
            347578406
          ]
        ],
        [
          [
            398471206,
            382686049
          ]
        ],
        [
          [
            512695074,
            -145270959
          ]
        ]
      ]
    ];
    
    var b[nFilters] = [
      528606474,
      517328024
    ];

    // 2. Generate Convolutional Network Output, Relu elements of 3D Matrix, and 
    // place the output into a flattened activations vector
    component convLayer = Conv2D(nRows, nCols, nChannels, nFilters, kernelSize, strides);
    convLayer.in <== x;
    convLayer.weights <== W;
    convLayer.bias <== b;

    component relu[convLayerOutputNumElements];
    // Now ReLu all of the elements in the 3D Matrix output of our Conv2D Layer
    // The ReLu'd outputs are stored in a flattened activations vector
    for (var row = 0; row < convLayerOutputRows; row++) {
        for (var col = 0; col < convLayerOutputCols; col++) {
            for (var depth = 0; depth < convLayerOutputDepth; depth++) {
                var indexFlattenedVector = (row * convLayerOutputCols * convLayerOutputDepth) + (col * convLayerOutputDepth) + depth;
                relu[indexFlattenedVector] = ReLU();
                relu[indexFlattenedVector].in <== convLayer.out[row][col][depth];
                activations[indexFlattenedVector] <== relu[indexFlattenedVector].out;
            }
        }
    }

    component mimc_hash_activations = MiMCSponge(convLayerOutputNumElements, 220, 1);
    mimc_hash_activations.ins <== activations;
    mimc_hash_activations.k <== 0;
    out <== mimc_hash_activations.outs[0];
}

component main { public [in_hash] } = HeadLayer(28, 28, 1, 2, 3, 1);
