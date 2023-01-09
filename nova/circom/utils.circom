pragma circom 2.1.1;

include "mimcsponge.circom";

// Template to mimc hash a matrix
template MimcHashMatrix(n) {
    signal input matrix[n][n];
    signal output hash;

    component mimc = MiMCSponge(n * n, 220, 1);
    mimc.k <== 0;

    for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) {
            mimc.ins[i * n + j] <== matrix[i][j];
        }
    }

    hash <== mimc.outs[0];
}