pragma circom 2.1.1;

include "mimcsponge.circom";

// Template to mimc hash a matrix
template mimc_hash_matrix(n, hash) {
    signal input weights[n][n];
    signal output hash;

    component mimc = MiMCSponge(n * n, 220, 1);
    mimc.k <== 0;

    for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) {
            mimc.in[i * n + j] <== weights[i][j];
        }
    }

    hash <== mimc.out[0];
}