pragma circom 2.1.1;

include "./utils/utils.circom";

template MiMC3D(H, W, D) {
    signal input dummy;
    signal output b;

    b <== dummy + 1;

    // signal input step_in[2];
    // signal output step_out[2];

    // signal input a;

    // step_out[0] <== step_in[0] + a;
    // step_out[1] <== step_in[1] + a;

    // signal input arr[H][W][D];
    // signal output h[3];

    // h[0] <== MimcHashMatrix3D(H, W, D)(arr);
    // h[1] <== 1;
    // h[2] <== 2;
}

component main { public [ dummy ] } = MiMC3D(28, 28, 4);

// component main { public [ arr ] } = MiMC3D(28, 28, 4);
