import os
import json
import pyperclip

import numpy as np

# simple script to write in massive weights matrix to the circom file / clipboard...
DEBUG = True

SCALE = 1e-16
PADDING = 1
EPOCHS = 10
DIMS = 28
N_BACKBONE_LAYERS = 510
if DEBUG:
    EPOCHS = 1
    DIMS = 4
    N_BACKBONE_LAYERS = 2
TRACE_F = f"json/trace_dim{DIMS}_nlayers{N_BACKBONE_LAYERS}.json"
OUT_F = f"json/PADDED_trace_dim{DIMS}_nlayers{N_BACKBONE_LAYERS}.json"


def copy_data():
    # open tail_layer_smoke_test.json
    with open(TRACE_F) as f:
        data = json.load(f)
        # print(data["tail"]["W"])

        circom = """pragma circom 2.1.1;

include "../../node_modules/circomlib-ml/circuits/Dense.circom";
include "mimcsponge.circom";

template TailLayer() {
    var inRows = 3136;
    var outCols = 10;

    signal input prevHash;
    signal input in[inRows];

    var activations[outCols] = [-28639993667, -34677528381, -21898008346, -19130430221, -40692321777, -30850053787, -43104949951, -6874659061, -25667932510, -25357667922];
    var bias[outCols] = [-16762316, 79053082, -54210544, -95354311, 56738410, -37802950, 4387544, 4411432, -51621947, -40832181];
    var weights[inRows][outCols] = ?;

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

    signal result[outCols];
    for (var i = 0; i < outCols; i++) {
        result[i] <== dense.out[i];
    }

    // 3. Compute hash of result
    component mimc_hash_activations = MiMCSponge(outCols, 220, 1);
    mimc_hash_activations.ins <== result;
    mimc_hash_activations.k <== 0;
    finalHash <== mimc_hash_activations.outs[0];
}

component main { public [prevHash, in] } = TailLayer();"""

        c = circom.replace("?", str(data["tail"]["W"]))

        data_np = np.array(data["tail"]["W"])

        # save c to clipboard
        pyperclip.copy(c)

        print(c)


def smoke():
    arr = "["
    for i in range(3136):
        arr += "0"
        if i != 3135:
            arr += ","
    arr += "]"
    pyperclip.copy(arr)


def pad_json():
    with open(TRACE_F) as f:
        data = json.load(f)

        data["x"] = np.pad(
            np.array(data["x"]), ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0).tolist()

        head_activ = np.array(data["head"]["a"])
        backbone = data["backbone"]

        head_activ = np.pad(
            head_activ, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)
        relud = np.maximum(0, head_activ)
        data["head"]["a"] = relud.tolist()
        print(relud.shape)
        print(relud)

        for idx, bone in enumerate(backbone):
            bone_activ = np.array(bone["a"]).reshape((DIMS, DIMS, 2))
            bone_activ = np.pad(
                bone_activ, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)
            relud = np.maximum(0, bone_activ)
            backbone[idx]["a"] = relud.tolist()
            print(bone_activ.shape)

        with open(OUT_F, 'w') as outfile:
            json.dump(data, outfile)


pad_json()
copy_data()
