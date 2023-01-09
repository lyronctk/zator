use nova_scotia::{
    circom::{
        circuit::{CircomCircuit, R1CS},
        reader::load_r1cs,
    },
    create_public_params, create_recursive_circuit, F1, F2, G1, G2,
};
use nova_snark::{
    traits::{circuit::TrivialTestCircuit, Group},
    CompressedSNARK, PublicParams, RecursiveSNARK,
};
use num_bigint::BigInt;
use num_traits::Num;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap, env::current_dir, fs::File, io::BufReader,
    path::PathBuf, time::Instant,
};

type C1 = CircomCircuit<<G1 as Group>::Scalar>;
type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;
type S1 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G1>;
type S2 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G2>;

const R1CS_F: &str = "./circom/out/dense_layer.r1cs";
const WASM_F: &str = "./circom/out/dense_layer.wasm";
const FWD_PASS_F: &str = "../models/json/inp1_three_layer_mnist.json";

#[derive(Debug, Deserialize)]
struct ForwardPass {
    x: Vec<u32>,
    weights: Vec<Vec<Vec<i32>>>,
    biases: Vec<Vec<i32>>,
    scale: f64,
    activations: Vec<Vec<i32>>,
    label: u32
}

fn read_fwd_pass(f: &str) -> ForwardPass {
    let f = File::open(f).unwrap();
    let rdr = BufReader::new(f);
    println!("- Done");
    serde_json::from_reader(rdr).unwrap()
}

fn main() {
    let root = current_dir().unwrap();
    let r1cs = load_r1cs(&root.join(R1CS_F));
    let witness_gen = root.join(WASM_F);

    let start = Instant::now();

    println!("== Reading forward pass");
    let fwd_pass = read_fwd_pass(FWD_PASS_F);
    let num_steps = fwd_pass.activations.len();
    println!("==");

    println!("** Total time to completion: ({:?})", start.elapsed());
}
