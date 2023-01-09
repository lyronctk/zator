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
    x: Vec<u64>,
    weights: Vec<Vec<Vec<i64>>>,
    biases: Vec<Vec<i64>>,
    scale: f64,
    activations: Vec<Vec<i64>>,
    label: u64
}

struct RecursionInputs {
    all_private: Vec<HashMap<String, Value>>,
    start_pub_primary: Vec<F1>,
    start_pub_secondary: Vec<F2>,
}

fn read_fwd_pass(f: &str) -> ForwardPass {
    let f = File::open(f).unwrap();
    let rdr = BufReader::new(f);
    println!("- Working");
    serde_json::from_reader(rdr).unwrap()
}

fn setup(r1cs: &R1CS<F1>) -> PublicParams<G1, G2, C1, C2> {
    let pp = create_public_params(r1cs.clone());

    println!(
        "- Number of constraints per step (primary): {}",
        pp.num_constraints().0
    );
    println!(
        "- Number of constraints per step (secondary): {}",
        pp.num_constraints().1
    );

    pp
}

fn construct_inputs(
    fwd_pass: &ForwardPass,
    num_steps: usize,
) {
    // let mut private_inputs = Vec::new();
    // for i in 0..num_steps {
    //     let inp = ;
    //     let priv_in = HashMap::from([
    //         (String::from("A"), json!(fwd_pass.weights[i])),
    //         (String::from("b"), json!(fwd_pass.biases[i])),
    //         (String::from("x"), json!(solved_maze.width)),
    //         (String::from("move"), json!([dr, dc])),
    //     ]);
    //     private_inputs.push(priv_in);
    // }

    // // [TODO] Checkings the grid hash currently disabled. Need to run poseidon
    // //        hash on the Vesta curve. Also need to load into F1, which only has
    // //        From() trait implemented for u64.
    // let z0_primary = vec![
    //     F1::from(123),
    //     F1::from(0),
    //     F1::from(0),
    //     F1::from(solved_maze.maze[0][0] as u64),
    // ];

    // // Secondary circuit is TrivialTestCircuit, filler val
    // let z0_secondary = vec![F2::zero()];

    // println!("- Done");
    // RecursionInputs {
    //     all_private: private_inputs,
    //     start_pub_primary: z0_primary,
    //     start_pub_secondary: z0_secondary,
    // }
}

fn main() {
    let root = current_dir().unwrap();
    // let r1cs = load_r1cs(&root.join(R1CS_F));
    // let witness_gen = root.join(WASM_F);

    let start = Instant::now();

    println!("== Loading forward pass");
    let fwd_pass = read_fwd_pass(FWD_PASS_F);
    let num_steps = fwd_pass.activations.len() + 1;
    println!("==");

    println!("== Creating circuit public parameters");
    // let pp = setup(&r1cs);
    println!("==");

    println!("== Constructing inputs");
    // let inputs = construct_inputs(&fwd_pass, num_steps);
    println!("==");


    println!("** Total time to completion: ({:?})", start.elapsed());
}
