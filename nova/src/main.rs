use mimc_rs::Mimc7;
use nova_scotia::{
    circom::{
        circuit::{CircomCircuit, R1CS},
        reader::{generate_witness_from_wasm, load_r1cs},
    },
    create_public_params, create_recursive_circuit, F1, F2, G1, G2,
};
use nova_snark::{
    traits::{circuit::TrivialTestCircuit, Group},
    CompressedSNARK, PublicParams, RecursiveSNARK,
};
use num_bigint::BigInt;
use num_traits::Num;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap, env::current_dir, fs, fs::File, io::BufReader, path::PathBuf,
    time::Instant,
};

type C1 = CircomCircuit<<G1 as Group>::Scalar>;
type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;
type S1 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G1>;
type S2 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G2>;

const R1CS_F: &str = "./circom/out/dense_layer.r1cs";
const WASM_F: &str = "./circom/out/dense_layer.wasm";
const FWD_PASS_F: &str = "../models/json/inp1_two_conv_mnist.json";

#[derive(Serialize)]
struct MiMC3DInput {
    arr: Vec<Vec<Vec<i64>>>,
}

#[derive(Debug, Deserialize)]
struct ConvLayer {
    // NOTE: Non-standard transposes on matrices
    // dims: [kernelSize x kernelSize x nChannels x nFilters]
    W: Vec<Vec<Vec<Vec<i64>>>>,
    // dims: [nFilters]
    b: Vec<i64>,
    // dims: [nRows x nCols x nChannels]
    a: Vec<Vec<Vec<i64>>>,
}

#[derive(Debug, Deserialize)]
struct DenseLayer {
    // NOTE: Non-standard transposes on matrices
    // dims: [nInputs x nOutput], note non-standard transpose
    W: Vec<Vec<i64>>,
    // dims: [nOutputs]
    b: Vec<i64>,
    // dims: [nOutputs]
    a: Vec<i64>,
}

#[derive(Debug, Deserialize)]
struct ForwardPass {
    x: Vec<u64>,
    head: ConvLayer,
    backbone: Vec<ConvLayer>,
    tail: DenseLayer,
    scale: f64,
    label: u64,
}

struct RecursionInputs {
    all_private: Vec<HashMap<String, Value>>,
    start_pub_primary: Vec<F1>,
    start_pub_secondary: Vec<F2>,
}

/*
 * Read in the forward pass (i.e. parameters and inputs/outputs for each layer).
 */
fn read_fwd_pass(f: &str) -> ForwardPass {
    let f = File::open(f).unwrap();
    let rdr = BufReader::new(f);
    println!("- Working");
    serde_json::from_reader(rdr).unwrap()
}

/*
 * Generates public parameters for Nova.
 */
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

/*
 * Constructs the inputs necessary for recursion. This includes 1) private
 * inputs for every step, and 2) initial public inputs for the first step of the
 * primary & secondary circuits.
 */
fn construct_inputs(fwd_pass: &ForwardPass, num_steps: usize) {
    let mut private_inputs = Vec::new();
    for i in 0..num_steps {
        let a = if i > 0 {
            &fwd_pass.backbone[i - 1].a
        } else {
            &fwd_pass.head.a
        };
        let priv_in = HashMap::from([
            (String::from("a"), json!(a)),
            (String::from("W"), json!(fwd_pass.backbone[i].W)),
            (String::from("b"), json!(fwd_pass.backbone[i].b)),
        ]);
        private_inputs.push(priv_in);
    }

    let root = current_dir().unwrap();
    let r1cs = load_r1cs(&root.join("./circom/out/MiMC3D.r1cs"));
    let witness_generator_file = root.join("./circom/out/MiMC3D.wasm");
    let witness_generator_input = root.join("circom_input.json");
    let witness_generator_output = root.join("circom_witness.wtns");
    let inp = MiMC3DInput {
        arr: fwd_pass.head.a.clone()
    };
    let input_json = serde_json::to_string(&inp).unwrap();
    fs::write(&witness_generator_input, input_json).unwrap();
    let witness = generate_witness_from_wasm::<<G1 as Group>::Scalar>(
        &witness_generator_file,
        &witness_generator_input,
        &witness_generator_output,
    );
    let circuit = CircomCircuit {
        r1cs: r1cs.clone(),
        witness: Some(witness),
    };
    let current_public_output = circuit.get_public_outputs();
    println!("output: {:?}", current_public_output.len());

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

/*
 * Uses Nova's folding scheme to produce a single relaxed R1CS instance that,
 * when satisfied, proves the proper execution of every step in the recursion.
 * Can be thought of as a pre-processing step for the final SNARK.
 */
fn recursion(
    witness_gen: PathBuf,
    r1cs: R1CS<F1>,
    inputs: &RecursionInputs,
    pp: &PublicParams<G1, G2, C1, C2>,
    num_steps: usize,
) -> RecursiveSNARK<G1, G2, C1, C2> {
    println!("- Creating RecursiveSNARK");
    let start = Instant::now();
    let recursive_snark = create_recursive_circuit(
        witness_gen,
        r1cs,
        inputs.all_private.clone(),
        inputs.start_pub_primary.clone(),
        &pp,
    )
    .unwrap();
    println!("- Done ({:?})", start.elapsed());

    println!("- Verifying RecursiveSNARK");
    let start = Instant::now();
    let res = recursive_snark.verify(
        &pp,
        num_steps,
        inputs.start_pub_primary.clone(),
        inputs.start_pub_secondary.clone(),
    );
    assert!(res.is_ok());
    println!("- Output of final step: {:?}", res.unwrap().0);
    println!("- Done ({:?})", start.elapsed());

    recursive_snark
}

/*
 * Uses Spartan w/ IPA-PC to prove knowledge of the output of Nova (a satisfied
 * relaxed R1CS instance) in a proof that can be verified with sub-linear cost.
 */
fn spartan(
    pp: &PublicParams<G1, G2, C1, C2>,
    recursive_snark: RecursiveSNARK<G1, G2, C1, C2>,
    num_steps: usize,
    inputs: &RecursionInputs,
) -> CompressedSNARK<G1, G2, C1, C2, S1, S2> {
    println!("- Generating");
    let start = Instant::now();
    type S1 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G1>;
    type S2 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G2>;
    let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &recursive_snark);
    assert!(res.is_ok());
    println!("- Done ({:?})", start.elapsed());
    let compressed_snark = res.unwrap();
    println!("- Proof: {:?}", compressed_snark.f_W_snark_primary);

    println!("- Verifying");
    let start = Instant::now();
    let res = compressed_snark.verify(
        &pp,
        num_steps,
        inputs.start_pub_primary.clone(),
        inputs.start_pub_secondary.clone(),
    );
    assert!(res.is_ok());
    println!("- Done ({:?})", start.elapsed());

    compressed_snark
}

fn main() {
    let root = current_dir().unwrap();
    // let r1cs = load_r1cs(&root.join(R1CS_F));
    // let witness_gen = root.join(WASM_F);

    let start = Instant::now();

    println!("== Loading forward pass");
    let fwd_pass = read_fwd_pass(FWD_PASS_F);
    let num_steps = fwd_pass.backbone.len();
    println!("==");

    println!("== Creating circuit public parameters");
    // let pp = setup(&r1cs);
    println!("==");

    println!("== Constructing inputs");
    let inputs = construct_inputs(&fwd_pass, num_steps);
    println!("==");

    println!("== Executing recursion using Nova");
    // let recursive_snark = recursion(witness_gen, r1cs, &inputs, &pp, num_steps);
    println!("==");

    println!("== Producing a CompressedSNARK using Spartan w/ IPA-PC");
    // let _compressed_snark = spartan(&pp, recursive_snark, num_steps, &inputs);
    println!("==");

    println!("** Total time to completion: ({:?})", start.elapsed());
}
