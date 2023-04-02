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
use pasta_curves::Fq;
use primitive_types::U256;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap, env::current_dir, fs, fs::File, path::PathBuf,
    time::{Duration, Instant},
    io::{BufReader, Write}, 
};

type C1 = CircomCircuit<<G1 as Group>::Scalar>;
type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;
type EE1 = nova_snark::provider::ipa_pc::EvaluationEngine<G1>;
type EE2 = nova_snark::provider::ipa_pc::EvaluationEngine<G2>;
type S1 = nova_snark::spartan::RelaxedR1CSSNARK<G1, EE1>;
type S2 = nova_snark::spartan::RelaxedR1CSSNARK<G2, EE2>;

const FWD_PASS_F: &str = "../models/json/PAD_inp1_two_conv_mnist.json";

const MIMC3D_R1CS_F: &str = "./circom/out/MiMC3D.r1cs";
const MIMC3D_WASM_F: &str = "./circom/out/MiMC3D.wasm";
const BACKBONE_R1CS_F: &str = "./circom/out/Backbone.r1cs";
// TODO:: PUT CPP WITNESS GEN BELOW, DONT NEED TO CHANGE ANYTHING ELSE 
const BACKBONE_WASM_F: &str = "./circom/out/Backbone.wasm";
const PROOF_OUT_F: &str = "./out/spartan_proof.json";

#[derive(Serialize)]
struct MiMC3DInput {
    dummy: String,
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
    x: Vec<Vec<Vec<i64>>>,
    head: ConvLayer,
    backbone: Vec<ConvLayer>,
    tail: DenseLayer,
    padding: usize,
    scale: f64,
    label: u64,
}

#[derive(Debug)]
struct RecursionInputs {
    all_private: Vec<HashMap<String, Value>>,
    start_pub_primary: Vec<F1>,
    start_pub_secondary: Vec<F2>,
}

#[derive(Serialize)]
struct StringifiedSpartanProof {
    nifs_primary: String,
    f_W_snark_primary: String,
    nifs_secondary: String,
    f_W_snark_secondary: String,
    zn_primary: String,
    zn_secondary: String
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

// On vesta curve
fn mimc3d(r1cs: &R1CS<F1>, wasm: &PathBuf, arr: Vec<Vec<Vec<i64>>>) -> BigInt {
    let witness_gen_input = PathBuf::from("circom_input.json");
    let witness_gen_output = PathBuf::from("circom_witness.wtns");

    let inp = MiMC3DInput {
        dummy: String::from("0"),
        arr: arr.clone(),
    };
    let input_json = serde_json::to_string(&inp).unwrap();
    fs::write(&witness_gen_input, input_json).unwrap();
    let witness = generate_witness_from_wasm::<<G1 as Group>::Scalar>(
        &wasm,
        &witness_gen_input,
        &witness_gen_output,
    );

    let circuit = CircomCircuit {
        r1cs: r1cs.clone(),
        witness: Some(witness),
    };
    let pub_outputs = circuit.get_public_outputs();

    fs::remove_file(witness_gen_input).unwrap();
    fs::remove_file(witness_gen_output).unwrap();

    let stripped = format!("{:?}", pub_outputs[0])
        .strip_prefix("0x")
        .unwrap()
        .to_string();
    BigInt::from_str_radix(&stripped, 16).unwrap()
}

fn rm_padding(arr: &Vec<Vec<Vec<i64>>>, padding: usize) -> Vec<Vec<Vec<i64>>> {
    let rows = arr.len() - padding * 2;
    let cols = arr[0].len() - padding * 2;

    let v = arr
        .iter()
        .map(|v| v[padding..padding + cols].to_vec())
        .collect::<Vec<Vec<Vec<i64>>>>();
    v[padding..padding + rows].to_vec()
}

/*
 * Constructs the inputs necessary for recursion. This includes 1) private
 * inputs for every step, and 2) initial public inputs for the first step of the
 * primary & secondary circuits.
 */
fn construct_inputs(
    fwd_pass: &ForwardPass,
    num_steps: usize,
    mimc3d_r1cs: &R1CS<F1>,
    mimc3d_wasm: &PathBuf,
) -> RecursionInputs {
    let mut private_inputs = Vec::new();
    for i in 0..num_steps {
        let a = if i > 0 {
            &fwd_pass.backbone[i - 1].a
        } else {
            &fwd_pass.head.a
        };
        let priv_in = HashMap::from([
            (String::from("a_prev"), json!(a)),
            (String::from("W"), json!(fwd_pass.backbone[i].W)),
            (String::from("b"), json!(fwd_pass.backbone[i].b)),
        ]);
        private_inputs.push(priv_in);
    }

    let v_1 = mimc3d(
        mimc3d_r1cs,
        mimc3d_wasm,
        rm_padding(&fwd_pass.head.a, fwd_pass.padding),
    )
    .to_str_radix(10);
    let z0_primary = vec![
        Fq::from(0),
        Fq::from_raw(U256::from_dec_str(&v_1).unwrap().0),
    ];

    // Secondary circuit is TrivialTestCircuit, filler val
    let z0_secondary = vec![F2::zero()];

    println!("- Done");
    RecursionInputs {
        all_private: private_inputs,
        start_pub_primary: z0_primary,
        start_pub_secondary: z0_secondary,
    }
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
) -> (RecursiveSNARK<G1, G2, C1, C2>, Duration, Duration) {
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
    let nova_snark_proving_time = start.elapsed();
    println!("- Done ({:?})", start.elapsed());

    println!("- Verifying RecursiveSNARK");
    println!("num_steps: {:?}", num_steps);
    println!("z0 PRIMARY: {:?}", inputs.start_pub_primary);
    println!("z0 SECONDARY: {:?}", inputs.start_pub_secondary);
    let start = Instant::now();
    let res = recursive_snark.verify(
        &pp,
        num_steps,
        inputs.start_pub_primary.clone(),
        inputs.start_pub_secondary.clone(),
    );
    let nova_snark_verifying_time = start.elapsed();
    assert!(res.is_ok());
    println!("- Output of final step: {:?}", res.unwrap().0);
    println!("- Done ({:?})", start.elapsed());

    (recursive_snark, nova_snark_proving_time, nova_snark_verifying_time)
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
    proof_f: &str,
) -> (CompressedSNARK<G1, G2, C1, C2, S1, S2>, Duration, Duration) {
    println!("- Generating");
    let start = Instant::now();
    type S1 = nova_snark::spartan::RelaxedR1CSSNARK<G1, EE1>;
    type S2 = nova_snark::spartan::RelaxedR1CSSNARK<G2, EE1>;
    let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &recursive_snark);
    assert!(res.is_ok());
    let spartan_proving_time = start.elapsed();
    println!("- Done ({:?})", start.elapsed());
    let compressed_snark = res.unwrap();

    println!("- Proof: {:?}", compressed_snark.f_W_snark_primary);
    let prf = StringifiedSpartanProof {
        nifs_primary: format!("{:?}", compressed_snark.nifs_primary),
        f_W_snark_primary: format!("{:?}", compressed_snark.f_W_snark_primary),
        nifs_secondary: format!("{:?}", compressed_snark.nifs_secondary),
        f_W_snark_secondary: format!("{:?}", compressed_snark.f_W_snark_secondary),
        zn_primary: format!("{:?}", compressed_snark.zn_primary),
        zn_secondary: format!("{:?}", compressed_snark.zn_secondary),
    };
    let prf_json = serde_json::to_string(&prf).unwrap();
    fs::write(&proof_f, prf_json).unwrap();

    println!("- Verifying");
    let start = Instant::now();
    let res = compressed_snark.verify(
        &pp,
        num_steps,
        inputs.start_pub_primary.clone(),
        inputs.start_pub_secondary.clone(),
    );
    assert!(res.is_ok());
    let spartan_verifying_time = start.elapsed();
    println!("- Done ({:?})", start.elapsed());

    (compressed_snark, spartan_proving_time, spartan_verifying_time)
}

fn main() {
    // Create a benchmark file
    let mut file = std::fs::File::create("../out/benchmark.csv").unwrap();
    file.write_all(b"num_layers,prover_time,verifier_time").unwrap();
    // This is the number of backbone layers, since we have a head & tail layer too, the 
    // total number of layers will be these values + 2
    // let arr_num_layers = [510, 254, 126, 63, 30, 14, 2];
    let arr_num_layers = [510];
    for num_layers in arr_num_layers {
        let root = current_dir().unwrap();
        let backbone_r1cs = load_r1cs(&root.join(BACKBONE_R1CS_F));
        let backbone_wasm = root.join(BACKBONE_WASM_F);
        let mimc3d_r1cs = load_r1cs(&root.join(MIMC3D_R1CS_F));
        let mimc3d_wasm = root.join(MIMC3D_WASM_F);
    
        let start = Instant::now();
    
        println!("== Loading forward pass");
        let fwd_pass = read_fwd_pass(FWD_PASS_F);
        let num_steps = fwd_pass.backbone.len();
        println!("==");
    
        println!("== Constructing inputs");
        let inputs = construct_inputs(&fwd_pass, num_steps, &mimc3d_r1cs, &mimc3d_wasm);
        println!("==");
    
        println!("== Creating circuit public parameters");
        let pp = setup(&backbone_r1cs);
        println!("==");
    
        println!("== Executing recursion using Nova");
        let (recursive_snark, nova_snark_proving_time, nova_snark_verifying_time) = recursion(backbone_wasm, backbone_r1cs, &inputs, &pp, num_steps);
        println!("==");
    
        println!("== Producing a CompressedSNARK using Spartan w/ IPA-PC");
        let (_compressed_snark, spartan_proving_time, spartan_verifying_time) = spartan(&pp, recursive_snark, num_steps, &inputs, PROOF_OUT_F);
        println!("==");
    
        println!("** Total time to completion: ({:?})", start.elapsed());

        let total_prover_time = nova_snark_proving_time + spartan_proving_time;
        let total_verifying_time = nova_snark_verifying_time + spartan_verifying_time;
        // Write the results to the benchmark file
        file.write_all(format!("{},{:?},{:?}\n", num_layers, total_prover_time, total_verifying_time).as_bytes()).unwrap();
    }
}