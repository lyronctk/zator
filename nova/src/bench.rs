/*
 * Verifies the forward pass of a CNN with a homogenous backbone. Does so using
 * Nova for IVC. Composed with Spartan to produce a final succinct proof.
 */
 use const_format::formatcp;
 use nova_scotia::{
     circom::{
         circuit::{CircomCircuit, R1CS},
         reader::{generate_witness_from_wasm, load_r1cs},
     },
     create_public_params, create_recursive_circuit, FileLocation, F1, F2, G1, G2, S1, S2,
 };
 use nova_snark::{
     traits::{circuit::TrivialTestCircuit, Group},
     CompressedSNARK, PublicParams, RecursiveSNARK,
 };
 
 use num_bigint::BigInt;
 use num_traits::Num;
 use primitive_types::U256;
 use serde::{Deserialize, Serialize};
 use serde_json::{json, Value};
 use std::{
     collections::HashMap, env::current_dir, fs, fs::File, path::PathBuf,
     time::{Duration, Instant},
     io::{BufReader, Write}
 };
 
 type C1 = CircomCircuit<<G1 as Group>::Scalar>;
 type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;
 
 const DEBUG: bool = false;
 
 const TRACE_NAME: &str = if DEBUG {
     "PADDED_trace_dim4_nlayers2.json"
 } else {
     "PADDED_trace_dim28_nlayers510.json"
 };
 const CIRCOM_PREFIX: &str = if DEBUG {
     "./circom/out_debug"
 } else {
     "./circom/out"
 };
 
 const FWD_PASS_F: &str = formatcp!("../models/json/{}", TRACE_NAME);
 const MIMC3D_R1CS_F: &str = formatcp!("{}/MiMC3D.r1cs", CIRCOM_PREFIX);
 const MIMC3D_WASM_F: &str = formatcp!("{}/MiMC3D.wasm", CIRCOM_PREFIX);
 const BACKBONE_R1CS_F: &str = formatcp!("{}/Backbone.r1cs", CIRCOM_PREFIX);
 const BACKBONE_F: &str = formatcp!("{}/Backbone", CIRCOM_PREFIX);
 
 #[derive(Serialize)]
 struct MiMC3DInput {
     dummy: String,
     arr: Vec<Vec<Vec<i64>>>,
 }
 
 #[derive(Debug, Deserialize)]
 struct ConvLayer {
     // dims: [kernelSize x kernelSize x nChannels x nFilters]
     W: Vec<Vec<Vec<Vec<i64>>>>,
     // dims: [nFilters]
     b: Vec<i64>,
     // dims: [nRows x nCols x nChannels]
     a: Vec<Vec<Vec<i64>>>,
 }
 
 #[derive(Debug, Deserialize)]
 struct DenseLayer {
     // dims: [nInputs x nOutput]
     W: Vec<Vec<i64>>,
     // dims: [nOutputs]
     b: Vec<i64>,
     // dims: [nOutputs]
     a: Vec<i64>,
 }
 
 #[derive(Debug, Deserialize)]
 struct ForwardPass {
     // dims: [nRows x nCols x nChannels]
     x: Vec<Vec<Vec<i64>>>,
     head: ConvLayer,
     backbone: Vec<ConvLayer>,
     tail: DenseLayer,
     // For convolution
     padding: usize,
     // Scale factor
     scale: f64,
     // Which run
     label: u64,
 }
 
 #[derive(Debug)]
 struct RecursionInputs {
     all_private: Vec<HashMap<String, Value>>,
     start_pub_primary: Vec<F1>,
     start_pub_secondary: Vec<F2>,
 }
 
 /*
  * Load in the forward pass (i.e. parameters and inputs/outputs for each layer).
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
  * Computes the MiMC hash of an input 3D array. Used to satisfy input hash
  * check for initial backbone layer.
  */
 fn mimc3d(r1cs: &R1CS<F1>, wasm: PathBuf, arr: Vec<Vec<Vec<i64>>>) -> BigInt {
     let witness_gen_output = PathBuf::from("circom_witness.wtns");
 
     let inp = MiMC3DInput {
         dummy: String::from("0"),
         arr: arr.clone(),
     };
     let input_json = serde_json::to_string(&inp).unwrap();
     let witness = generate_witness_from_wasm::<<G1 as Group>::Scalar>(
         &FileLocation::PathBuf(wasm),
         &input_json,
         &witness_gen_output,
     );
 
     let circuit = CircomCircuit {
         r1cs: r1cs.clone(),
         witness: Some(witness),
     };
     let pub_outputs = circuit.get_public_outputs();
     fs::remove_file(witness_gen_output).unwrap();
 
     let stripped = format!("{:?}", pub_outputs[0])
         .strip_prefix("0x")
         .unwrap()
         .to_string();
     BigInt::from_str_radix(&stripped, 16).unwrap()
 }
 
 /*
  * Utility function for removing the 1 element padding around activation
  * volumes.
  */
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
     mimc3d_wasm: PathBuf,
 ) -> RecursionInputs {
     let mut private_inputs = Vec::new();
     if num_steps == 510 {
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
     } else if num_steps == 255 {
        for i in (0..510).step_by(2) {
            let a_prev_1 = if i > 0 {
                &fwd_pass.backbone[i - 1].a
            } else {
                &fwd_pass.head.a
            };
            let a_prev_2 = &fwd_pass.backbone[i].a;
            let priv_in = HashMap::from([
                (String::from("a_prev_1"), json!(a_prev_1)),
                (String::from("W_1"), json!(fwd_pass.backbone[i].W)),
                (String::from("b_1"), json!(fwd_pass.backbone[i].b)),
                (String::from("a_prev_2"), json!(a_prev_2)),
                (String::from("W_2"), json!(fwd_pass.backbone[i + 1].W)),
                (String::from("b_2"), json!(fwd_pass.backbone[i + 1].b)),
            ]);
            private_inputs.push(priv_in);
        }
    } else { // num_steps = 170
            for i in (0..510).step_by(3) {
                let a_prev_1 = if i > 0 {
                    &fwd_pass.backbone[i - 1].a
                } else {
                    &fwd_pass.head.a
                };
                let a_prev_2 = &fwd_pass.backbone[i].a;
                let a_prev_3 = &fwd_pass.backbone[i + 1].a;
                let priv_in = HashMap::from([
                    (String::from("a_prev_1"), json!(a_prev_1)),
                    (String::from("W_1"), json!(fwd_pass.backbone[i].W)),
                    (String::from("b_1"), json!(fwd_pass.backbone[i].b)),
                    (String::from("a_prev_2"), json!(a_prev_2)),
                    (String::from("W_2"), json!(fwd_pass.backbone[i + 1].W)),
                    (String::from("b_2"), json!(fwd_pass.backbone[i + 1].b)),
                    (String::from("a_prev_3"), json!(a_prev_3)),
                    (String::from("W_3"), json!(fwd_pass.backbone[i + 2].W)),
                    (String::from("b_3"), json!(fwd_pass.backbone[i + 2].b)),
                ]);
                private_inputs.push(priv_in);
        }
     }
 
     let v_1 = mimc3d(
         mimc3d_r1cs,
         mimc3d_wasm,
         rm_padding(&fwd_pass.head.a, fwd_pass.padding),
     )
     .to_str_radix(10);
     let z0_primary = vec![
         F1::from(0),
         F1::from_raw(U256::from_dec_str(&v_1).unwrap().0),
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
         FileLocation::PathBuf(witness_gen),
         r1cs,
         inputs.all_private.clone(),
         inputs.start_pub_primary.clone(),
         &pp,
     )
     .unwrap();
     let nova_snark_proving_time = start.elapsed();
     println!("- Done ({:?})", start.elapsed());
 
     println!("- Verifying RecursiveSNARK");
     println!("  - num_steps: {:?}", num_steps);
     println!("  - z0 PRIMARY: {:?}", inputs.start_pub_primary);
     println!("  - z0 SECONDARY: {:?}", inputs.start_pub_secondary);
     let start = Instant::now();
     let res = recursive_snark.verify(
         &pp,
         num_steps,
         inputs.start_pub_primary.clone(),
         inputs.start_pub_secondary.clone(),
     );
     let nova_snark_verifying_time = start.elapsed();
     assert!(res.is_ok());
     println!("  - Output of final step: {:?}", res.unwrap().0);
     println!("- Done ({:?})", start.elapsed());
 
     (recursive_snark, nova_snark_proving_time, nova_snark_verifying_time)
 }
 
 /*
  * Uses Spartan w/ IPA-PC to prove knowledge of a valid Nova IVC proof.
  * Composition is for producing a final succinct proof of size log|C|.
  */
 fn spartan(
     pp: &PublicParams<G1, G2, C1, C2>,
     recursive_snark: RecursiveSNARK<G1, G2, C1, C2>,
     num_steps: usize,
     inputs: &RecursionInputs
 ) -> (CompressedSNARK<G1, G2, C1, C2, S1, S2>, Duration, Duration) {
     println!("- Generating");
     let start = Instant::now();
     let (pk, vk) = CompressedSNARK::<_, _, _, _, S1, S2>::setup(&pp).unwrap();
     let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark);
 
     assert!(res.is_ok());
     let spartan_proving_time = start.elapsed();
     println!("- Done ({:?})", start.elapsed());
     let compressed_snark = res.unwrap();
 
     println!("- Verifying");
     let start = Instant::now();
     let res = compressed_snark.verify(
         &vk,
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
    let mut file = std::fs::File::create("./out/benchmark.csv").unwrap();
    file.write_all(b"Number of Recrusive Steps, Number of Layers per Step, Prover Time, Verifier Time").unwrap();
    let mut num_layers_to_backbone_r1cs = HashMap::new();
    num_layers_to_backbone_r1cs.insert(1, formatcp!("{}/Backbone1.r1cs", CIRCOM_PREFIX));
    num_layers_to_backbone_r1cs.insert(2, formatcp!("{}/Backbone2.r1cs", CIRCOM_PREFIX));
    num_layers_to_backbone_r1cs.insert(3, formatcp!("{}/Backbone3.r1cs", CIRCOM_PREFIX));
    let mut num_layers_to_backbone = HashMap::new();
    num_layers_to_backbone.insert(1, formatcp!("{}/Backbone1", CIRCOM_PREFIX));
    num_layers_to_backbone.insert(2, formatcp!("{}/Backbone2", CIRCOM_PREFIX));
    num_layers_to_backbone.insert(3, formatcp!("{}/Backbone3", CIRCOM_PREFIX));

    // Number of layers per step
    let arr_num_layers = [1, 2, 3];
    for num_layers in arr_num_layers {
        let root = current_dir().unwrap();
        let backbone_r1cs = load_r1cs(&FileLocation::PathBuf(root.join(num_layers_to_backbone_r1cs.get(&num_layers).unwrap())));
        let backbone_witgen = root.join(num_layers_to_backbone.get(&num_layers).unwrap());
        let mimc3d_r1cs = load_r1cs(&FileLocation::PathBuf(root.join(MIMC3D_R1CS_F)));
        let mimc3d_wasm = root.join(MIMC3D_WASM_F);
    
        let start = Instant::now();
    
        println!("== Loading forward pass");
        let fwd_pass = read_fwd_pass(FWD_PASS_F);
        let num_steps = fwd_pass.backbone.len() / num_layers;
        println!("==");
    
        println!("== Constructing inputs");
        let inputs = construct_inputs(&fwd_pass, num_steps, &mimc3d_r1cs, mimc3d_wasm);
        println!("==");
    
        println!("== Creating circuit public parameters");
        let pp = setup(&backbone_r1cs);
        println!("==");
    
        println!("== Executing recursion using Nova");
        let (recursive_snark, nova_snark_proving_time, nova_snark_verifying_time) = recursion(backbone_witgen, backbone_r1cs, &inputs, &pp, num_steps);
        println!("==");
    
        println!("== Producing a CompressedSNARK using Spartan w/ IPA-PC");
        let (_compressed_snark, spartan_proving_time, spartan_verifying_time) = spartan(&pp, recursive_snark, num_steps, &inputs);
        println!("==");
    
        println!("** Total time to completion: ({:?})", start.elapsed());

        let total_prover_time = nova_snark_proving_time + spartan_proving_time;
        let total_verifying_time = nova_snark_verifying_time + spartan_verifying_time;
        // Write the results to the benchmark file
        file.write_all(format!("{},{},{:?},{:?}\n", num_steps, num_layers, total_prover_time, total_verifying_time).as_bytes()).unwrap();
    }
 } 