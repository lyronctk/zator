use serde::Deserialize;
use serde_json::{json, Value};
use std::{fs::File, io::BufReader};

const FWD_PASS_F: &str = "../models/json/inp1_three_layer_mnist.json";

#[derive(Debug, Deserialize)]
struct ForwardPass {
    fwd_in: Vec<u32>,
    weights: Vec<Vec<Vec<i32>>>,
    biases: Vec<Vec<i32>>,
    scale: f64,
    activations: Vec<Vec<i32>>,
    label: u32,
    fwd_out: ForwardOut
}

fn read_fwd_pass(f: &str) -> ForwardPass {
    let f = File::open(f).unwrap();
    let rdr = BufReader::new(f);
    println!("- Done");
    serde_json::from_reader(rdr).unwrap()
}

fn main() {
    let fwd_pass = read_fwd_pass(FWD_PASS_F);
    println!("{:?}", fwd_pass);
    println!("Hello, world!");
}
