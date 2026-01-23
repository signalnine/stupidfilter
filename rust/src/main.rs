// stupidfilter - Rust port
// Original: Copyright 2008 Rarefied Technologies, Inc. (GPL v2)
// Rust port: 2026

use std::env;
use std::io::{self, Read};

mod features;
mod svm;

use features::extract_features;
use svm::SvmModel;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("usage: {} [model filename]", args[0]);
        std::process::exit(1);
    }

    let model_base = &args[1];

    // Load model and scale factors
    let model = match SvmModel::load(model_base) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            std::process::exit(1);
        }
    };

    // Read input from stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("Failed to read stdin");

    // Trim trailing whitespace (flex scanner behavior)
    let input = input.trim_end();

    // Extract features and predict
    let features = extract_features(input);
    let prediction = model.predict(&features);

    println!("{:.6}", prediction);
}
