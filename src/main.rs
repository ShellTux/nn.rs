use nn::matrix;
use nn::{
    matrix::Matrix,
    nn::{NN, activation::Activation},
};

fn main() {
    let inputs = vec![
        matrix![0.0; 0.0],
        matrix![0.0; 1.0],
        matrix![1.0; 0.0],
        matrix![1.0; 1.0],
    ];

    let targets = vec![matrix![0.0], matrix![1.0], matrix![1.0], matrix![0.0]];

    let mut nn = NN::new(&[2, 3, 1], Activation::Sigmoid, 0.5);

    nn.train(&inputs, &targets, 100_000);

    for input in &inputs {
        println!("input:");
        println!("{}", input);
        println!("prediction:");
        println!("{}", nn.feed_forward(input));
    }

    println!("nn = {}", nn);
}
