use crate::matrix::Matrix;
use activation::Activation;
use std::fmt;

pub mod activation;

#[derive(Debug, Clone)]
pub struct NN {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation,
    learning_rate: f64,
}

impl NN {
    pub fn new(layers: &[usize], activation: Activation, learning_rate: f64) -> Self {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Self {
            layers: layers.to_vec(),
            weights,
            biases,
            data: vec![],
            activation,
            learning_rate,
        }
    }

    pub fn feed_forward(&mut self, inputs: &Matrix) -> Matrix {
        let inputs = inputs.clone();

        assert!(
            self.layers[0] == inputs.data.len(),
            "Invalid Number of Inputs"
        );

        let mut current = inputs;

        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .matmul(&current)
                .add(&self.biases[i])
                .map(|x| self.activation.activate(x));

            self.data.push(current.clone());
        }

        current
    }

    pub fn back_propogate(&mut self, inputs: Matrix, targets: &Matrix) {
        let targets = targets.clone();

        let mut errors = targets.sub(&inputs);

        let mut gradients = inputs.clone().map(|x| self.activation.derivative(x));

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.hadamard(&errors).map(|x| x * 0.5); // learning rate

            self.weights[i] = self.weights[i].add(&gradients.matmul(&self.data[i].transpose()));

            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].transpose().matmul(&errors);
            gradients = self.data[i].map(|x| self.activation.derivative(x));
        }
    }

    pub fn train(&mut self, inputs: &[Matrix], targets: &[Matrix], epochs: u32) {
        let inputs = inputs.to_owned();
        let targets = targets.to_owned();

        for i in 1..=epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
                let percentage = i as f32 / epochs as f32 * 100.;
                println!("Epoch {} of {} ({:.1}%)", i, epochs, percentage);
            }

            for (input, target) in inputs.iter().zip(targets.iter()) {
                let outputs = self.feed_forward(input);
                self.back_propogate(outputs, target);
            }
        }
    }
}

impl fmt::Display for NN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Number of layers
        writeln!(f, "Neural Network:")?;
        writeln!(f, "  Layers: {:?}", self.layers)?;

        // Number of weights and biases
        writeln!(f, "  Number of weight matrices: {}", self.weights.len())?;
        writeln!(f, "  Number of bias matrices: {}", self.biases.len())?;

        // Dimensions of each weight matrix
        writeln!(f, "  Weights and biases dimensions:")?;
        for (i, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            // Assuming Matrix has a way to get dimensions, e.g., rows() and cols()
            writeln!(
                f,
                "    Layer {}: weight shape: {}x{}, bias shape: {}x{}",
                i, w.rows, w.cols, b.rows, b.cols
            )?;
        }

        // Activation function
        // Assuming Activation implements Display
        writeln!(f, "  Activation: {:?}", self.activation)?;

        // Learning rate with fixed formatting
        writeln!(f, "  Learning rate: {:.4}", self.learning_rate)?;

        Ok(())
    }
}
