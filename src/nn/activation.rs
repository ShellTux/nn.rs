use std::f64::consts::E;

#[derive(Debug, Clone)]
pub enum Activation {
    Sigmoid,
    // Others can be added, e.g., ReLU
}

impl Activation {
    pub fn activate(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => 1. / (1. + E.powf(-x)),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => x * (1. - x),
        }
    }
}
