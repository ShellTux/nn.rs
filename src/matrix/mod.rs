use rand::{Rng, rng};
use std::fmt;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

#[macro_export]
macro_rules! matrix {
    ( $( $($val:expr),+ );* $(;)? ) => {
        {
            let mut data = Vec::<f64>::new();
            let mut rows = 0;
            let mut cols = 0;
            $(
                let row_data = vec![$($val),+];
                data.extend(row_data);
                rows += 1;
                let row_len = vec![$($val),+].len();
                if cols == 0 {
                    cols = row_len;
                } else if cols != row_len {
                    panic!("Inconsistent number of elements in the matrix rows");
                }
            )*

            Matrix { rows, cols, data }
        }
    };
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::zeros(rows, cols)
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rng();

        let data = (0..rows * cols)
            .map(|_| rng.random_range(0.0..1.0))
            .collect();

        Self { rows, cols, data }
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    pub fn zip<F>(&self, other: &Self, f: F) -> Self
    where
        F: FnMut((&f64, &f64)) -> f64,
    {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(other.data.iter()).map(f).collect(),
        }
    }

    pub fn matmul(&self, other: &Self) -> Self {
        if self.cols != other.rows {
            dbg!(self);
            dbg!(other);

            panic!(
                "Attempted to add matrix of incorrect dimensions left: {:?}, right: {:?}",
                (self.rows, self.cols),
                (other.rows, other.cols)
            );
        }

        let mut result = Self::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;

                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }

                result.set(i, j, sum);
            }
        }

        result
    }

    pub fn add(&self, other: &Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Attempted to add matrix of incorrect dimensions left: {:?}, right: {:?}",
                (self.rows, self.cols),
                (other.rows, other.cols)
            );
        }

        self.zip(other, |(a, b)| a + b)
    }

    pub fn sub(&self, other: &Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Attempted to add matrix of incorrect dimensions left: {:?}, right: {:?}",
                (self.rows, self.cols),
                (other.rows, other.cols)
            );
        }

        self.zip(other, |(a, b)| a - b)
    }

    pub fn scalar_mul(&self, scalar: f64) -> Self {
        self.map(|x| x * scalar)
    }

    pub fn hadamard(&self, other: &Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Attempted to add matrix of incorrect dimensions left: {:?}, right: {:?}",
                (self.rows, self.cols),
                (other.rows, other.cols)
            );
        }

        self.zip(other, |(a, b)| a * b)
    }

    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let mut m = self.clone();

        for val in &mut m.data {
            *val = f(*val);
        }

        m
    }

    pub fn apply<F>(&self, func: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        self.map(func)
    }

    pub fn transpose(&self) -> Self {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let width = 7 * self.cols + 1;

        writeln!(f, "┌ {} ┐", " ".repeat(width))?;
        for row in 0..self.rows {
            for col in 0..self.cols {
                let index = row * self.cols + col;

                if col == 0 {
                    write!(f, "│ ")?;
                }

                write!(f, " {:5.3} ", self.data[index])?;
            }

            writeln!(f, "  │ ")?;
        }

        write!(f, "└ {} ┘", " ".repeat(width))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_random_matrix() {
        let rows = 3;
        let cols = 4;
        let matrix = Matrix::random(rows, cols);

        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.cols, cols);
        assert_eq!(matrix.data.len(), rows * cols);

        for &num in &matrix.data {
            assert!(0. <= num && num < 1.0);
        }
    }

    #[test]
    fn test_elementwise_multiply() {
        // Create two matrices for testing
        let matrix1 = matrix![1., 2., 3., 4.];
        let matrix2 = matrix![5., 6., 7., 8.];

        assert_eq!(matrix1.rows, 1);

        let result = matrix1.hadamard(&matrix2);

        // Define the expected result
        let expected_result = matrix![5., 12., 21., 32.];

        // Check if the actual result matches the expected result
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_subtract_same_dimensions() {
        let matrix1 = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];

        let matrix2 = matrix![
            5.0, 6.0;
            7.0, 8.0
        ];

        let result = matrix1.sub(&matrix2);

        let expected = matrix![
            -4.0, -4.0;
            -4.0, -4.0
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_dot_multiply() {
        let a = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0
        ];
        let b = matrix![
            7.0, 8.0;
            9.0, 10.0;
            11.0, 12.0
        ];

        let result = a.matmul(&b);

        let expected_result = matrix![
            58.0, 64.0;
            139.0, 154.0
        ];

        assert_eq!(result, expected_result);
    }

    #[test]
    #[should_panic]
    fn test_subtract_different_dimensions() {
        let matrix1 = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];

        let matrix2 = matrix![
            5.0, 6.0, 7.0;
            8.0, 9.0, 10.0
        ];

        let _ = matrix1.sub(&matrix2);
    }

    #[test]
    fn test_matrix_addition() {
        let a = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0
        ];

        let b = matrix![
            5.0, 6.0, 7.0;
            8.0, 9.0, 10.0;
            11.0, 12.0, 13.0
        ];

        let expected_result = matrix![
            6.0, 8.0, 10.0;
            12.0, 14.0, 16.0;
            18.0, 20.0, 22.0
        ];

        let result = a.add(&b);

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_transpose_2x2() {
        let matrix = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];
        let transposed = matrix.transpose();

        let expected = matrix![
            1.0, 3.0;
            2.0, 4.0
        ];
        assert_eq!(transposed, expected);
    }

    #[test]
    fn test_transpose_3x3() {
        let matrix = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0
        ];
        let transposed = matrix.transpose();

        let expected = matrix![
            1.0, 4.0, 7.0;
            2.0, 5.0, 8.0;
            3.0, 6.0, 9.0
        ];
        assert_eq!(transposed, expected);
    }

    #[test]
    fn test_transpose_4x3() {
        let matrix = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0;
            10.0, 11.0, 12.0
        ];
        let transposed = matrix.transpose();

        let expected = matrix![
            1.0, 4.0, 7.0, 10.0;
            2.0, 5.0, 8.0, 11.0;
            3.0, 6.0, 9.0, 12.0
        ];
        assert_eq!(transposed, expected);
    }

    #[test]
    fn test_map_add_one() {
        let matrix = matrix![
            1., 2.;
            3., 4.
        ];
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);

        let transformed = matrix.map(|x| x + 1.0);

        let expected = matrix![
            2., 3.;
            4., 5.
        ];

        assert_eq!(transformed, expected);
    }

    #[test]
    fn test_map_square() {
        let matrix = matrix![1., 2.; 3., 4.];

        let transformed = matrix.map(|x| x * x);

        let expected = matrix![1., 4.; 9., 16.];

        assert_eq!(transformed, expected);
    }
}
