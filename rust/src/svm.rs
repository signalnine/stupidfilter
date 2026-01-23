// SVM model module
// Loads libsvm format models and performs RBF kernel prediction

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::features::Features;

/// A sparse feature vector (index, value pairs)
#[derive(Debug, Clone)]
pub struct SparseVector {
    pub indices: Vec<usize>,
    pub values: Vec<f64>,
}

/// A support vector with its coefficient
#[derive(Debug, Clone)]
pub struct SupportVector {
    pub alpha_y: f64, // alpha * y (coefficient from SV line)
    pub features: SparseVector,
}

/// SVM Model for binary classification with RBF kernel
#[derive(Debug)]
pub struct SvmModel {
    pub gamma: f64,
    pub rho: f64,
    pub support_vectors: Vec<SupportVector>,
    pub scale_factors: Vec<f64>,
    pub num_features: usize,
}

impl SvmModel {
    /// Load model from base filename (loads .mod and .sf files)
    pub fn load(base_path: &str) -> Result<Self, String> {
        let model_path = format!("{}.mod", base_path);
        let scale_path = format!("{}.sf", base_path);

        let (gamma, rho, support_vectors) = Self::load_model(&model_path)?;
        let scale_factors = Self::load_scale_factors(&scale_path)?;

        Ok(SvmModel {
            gamma,
            rho,
            support_vectors,
            num_features: scale_factors.len(),
            scale_factors,
        })
    }

    /// Load the libsvm model file
    fn load_model(path: &str) -> Result<(f64, f64, Vec<SupportVector>), String> {
        let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let mut gamma = 0.0;
        let mut rho = 0.0;
        let mut support_vectors = Vec::new();
        let mut in_sv_section = false;

        for line in lines {
            let line = line.map_err(|e| format!("Read error: {}", e))?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            if in_sv_section {
                // Parse support vector line: "alpha_y index:value index:value ..."
                let sv = Self::parse_sv_line(line)?;
                support_vectors.push(sv);
            } else if line.starts_with("gamma ") {
                gamma = line[6..]
                    .trim()
                    .parse()
                    .map_err(|_| "Invalid gamma value")?;
            } else if line.starts_with("rho ") {
                rho = line[4..]
                    .trim()
                    .parse()
                    .map_err(|_| "Invalid rho value")?;
            } else if line == "SV" {
                in_sv_section = true;
            }
            // Ignore other header lines (svm_type, kernel_type, nr_class, etc.)
        }

        Ok((gamma, rho, support_vectors))
    }

    /// Parse a support vector line
    fn parse_sv_line(line: &str) -> Result<SupportVector, String> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Err("Empty SV line".to_string());
        }

        let alpha_y: f64 = parts[0].parse().map_err(|_| "Invalid alpha value")?;

        let mut indices = Vec::new();
        let mut values = Vec::new();

        for part in &parts[1..] {
            if let Some((idx_str, val_str)) = part.split_once(':') {
                let idx: usize = idx_str.parse().map_err(|_| "Invalid index")?;
                let val: f64 = val_str.parse().map_err(|_| "Invalid value")?;
                indices.push(idx);
                values.push(val);
            }
        }

        Ok(SupportVector {
            alpha_y,
            features: SparseVector { indices, values },
        })
    }

    /// Load scale factors from .sf file
    fn load_scale_factors(path: &str) -> Result<Vec<f64>, String> {
        let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // First line is count
        let count_line = lines
            .next()
            .ok_or("Empty scale file")?
            .map_err(|e| format!("Read error: {}", e))?;
        let count: usize = count_line
            .trim()
            .parse()
            .map_err(|_| "Invalid scale factor count")?;

        let mut factors = Vec::with_capacity(count);
        for line in lines.take(count) {
            let line = line.map_err(|e| format!("Read error: {}", e))?;
            let factor: f64 = line.trim().parse().map_err(|_| "Invalid scale factor")?;
            factors.push(factor);
        }

        Ok(factors)
    }

    /// Scale features using the loaded scale factors
    fn scale_features(&self, features: &[f64; 8]) -> Vec<f64> {
        features
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                if i < self.scale_factors.len() {
                    val * self.scale_factors[i]
                } else {
                    val
                }
            })
            .collect()
    }

    /// Compute RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
    fn rbf_kernel(&self, x: &[f64], sv: &SparseVector) -> f64 {
        // Compute squared Euclidean distance
        // x is dense, sv is sparse
        let mut dist_sq = 0.0;

        // For each dimension in x
        for (i, &xi) in x.iter().enumerate() {
            let idx = i + 1; // libsvm indices are 1-based

            // Find corresponding value in sparse vector
            let sv_val = sv
                .indices
                .iter()
                .position(|&j| j == idx)
                .map(|pos| sv.values[pos])
                .unwrap_or(0.0);

            let diff = xi - sv_val;
            dist_sq += diff * diff;
        }

        (-self.gamma * dist_sq).exp()
    }

    /// Predict class for the given features
    pub fn predict(&self, features: &Features) -> f64 {
        let feature_array = features.to_array();
        let scaled = self.scale_features(&feature_array);

        // SVM decision function: sign(sum(alpha_i * y_i * K(x, x_i)) - rho)
        let mut sum = 0.0;
        for sv in &self.support_vectors {
            let kernel_val = self.rbf_kernel(&scaled, &sv.features);
            sum += sv.alpha_y * kernel_val;
        }

        // Decision: positive = class 1, negative = class 0
        // The original returns 1.0 for "non-stupid" and 0.0 for "stupid"
        if sum > self.rho {
            1.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sv_line() {
        let line = "1 1:0.739726 2:0.054795 3:0.027397";
        let sv = SvmModel::parse_sv_line(line).unwrap();
        assert_eq!(sv.alpha_y, 1.0);
        assert_eq!(sv.features.indices, vec![1, 2, 3]);
        assert!((sv.features.values[0] - 0.739726).abs() < 1e-6);
    }
}
