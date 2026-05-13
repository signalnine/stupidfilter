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
    /// Class returned when the decision function is positive (sum > rho).
    /// Read from the libsvm 'label' line; libsvm orders labels by their
    /// first appearance in the training file, so this is not always 1.
    pub label_pos: f64,
    /// Class returned when the decision function is non-positive.
    pub label_neg: f64,
    pub support_vectors: Vec<SupportVector>,
    pub scale_factors: Vec<f64>,
    pub num_features: usize,
}

struct ModelHeader {
    gamma: f64,
    rho: f64,
    label_pos: f64,
    label_neg: f64,
    support_vectors: Vec<SupportVector>,
}

impl SvmModel {
    /// Load model from base filename (loads .mod and .sf files)
    pub fn load(base_path: &str) -> Result<Self, String> {
        let model_path = format!("{}.mod", base_path);
        let scale_path = format!("{}.sf", base_path);

        let header = Self::load_model(&model_path)?;
        let scale_factors = Self::load_scale_factors(&scale_path)?;

        Ok(SvmModel {
            gamma: header.gamma,
            rho: header.rho,
            label_pos: header.label_pos,
            label_neg: header.label_neg,
            support_vectors: header.support_vectors,
            num_features: scale_factors.len(),
            scale_factors,
        })
    }

    /// Load the libsvm model file
    fn load_model(path: &str) -> Result<ModelHeader, String> {
        let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
        let reader = BufReader::new(file);
        let lines = reader.lines();

        let mut gamma: Option<f64> = None;
        let mut rho: Option<f64> = None;
        let mut labels: Option<(f64, f64)> = None;
        let mut kernel_type: Option<String> = None;
        let mut svm_type: Option<String> = None;
        let mut nr_class: Option<u32> = None;
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
            } else if let Some(rest) = line.strip_prefix("gamma ") {
                gamma = Some(rest.trim().parse().map_err(|_| "Invalid gamma value")?);
            } else if let Some(rest) = line.strip_prefix("rho ") {
                rho = Some(rest.trim().parse().map_err(|_| "Invalid rho value")?);
            } else if let Some(rest) = line.strip_prefix("label ") {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if parts.len() < 2 {
                    return Err(format!(
                        "Invalid 'label' line (need 2 values for binary classifier): {}",
                        line
                    ));
                }
                let l0: f64 = parts[0].parse().map_err(|_| "Invalid label[0] value")?;
                let l1: f64 = parts[1].parse().map_err(|_| "Invalid label[1] value")?;
                labels = Some((l0, l1));
            } else if let Some(rest) = line.strip_prefix("kernel_type ") {
                kernel_type = Some(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("svm_type ") {
                svm_type = Some(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("nr_class ") {
                nr_class = Some(
                    rest.trim()
                        .parse()
                        .map_err(|_| "Invalid nr_class value")?,
                );
            } else if line == "SV" {
                in_sv_section = true;
            }
            // Ignore other header lines (total_sv, nr_sv).
        }

        // This port only supports binary C-SVC with the RBF kernel. Anything
        // else would silently apply rbf_kernel() to incompatible parameters
        // and return garbage. Reject up front with a clear message.
        if let Some(ref t) = svm_type {
            if t != "c_svc" {
                return Err(format!(
                    "unsupported svm_type '{}': only 'c_svc' is supported",
                    t
                ));
            }
        }
        if let Some(ref k) = kernel_type {
            if k != "rbf" {
                return Err(format!(
                    "unsupported kernel_type '{}': only 'rbf' is supported",
                    k
                ));
            }
        }
        if let Some(n) = nr_class {
            if n != 2 {
                return Err(format!(
                    "unsupported nr_class {}: only binary classifiers (nr_class 2) are supported",
                    n
                ));
            }
        }

        let rho = rho.ok_or("model file missing required 'rho' line")?;
        let gamma = gamma.ok_or("model file missing required 'gamma' line for RBF kernel")?;
        // libsvm always writes a 'label' line for classification models, but
        // fall back to the historical (1, 0) ordering if absent so trivially
        // hand-constructed test models still work.
        let (label_pos, label_neg) = labels.unwrap_or((1.0, 0.0));

        Ok(ModelHeader {
            gamma,
            rho,
            label_pos,
            label_neg,
            support_vectors,
        })
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
        for _ in 0..count {
            let line = lines
                .next()
                .ok_or_else(|| format!(
                    "Truncated scale factors file {}: expected {} factors, got {}",
                    path,
                    count,
                    factors.len()
                ))?
                .map_err(|e| format!("Read error: {}", e))?;
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

        // Match libsvm's svm_predict: when the decision function is positive
        // return label[0], otherwise label[1]. For the bundled c_rbf model
        // these are 1 and 0, but a retrain can produce 'label 0 1' and the
        // meaning of the output flips with it.
        if sum > self.rho {
            self.label_pos
        } else {
            self.label_neg
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::extract_features;
    use std::io::Write;

    fn bundled_model() -> SvmModel {
        // Unit tests run with cwd = crate root (rust/), same as parity.rs.
        SvmModel::load("../data/c_rbf").expect("load bundled c_rbf model")
    }

    fn empty_model() -> SvmModel {
        SvmModel {
            gamma: 0.5,
            rho: 0.0,
            label_pos: 1.0,
            label_neg: 0.0,
            support_vectors: Vec::new(),
            scale_factors: Vec::new(),
            num_features: 0,
        }
    }

    #[test]
    fn parse_sv_line_basic() {
        let line = "1 1:0.739726 2:0.054795 3:0.027397";
        let sv = SvmModel::parse_sv_line(line).unwrap();
        assert_eq!(sv.alpha_y, 1.0);
        assert_eq!(sv.features.indices, vec![1, 2, 3]);
        assert!((sv.features.values[0] - 0.739726).abs() < 1e-6);
    }

    #[test]
    fn predict_hello_world_matches_cpp_reference() {
        // Reference C++ classification for "Hello world\n" is 1.0 (not stupid).
        let model = bundled_model();
        let features = extract_features("Hello world\n");
        assert_eq!(model.predict(&features), 1.0);
    }

    #[test]
    fn predict_l33t_matches_cpp_reference() {
        // Reference C++ classification for the canonical stupid l33t example
        // is 0.0. classify_test.sh asserts the same end-to-end.
        let model = bundled_model();
        let features = extract_features("OMG UR SO DUMB 4 REAL\n");
        assert_eq!(model.predict(&features), 0.0);
    }

    #[test]
    fn load_model_rejects_missing_files() {
        let err = SvmModel::load("/nonexistent/path/that/does/not/exist")
            .expect_err("load should fail when files are absent");
        assert!(
            err.to_lowercase().contains("cannot open"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn load_model_rejects_non_rbf_kernel() {
        let dir = tempdir("non_rbf_kernel");
        let mod_text =
            std::fs::read_to_string("../data/c_rbf.mod").expect("read bundled model");
        let altered = mod_text.replace("kernel_type rbf", "kernel_type linear");
        std::fs::write(dir.join("m.mod"), altered).unwrap();
        std::fs::copy("../data/c_rbf.sf", dir.join("m.sf")).unwrap();

        let err = SvmModel::load(dir.join("m").to_str().unwrap())
            .expect_err("non-RBF kernel must be rejected");
        assert!(
            err.contains("kernel_type"),
            "expected kernel_type error, got: {}",
            err
        );
    }

    #[test]
    fn load_model_rejects_non_csvc_svm_type() {
        let dir = tempdir("non_csvc");
        let mod_text =
            std::fs::read_to_string("../data/c_rbf.mod").expect("read bundled model");
        let altered = mod_text.replace("svm_type c_svc", "svm_type nu_svc");
        std::fs::write(dir.join("m.mod"), altered).unwrap();
        std::fs::copy("../data/c_rbf.sf", dir.join("m.sf")).unwrap();

        let err = SvmModel::load(dir.join("m").to_str().unwrap())
            .expect_err("non c_svc svm_type must be rejected");
        assert!(
            err.contains("svm_type"),
            "expected svm_type error, got: {}",
            err
        );
    }

    #[test]
    fn load_model_rejects_multiclass() {
        let dir = tempdir("multiclass");
        let mod_text =
            std::fs::read_to_string("../data/c_rbf.mod").expect("read bundled model");
        let altered = mod_text.replace("nr_class 2", "nr_class 3");
        std::fs::write(dir.join("m.mod"), altered).unwrap();
        std::fs::copy("../data/c_rbf.sf", dir.join("m.sf")).unwrap();

        let err = SvmModel::load(dir.join("m").to_str().unwrap())
            .expect_err("multiclass model must be rejected");
        assert!(
            err.contains("nr_class"),
            "expected nr_class error, got: {}",
            err
        );
    }

    #[test]
    fn load_model_rejects_truncated_scale_file() {
        let dir = tempdir("truncated_sf");
        std::fs::copy("../data/c_rbf.mod", dir.join("m.mod")).unwrap();
        // Claim 8 scale factors but only provide 3.
        let mut sf = std::fs::File::create(dir.join("m.sf")).unwrap();
        writeln!(sf, "8\n1.0\n2.0\n3.0").unwrap();

        let err = SvmModel::load(dir.join("m").to_str().unwrap())
            .expect_err("truncated scale file must be rejected");
        assert!(
            err.to_lowercase().contains("truncated") || err.contains("scale"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn rbf_kernel_matches_hand_computed_value() {
        // K(x, y) = exp(-gamma * ||x - y||^2). With gamma = 0.5, x = [1.0, 2.0],
        // sv values at the matching libsvm 1-based indices = [0.5, 1.5]:
        //   diff_sq = (1.0 - 0.5)^2 + (2.0 - 1.5)^2 = 0.25 + 0.25 = 0.5
        //   K = exp(-0.5 * 0.5) = exp(-0.25)
        let mut model = empty_model();
        model.gamma = 0.5;
        let sv = SparseVector {
            indices: vec![1, 2],
            values: vec![0.5, 1.5],
        };
        let k = model.rbf_kernel(&[1.0, 2.0], &sv);
        let expected = (-0.25f64).exp();
        assert!((k - expected).abs() < 1e-12, "got {}, want {}", k, expected);
    }

    #[test]
    fn rbf_kernel_handles_missing_sparse_indices() {
        // Missing indices in the sparse SV are treated as 0.0. With gamma=1,
        // x = [3.0, 0.0, 4.0] and sv with only index 2 set to 5.0:
        //   diff_sq = 9 + 25 + 16 = 50; K = exp(-50)
        let mut model = empty_model();
        model.gamma = 1.0;
        let sv = SparseVector {
            indices: vec![2],
            values: vec![5.0],
        };
        let k = model.rbf_kernel(&[3.0, 0.0, 4.0], &sv);
        let expected = (-50.0f64).exp();
        assert!((k - expected).abs() < 1e-300, "got {}, want {}", k, expected);
    }

    #[test]
    fn scale_features_passes_through_when_scale_factors_short() {
        // scale_features doubles each feature scaled by the corresponding
        // factor; trailing features are returned unscaled.
        let mut model = empty_model();
        model.scale_factors = vec![2.0, 3.0, 4.0];
        let scaled = model.scale_features(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(scaled, vec![2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn scale_features_ignores_extra_scale_factors() {
        // More scale factors than features just leaves the tail unused.
        let mut model = empty_model();
        model.scale_factors = vec![2.0; 12];
        let scaled = model.scale_features(&[1.0; 8]);
        assert_eq!(scaled, vec![2.0; 8]);
    }

    fn tempdir(tag: &str) -> std::path::PathBuf {
        // CARGO_TARGET_TMPDIR is only set for integration tests, so unit tests
        // fall back to the system temp dir. Use the process id plus a tag so
        // parallel test runs don't collide.
        let dir = std::env::temp_dir()
            .join(format!("stupidfilter-svm-test-{}", std::process::id()))
            .join(tag);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
