//! Machine-learning layer.
//!
//! Uses the `ndarray` crate — the foundational Rust tensor/ML library that
//! `linfa` and `smartcore` are built on — to hold the feature matrix, plus a
//! hand-written [`StandardScaler`] and [`LogisticClassifier`] that *quantify*
//! whether the two classes form separable clouds on the plotted axes.
//!
//! **SRP**: preprocessing and classification are separate types.
//! **DIP / LSP**: callers depend on [`DiagnosisClassifier`]; any classifier
//! can be substituted in.

use ndarray::{Array1, Array2};
use rayon::prelude::*;

/// Any binary diagnosis classifier (**Interface Segregation** + **Liskov**).
pub trait DiagnosisClassifier: Sync {
    fn predict(&self, features: &[f64]) -> usize;
    fn name(&self) -> &'static str;
}

/// Standardizes each column to zero mean / unit variance.
///
/// Column statistics are computed **in parallel** with `rayon` — one rayon
/// task per feature column.
pub struct StandardScaler {
    mean: Array1<f64>,
    std: Array1<f64>,
}

impl StandardScaler {
    pub fn fit(x: &Array2<f64>) -> Self {
        let nrows = x.nrows();
        let ncols = x.ncols();

        let stats: Vec<(f64, f64)> = (0..ncols)
            .into_par_iter()
            .map(|j| {
                let mut sum = 0.0;
                for i in 0..nrows {
                    sum += *x.get((i, j)).unwrap();
                }
                let mean = sum / nrows as f64;
                let mut var = 0.0;
                for i in 0..nrows {
                    let d = *x.get((i, j)).unwrap() - mean;
                    var += d * d;
                }
                let std = (var / nrows as f64).sqrt().max(1e-12);
                (mean, std)
            })
            .collect();

        let mean = Array1::from_vec(stats.iter().map(|(m, _)| *m).collect());
        let std = Array1::from_vec(stats.iter().map(|(_, s)| *s).collect());
        Self { mean, std }
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut out = x.to_owned();
        for j in 0..x.ncols() {
            let m = self.mean[j];
            let s = self.std[j];
            for i in 0..x.nrows() {
                out[(i, j)] = (*x.get((i, j)).unwrap() - m) / s;
            }
        }
        out
    }
}

/// Logistic regression trained by batch gradient descent.
///
/// The per-sample gradient contribution is computed **in parallel** with
/// `rayon` every epoch (one rayon task per training sample); the partial
/// gradients are then reduced sequentially.
pub struct LogisticClassifier {
    weights: Vec<f64>,
    bias: f64,
}

impl LogisticClassifier {
    pub fn train(x: &Array2<f64>, y: &Array1<usize>, epochs: usize, lr: f64) -> Self {
        let n = x.nrows();
        let d = x.ncols();
        let y: Vec<f64> = y.iter().map(|&v| v as f64).collect();

        let mut w = vec![0.0_f64; d];
        let mut b = 0.0_f64;

        for _ in 0..epochs {
            // Parallel: each sample produces (bias-grad, weight-grad).
            let grads: Vec<(f64, Vec<f64>)> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let row: Vec<f64> = (0..d).map(|j| *x.get((i, j)).unwrap()).collect();
                    let z = b + w.iter().zip(&row).map(|(wj, xj)| wj * xj).sum::<f64>();
                    let p = 1.0 / (1.0 + (-z).exp());
                    let err = p - y[i];
                    let g: Vec<f64> = row.iter().map(|xj| err * xj).collect();
                    (err, g)
                })
                .collect();

            // Sequential reduction.
            let mut gw = vec![0.0; d];
            let mut gb = 0.0;
            for (err, g) in &grads {
                gb += err;
                for j in 0..d {
                    gw[j] += g[j];
                }
            }
            let inv_n = 1.0 / n as f64;
            for j in 0..d {
                w[j] -= lr * gw[j] * inv_n;
            }
            b -= lr * gb * inv_n;
        }

        Self { weights: w, bias: b }
    }

    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        let z = self.bias
            + self
                .weights
                .iter()
                .zip(features)
                .map(|(wj, xj)| wj * xj)
                .sum::<f64>();
        1.0 / (1.0 + (-z).exp())
    }

    pub fn accuracy(&self, x: &Array2<f64>, y: &Array1<usize>) -> f64 {
        let n = x.nrows();
        let d = x.ncols();
        let yv: Vec<usize> = y.to_vec();
        let correct = (0..n)
            .into_par_iter()
            .filter(|&i| {
                let row: Vec<f64> = (0..d).map(|j| *x.get((i, j)).unwrap()).collect();
                (self.predict_proba(&row) >= 0.5) as usize == yv[i]
            })
            .count();
        correct as f64 / n as f64
    }
}

impl DiagnosisClassifier for LogisticClassifier {
    fn predict(&self, features: &[f64]) -> usize {
        (self.predict_proba(features) >= 0.5) as usize
    }
    fn name(&self) -> &'static str {
        "LogisticRegression"
    }
}
