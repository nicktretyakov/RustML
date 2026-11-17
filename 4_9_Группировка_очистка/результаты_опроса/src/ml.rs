//! Auxiliary machine-learning step.
//!
//! A linear regression (fit with the `smartcore` crate) models the rating as
//! a function of device (one-hot encoded) and age group (ordinal encoded).
//! Its diagnostics are emitted on stderr so the main report on stdout stays
//! byte-identical to the layout the team expects.

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::{
    LinearRegression, LinearRegressionParameters, LinearRegressionSolverName,
};

use crate::models::SurveyAnswer;
use crate::traits::{FitSummary, SatisfactionModel};

/// One-hot device features plus an ordinal age feature.
pub struct LinearScoreRegressor;

impl LinearScoreRegressor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LinearScoreRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode "18-29" -> 0.0, "30-44" -> 1.0, "45+" -> 2.0 so the regressor sees
/// the natural ordering of the age buckets. Unknown labels get the middle
/// value so they never skew the slope toward an extreme.
fn age_ordinal(group: &str) -> f64 {
    match group {
        "18-29" => 0.0,
        "30-44" => 1.0,
        "45+" => 2.0,
        _ => 1.0,
    }
}

impl SatisfactionModel for LinearScoreRegressor {
    fn fit(&self, answers: &[SurveyAnswer]) -> Option<FitSummary> {
        // A regression needs noticeably more samples than features to be
        // meaningful; bail out quietly otherwise (the report still prints).
        if answers.len() < 4 {
            return None;
        }

        // Sorted, de-duplicated device names => one column per device.
        let mut devices: Vec<String> = answers.iter().map(|a| a.device.clone()).collect();
        devices.sort();
        devices.dedup();

        let n_features = devices.len() + 1; // +1 for the age ordinal
        let n_samples = answers.len();

        // Build the design matrix row by row: one-hot device dummies + age.
        let rows: Vec<Vec<f64>> = answers
            .iter()
            .map(|a| {
                let mut row: Vec<f64> = devices
                    .iter()
                    .map(|d| if a.device == *d { 1.0 } else { 0.0 })
                    .collect();
                row.push(age_ordinal(&a.age_group));
                row
            })
            .collect();
        let row_refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        let x = DenseMatrix::from_2d_array(&row_refs).ok()?;
        let y: Vec<f64> = answers.iter().map(|a| a.score).collect();

        // SVD is the more stable solver; it tolerates the mild collinearity
        // that one-hot dummies introduce against the intercept.
        let params =
            LinearRegressionParameters::default().with_solver(LinearRegressionSolverName::SVD);
        let model = LinearRegression::fit(&x, &y, params).ok()?;
        let y_hat: Vec<f64> = model.predict(&x).ok()?;

        // Evaluate the fit: mean squared error and R^2.
        let mean_y = y.iter().sum::<f64>() / n_samples as f64;
        let ss_res: f64 = y
            .iter()
            .zip(y_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let ss_tot: f64 = y.iter().map(|a| (a - mean_y).powi(2)).sum();
        let mse = ss_res / n_samples as f64;
        let r2 = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        let mut feature_names = devices.clone();
        feature_names.push("age_ordinal".to_string());

        Some(FitSummary {
            n_samples,
            n_features,
            feature_names,
            mse,
            r2,
        })
    }
}
