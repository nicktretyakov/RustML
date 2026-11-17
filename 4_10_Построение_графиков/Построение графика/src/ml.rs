//! Machine-learning layer.
//!
//! Trains an ordinary-least-squares linear regression
//! `level_cm ~ cumulative_rain_mm` with the [`smartcore`](https://docs.rs/smartcore)
//! ML crate (pure Rust, QR solver). The [`RegressionModel`] trait is the
//! dependency-inversion seam: the orchestrator depends on the *abstraction*,
//! not on `smartcore` concretely, so a `RidgeRegression` or a hand-rolled OLS
//! type could be dropped in without changing any caller.

use std::fmt;

use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::{
    LinearRegression, LinearRegressionParameters, LinearRegressionSolverName,
};

/// Result of fitting a regression model.
#[derive(Debug, Clone)]
pub struct FitReport {
    pub intercept: f64,
    pub slope: f64,
    pub predictions: Vec<f64>,
    pub r_squared: f64,
}

/// Abstraction every regressor must satisfy.
pub trait RegressionModel: Send {
    /// `features` is row-major with one row per sample; `targets` parallels it.
    fn fit(&self, features: &[f64], targets: &[f64]) -> Result<FitReport, MlError>;
}

#[derive(Debug)]
pub enum MlError {
    EmptyInput,
    MatrixBuild(String),
    FitFailed(String),
    PredictFailed(String),
}

impl fmt::Display for MlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlError::EmptyInput => write!(f, "ML error: empty input"),
            MlError::MatrixBuild(m) => write!(f, "ML error: cannot build matrix: {m}"),
            MlError::FitFailed(m) => write!(f, "ML error: fit failed: {m}"),
            MlError::PredictFailed(m) => write!(f, "ML error: predict failed: {m}"),
        }
    }
}

impl std::error::Error for MlError {}

/// OLS linear regression backed by `smartcore`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SmartcoreLinearRegression;

impl SmartcoreLinearRegression {
    pub fn new() -> Self {
        Self
    }
}

impl RegressionModel for SmartcoreLinearRegression {
    fn fit(&self, features: &[f64], targets: &[f64]) -> Result<FitReport, MlError> {
        let n = features.len();
        if n == 0 || targets.len() != n {
            return Err(MlError::EmptyInput);
        }

        // Build the (n x 1) feature matrix expected by smartcore.
        let rows: Vec<[f64; 1]> = features.iter().map(|&v| [v]).collect();
        let row_refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        let x = DenseMatrix::from_2d_array(&row_refs)
            .map_err(|e| MlError::MatrixBuild(format!("{e:?}")))?;
        let y: Vec<f64> = targets.to_vec();

        // Fit with the numerically stable QR solver (smartcore's recommended
        // default when the design matrix is full rank).
        let model = LinearRegression::fit(
            &x,
            &y,
            LinearRegressionParameters::default()
                .with_solver(LinearRegressionSolverName::QR),
        )
        .map_err(|e| MlError::FitFailed(format!("{e:?}")))?;

        let predictions = model
            .predict(&x)
            .map_err(|e| MlError::PredictFailed(format!("{e:?}")))?;

        // `coefficients()` is a DenseMatrix with shape (n_features, n_targets);
        // for the single-feature, single-target case element (0,0) is the slope.
        let slope = *model.coefficients().get((0, 0));
        let intercept = *model.intercept();
        let r_squared = coefficient_of_determination(&y, &predictions);

        Ok(FitReport {
            intercept,
            slope,
            predictions,
            r_squared,
        })
    }
}

/// R² = 1 - SS_res / SS_tot. Computed from the predictions the model produced
/// rather than read back from `smartcore`, so the metric stays crate-agnostic.
pub fn coefficient_of_determination(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.is_empty() {
        return 0.0;
    }
    let mean = actual.iter().sum::<f64>() / actual.len() as f64;
    let ss_tot: f64 = actual.iter().map(|a| (a - mean).powi(2)).sum();
    let ss_res: f64 = actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).powi(2))
        .sum();
    if ss_tot == 0.0 {
        0.0
    } else {
        1.0 - ss_res / ss_tot
    }
}
