//! ML-flavoured analytics.
//!
//! The problem statement — "one mansion can make the whole street look
//! expensive" — is a textbook illustration of why the mean is sensitive to
//! outliers. This module turns that observation into an algorithm: a z-score
//! outlier detector that flags the "mansion" skewing the mean.
//!
//! `ndarray` carries the data (the same container used by Rust ML libraries),
//! `rayon` computes the variance reduction in parallel, and the detector is
//! behind a trait so different models (IQR, Isolation Forest, ...) can be
//! plugged in without changing callers.

use crate::models::HousePrices;
use crate::stats::MeanCalculator;
use rayon::prelude::*;

/// Flags outlier prices.
pub trait OutlierDetector {
    /// Returns the indices of prices whose score crosses `threshold`.
    fn detect(&self, prices: &HousePrices, threshold: f64) -> Vec<usize>;
}

/// Z-score outlier detector.
///
/// A price is an outlier when `|x - mean| / std_dev >= threshold`.
/// Uses the population standard deviation, computed in parallel.
pub struct ZScoreOutlierDetector<'a> {
    mean_calc: &'a dyn MeanCalculator,
}

impl<'a> ZScoreOutlierDetector<'a> {
    /// Dependency Inversion (SOLID-D): depend on the `MeanCalculator` trait,
    /// not a concrete engine, so any implementation can supply the mean.
    pub fn new(mean_calc: &'a dyn MeanCalculator) -> Self {
        Self { mean_calc }
    }
}

impl<'a> OutlierDetector for ZScoreOutlierDetector<'a> {
    fn detect(&self, prices: &HousePrices, threshold: f64) -> Vec<usize> {
        let data = prices.data();
        if data.len() < 2 {
            return Vec::new();
        }
        let mean = self.mean_calc.mean(prices);

        // Parallel sum of squared deviations.
        let sq_diffs: Vec<f64> =
            data.iter().copied().map(|x| (x - mean).powi(2)).collect();
        let variance: f64 = sq_diffs.par_iter().sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Vec::new();
        }

        data.iter()
            .copied()
            .enumerate()
            .filter(|(_, x)| ((x - mean).abs() / std_dev) >= threshold)
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::ParallelStatisticsEngine;
    use ndarray::Array1;

    #[test]
    fn detects_the_mansion() {
        let prices = HousePrices::new(Array1::from(vec![
            320.0, 300.0, 2500.0, 310.0, 335.0,
        ]));
        let engine = ParallelStatisticsEngine::new();
        let detector = ZScoreOutlierDetector::new(&engine);
        // Only the 2500 "mansion" clears a 1.5-sigma threshold.
        assert_eq!(detector.detect(&prices, 1.5), vec![2]);
    }

    #[test]
    fn no_outliers_on_uniform_street() {
        let prices = HousePrices::new(Array1::from(vec![300.0, 310.0, 305.0]));
        let engine = ParallelStatisticsEngine::new();
        let detector = ZScoreOutlierDetector::new(&engine);
        assert!(detector.detect(&prices, 1.5).is_empty());
    }
}
