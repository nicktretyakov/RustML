//! Statistics engine.
//!
//! Interface Segregation (SOLID-I): instead of one fat "do everything" trait,
//! each statistical capability lives in its own trait so a consumer only has
//! to depend on what it actually uses.
//!
//! Open/Closed (SOLID-O): a new algorithm (streaming mean, GPU median, ...)
//! is added by writing another impl, never by editing existing code.
//!
//! The concrete `ParallelStatisticsEngine` computes everything with `rayon`
//! data-parallel iteration over the `ndarray` price vector.

use crate::models::HousePrices;
use rayon::prelude::*;

/// Computes the arithmetic mean.
pub trait MeanCalculator {
    fn mean(&self, prices: &HousePrices) -> f64;
}

/// Computes the median (middle value for odd n, average of the two middle
/// values for even n — matches `numpy.median`).
pub trait MedianCalculator {
    fn median(&self, prices: &HousePrices) -> f64;
}

/// Counts how many prices are strictly below the mean.
pub trait BelowMeanCounter {
    fn count_below_mean(&self, prices: &HousePrices) -> usize;
}

/// Convenience aggregate: anything that can do all three is a full engine.
pub trait StatisticsEngine: MeanCalculator + MedianCalculator + BelowMeanCounter {}

// Blanket impl — any type satisfying the three sub-traits is automatically an
// engine. Liskov (SOLID-L): every engine is substitutable for the trait
// object `&dyn StatisticsEngine`.
impl<T> StatisticsEngine for T where T: MeanCalculator + MedianCalculator + BelowMeanCounter {}

/// Multithreaded statistics engine.
///
/// Sums and filtering run across all CPU cores via `rayon`; the data lives in
/// an `ndarray` tensor so it can be reused by downstream ML code without
/// copying.
#[derive(Default)]
pub struct ParallelStatisticsEngine;

impl ParallelStatisticsEngine {
    pub fn new() -> Self {
        Self
    }
}

impl MeanCalculator for ParallelStatisticsEngine {
    fn mean(&self, prices: &HousePrices) -> f64 {
        let data = prices.data();
        if data.is_empty() {
            return 0.0;
        }
        // Parallel reduction across the price vector.
        let sum: f64 = data.iter().copied().collect::<Vec<_>>().par_iter().sum();
        sum / data.len() as f64
    }
}

impl MedianCalculator for ParallelStatisticsEngine {
    fn median(&self, prices: &HousePrices) -> f64 {
        let data = prices.data();
        if data.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = data.iter().copied().collect();
        sorted.sort_by(|a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let n = sorted.len();
        if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }
    }
}

impl BelowMeanCounter for ParallelStatisticsEngine {
    fn count_below_mean(&self, prices: &HousePrices) -> usize {
        let mean = self.mean(prices);
        let data = prices.data();
        // Parallel filter + count.
        data.iter()
            .copied()
            .collect::<Vec<_>>()
            .par_iter()
            .filter(|&&p| p < mean)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn fixture() -> HousePrices {
        // The exact example from the problem statement.
        HousePrices::new(Array1::from(vec![320.0, 300.0, 2500.0, 310.0, 335.0]))
    }

    #[test]
    fn mean_matches_example() {
        let engine = ParallelStatisticsEngine::new();
        assert!((engine.mean(&fixture()) - 753.0).abs() < 1e-9);
    }

    #[test]
    fn median_matches_example() {
        let engine = ParallelStatisticsEngine::new();
        assert!((engine.median(&fixture()) - 320.0).abs() < 1e-9);
    }

    #[test]
    fn count_below_mean_matches_example() {
        let engine = ParallelStatisticsEngine::new();
        assert_eq!(engine.count_below_mean(&fixture()), 4);
    }

    #[test]
    fn even_count_median_is_average_of_middles() {
        let engine = ParallelStatisticsEngine::new();
        let prices = HousePrices::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]));
        assert!((engine.median(&prices) - 2.5).abs() < 1e-9);
    }

    #[test]
    fn empty_prices_are_safe() {
        let engine = ParallelStatisticsEngine::new();
        let prices = HousePrices::new(Array1::from(vec![]));
        assert_eq!(engine.mean(&prices), 0.0);
        assert_eq!(engine.median(&prices), 0.0);
        assert_eq!(engine.count_below_mean(&prices), 0);
    }
}
