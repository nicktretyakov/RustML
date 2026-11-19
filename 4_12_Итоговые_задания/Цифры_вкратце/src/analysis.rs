//! Analysis layer (Single Responsibility: only analyses data).
//!
//! The mean image of a digit is computed by reducing, for every pixel
//! position, the values of all samples whose label equals `d`. The 64 pixel
//! reductions are independent, so they are run in parallel across threads
//! with the `rayon` data-parallelism crate.
use crate::dataset::DigitDataset;
use rayon::prelude::*;

/// Interface (Interface Segregation / Open-Closed) for any image analysis.
pub trait ImageAnalyzer {
    /// Mean image of `digit`: one floating-point value per pixel (length 64).
    fn average_image(&self, digit: i64) -> Vec<f64>;
    /// (row, column) of the brightest pixel, 0-indexed, first max wins.
    fn brightest_pixel(&self, image: &[f64]) -> (usize, usize);
    /// Mean of the 64 pixel means ("average ink").
    fn average_ink(&self, image: &[f64]) -> f64;
}

/// Mean-image analyzer backed by any `DigitDataset` (Dependency Inversion:
/// it depends on the trait, not on a concrete loader).
pub struct MeanImageAnalyzer<'d, D: DigitDataset> {
    data: &'d D,
}

impl<'d, D: DigitDataset> MeanImageAnalyzer<'d, D> {
    pub fn new(data: &'d D) -> Self {
        Self { data }
    }
}

impl<'d, D: DigitDataset> ImageAnalyzer for MeanImageAnalyzer<'d, D> {
    fn average_image(&self, digit: i64) -> Vec<f64> {
        let nf = self.data.n_features();
        let data = self.data.data();

        // Indices of the samples whose label equals `digit`.
        let selected: Vec<usize> = self
            .data
            .target()
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == digit)
            .map(|(i, _)| i)
            .collect();
        let count = selected.len().max(1) as f64;

        // Parallel reduction: each pixel position is an independent thread of work.
        (0..nf)
            .into_par_iter()
            .map(|p| {
                let mut sum = 0.0_f64;
                for &s in &selected {
                    sum += data[s * nf + p];
                }
                sum / count
            })
            .collect()
    }

    fn brightest_pixel(&self, image: &[f64]) -> (usize, usize) {
        // First occurrence of the maximum (matches `numpy.argmax`).
        let mut best_i = 0usize;
        let mut best_v = f64::MIN;
        for (i, &v) in image.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        (best_i / 8, best_i % 8)
    }

    fn average_ink(&self, image: &[f64]) -> f64 {
        let n = image.len().max(1) as f64;
        image.iter().sum::<f64>() / n
    }
}
