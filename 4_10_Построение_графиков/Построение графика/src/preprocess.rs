//! Parallel preprocessing.
//!
//! This is where **multithreading** enters the pipeline. The [`Preprocessor`]
//! trait keeps the contract small (interface segregation); [`RayonPreprocessor`]
//! turns raw records into everything downstream stages need:
//!
//! * `points`  — `(hour, level_cm)` pairs for the plot,
//! * `features` — cumulative rainfall (running total of `rain_mm`), the
//!   engineered predictor for the regression,
//! * `targets`  — `level_cm`, the regression response,
//! * summary statistics computed with data-parallel reductions.
//!
//! `rayon`'s `par_iter` distributes the per-record work and the min/max/mean
//! reductions across the available CPU cores.

use rayon::prelude::*;

use crate::model::RiverRecord;

/// Output of the preprocessing stage — a plain data object shared with the
/// ML and plotting stages. Kept free of behaviour on purpose (SRP).
#[derive(Debug, Clone)]
pub struct ProcessedData {
    /// `(hour, level_cm)` — exactly what `river.plot(x="hour", y="level_cm")`
    /// visualises in the reference Python snippet.
    pub points: Vec<(f64, f64)>,
    /// Engineered explanatory variable: cumulative `rain_mm` (running total).
    /// The river level integrates rainfall over time — runoff lags the rain by
    /// several hours — so the contemporaneous `rain_mm` barely predicts the
    /// level while the running total tracks it closely. This is standard ML
    /// feature engineering, and it keeps the univariate regression meaningful.
    pub features: Vec<f64>,
    /// Response variable for the regression: `level_cm`.
    pub targets: Vec<f64>,
    pub hour_min: f64,
    pub hour_max: f64,
    pub level_min: f64,
    pub level_max: f64,
    pub mean_level: f64,
}

/// Contract for turning raw records into [`ProcessedData`].
pub trait Preprocessor: Send {
    fn process(&self, records: &[RiverRecord]) -> ProcessedData;
}

/// `rayon`-backed implementation.
#[derive(Debug, Default, Clone, Copy)]
pub struct RayonPreprocessor;

impl RayonPreprocessor {
    pub fn new() -> Self {
        Self
    }
}

impl Preprocessor for RayonPreprocessor {
    fn process(&self, records: &[RiverRecord]) -> ProcessedData {
        assert!(!records.is_empty(), "Preprocessor received no records");

        // --- data-parallel map: build the (hour, level) plot points and collect
        // the raw hourly rainfall in one pass. Each iteration is independent, so
        // rayon can split the slice freely. ---
        let mapped: Vec<((f64, f64), f64)> = records
            .par_iter()
            .map(|r| ((r.hour, r.level_cm), r.feature_rain()))
            .collect();
        let points: Vec<(f64, f64)> = mapped.iter().map(|(p, _)| *p).collect();
        let rain: Vec<f64> = mapped.iter().map(|(_, r)| *r).collect();
        let targets: Vec<f64> = points.iter().map(|(_, l)| *l).collect();

        // --- feature engineering: cumulative rainfall (sequential prefix sum;
        // for n=24 a scan is cheaper than a parallel prefix network). ---
        let mut features = Vec::with_capacity(rain.len());
        let mut acc = 0.0;
        for &r in &rain {
            acc += r;
            features.push(acc);
        }

        // --- data-parallel reductions for axis ranges and summary stats.
        // `fold` builds a (min, max) accumulator per partition, then `reduce`
        // merges the partitions — a single pass over the data, in parallel. ---
        let (hour_min, hour_max) = points
            .par_iter()
            .map(|(h, _)| *h)
            .fold(
                || (f64::INFINITY, f64::NEG_INFINITY),
                |(lo, hi), h| (lo.min(h), hi.max(h)),
            )
            .reduce(
                || (f64::INFINITY, f64::NEG_INFINITY),
                |(a_lo, a_hi), (b_lo, b_hi)| (a_lo.min(b_lo), a_hi.max(b_hi)),
            );
        let (level_min, level_max) = points
            .par_iter()
            .map(|(_, l)| *l)
            .fold(
                || (f64::INFINITY, f64::NEG_INFINITY),
                |(lo, hi), l| (lo.min(l), hi.max(l)),
            )
            .reduce(
                || (f64::INFINITY, f64::NEG_INFINITY),
                |(a_lo, a_hi), (b_lo, b_hi)| (a_lo.min(b_lo), a_hi.max(b_hi)),
            );
        let level_sum: f64 = targets.par_iter().sum();
        let mean_level = level_sum / targets.len() as f64;

        ProcessedData {
            points,
            features,
            targets,
            hour_min,
            hour_max,
            level_min,
            level_max,
            mean_level,
        }
    }
}
