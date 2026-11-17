//! Parallel aggregation of clean answers, powered by the `rayon` crate.
//!
//! Per-age-group statistics are computed concurrently on the rayon thread
//! pool (one group per task), and the device tally is built with a parallel
//! fold/reduce so the work scales with the number of cores.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::models::{AgeGroupStats, DeviceCount, SurveyAnswer, SurveyReport};
use crate::traits::SurveyAnalyzer;

/// Computes the report using rayon.
#[derive(Default)]
pub struct ParallelSurveyAnalyzer;

impl ParallelSurveyAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

/// Stable, human-friendly ordering of age buckets. Unknown labels fall back
/// to a large key so they sort after the known ones and stay deterministic
/// via a secondary lexicographic comparison at the call site.
fn age_rank(group: &str) -> u8 {
    match group {
        "18-29" => 0,
        "30-44" => 1,
        "45+" => 2,
        _ => 3,
    }
}

impl SurveyAnalyzer for ParallelSurveyAnalyzer {
    fn analyze(&self, answers: &[SurveyAnswer], missing: usize) -> SurveyReport {
        let kept = answers.len();

        // ---- device counts ------------------------------------------------
        // Tally in parallel: each rayon thread builds a partial map for its
        // chunk, then `reduce` merges the shards into one (cheap, small maps).
        let counts: HashMap<String, usize> = answers
            .par_iter()
            .fold(HashMap::new, |mut acc, a| {
                *acc.entry(a.device.clone()).or_insert(0) += 1;
                acc
            })
            .reduce(HashMap::new, |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_insert(0) += v;
                }
                a
            });

        let mut device_counts: Vec<DeviceCount> = counts
            .into_iter()
            .map(|(device, count)| DeviceCount { device, count })
            .collect();
        // Most common first; alphabetical tiebreak for determinism.
        device_counts.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.device.cmp(&b.device)));

        // ---- per-age-group stats, one group per rayon task ----------------
        let mut age_keys: Vec<String> = answers.iter().map(|a| a.age_group.clone()).collect();
        age_keys.sort();
        age_keys.dedup();

        let mut age_stats: Vec<AgeGroupStats> = age_keys
            .par_iter()
            .map(|group| {
                let scores: Vec<f64> = answers
                    .iter()
                    .filter(|a| &a.age_group == group)
                    .map(|a| a.score)
                    .collect();
                let count = scores.len();
                let mean = scores.iter().sum::<f64>() / count as f64;
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                AgeGroupStats {
                    age_group: group.clone(),
                    mean,
                    max,
                    count,
                }
            })
            .collect();
        age_stats.sort_by(|a, b| {
            age_rank(&a.age_group)
                .cmp(&age_rank(&b.age_group))
                .then_with(|| a.age_group.cmp(&b.age_group))
        });

        SurveyReport {
            missing,
            kept,
            device_counts,
            age_stats,
        }
    }
}
