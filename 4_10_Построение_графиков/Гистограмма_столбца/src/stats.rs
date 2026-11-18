//! Spread metrics + parallel computation.
//!
//! SOLID mapping:
//! * **ISP**: `SpreadMetric` exposes exactly two methods (`name`, `compute`); nothing else.
//! * **OCP**: a new metric (e.g. MAD) is a one-line `spread_metric!` invocation — no
//!   existing code changes.
//! * **LSP**: any `Box<dyn SpreadMetric>` is substitutable; `StatsComputer` does not care
//!   which concrete type it holds.
//! * **DIP**: `StatsComputer` depends on `Vec<Box<dyn SpreadMetric>>`, never on a concrete
//!   metric type.
//! * **SRP**: `group_by_mode` only groups; `StatsComputer` only orchestrates.
//!
//! Multithreading: every (group, metric) pair is evaluated on the `rayon` thread pool.

use rayon::prelude::*;

use crate::models::{Group, Mode, Trip};

/// One measure of spread. Implementations are `Send + Sync` so they can cross threads.
pub trait SpreadMetric: Send + Sync {
    fn name(&self) -> &'static str;
    fn compute(&self, values: &[f64]) -> f64;
}

/// Declarative macro that defines a `SpreadMetric` implementation from a name and a
/// formula. This is the OCP extension point: adding a metric is a single line and it
/// flows through the same generic machinery as the built-in ones.
macro_rules! spread_metric {
    ($name:ident, $label:expr, |$v:ident| $body:expr) => {
        pub struct $name;

        impl SpreadMetric for $name {
            fn name(&self) -> &'static str {
                $label
            }
            fn compute(&self, $v: &[f64]) -> f64 {
                $body
            }
        }
    };
}

// --- Built-in metrics, all produced by the macro ---------------------------

spread_metric!(StdDev, "std", |v| {
    if v.len() < 2 {
        0.0
    } else {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        variance.sqrt()
    }
});

spread_metric!(Range, "range", |v| {
    if v.is_empty() {
        0.0
    } else {
        let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    }
});

spread_metric!(Iqr, "IQR", |v| {
    if v.len() < 4 {
        0.0
    } else {
        let mut sorted = v.to_vec();
        sorted.sort_by(f64::total_cmp);
        percentile_of_sorted(&sorted, 0.75) - percentile_of_sorted(&sorted, 0.25)
    }
});

/// Linear-interpolation percentile on an already sorted slice (helper, not a metric).
fn percentile_of_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = p * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// One computed (group, metric, value) tuple.
#[derive(Debug, Clone)]
pub struct MetricResult {
    pub group: String,
    pub metric: String,
    pub value: f64,
}

/// Computes a set of metrics over a set of groups, in parallel.
///
/// Depends only on the `SpreadMetric` abstraction (DIP). The work is fan-out across
/// `rayon` so each (group, metric) pair runs on its own worker thread.
pub struct StatsComputer {
    metrics: Vec<Box<dyn SpreadMetric>>,
}

impl StatsComputer {
    pub fn new(metrics: Vec<Box<dyn SpreadMetric>>) -> Self {
        Self { metrics }
    }

    pub fn compute(&self, groups: &[Group]) -> Vec<MetricResult> {
        // Build the cartesian product of groups x metrics as independent work items.
        let work: Vec<(usize, usize)> = (0..groups.len())
            .flat_map(|gi| (0..self.metrics.len()).map(move |mi| (gi, mi)))
            .collect();

        // `par_iter` distributes the work items across the rayon thread pool.
        work.par_iter()
            .map(|&(gi, mi)| {
                let group = &groups[gi];
                let metric = &self.metrics[mi];
                MetricResult {
                    group: group.name.clone(),
                    metric: metric.name().to_string(),
                    value: metric.compute(&group.values),
                }
            })
            .collect()
    }
}

/// Split raw trips into one `Group` per `Mode` (SRP: this function only groups).
/// Group names come from `Mode::as_str` so there is a single source of truth for the
/// label strings and `as_str` is actually exercised (no dead-code warning).
pub fn group_by_mode(trips: &[Trip]) -> Vec<Group> {
    // Stable display order: bike first, then bus.
    [Mode::Bike, Mode::Bus]
        .into_iter()
        .map(|mode| {
            let values: Vec<f64> = trips
                .iter()
                .filter(|t| t.mode == mode)
                .map(|t| t.minutes)
                .collect();
            Group {
                name: mode.as_str().to_string(),
                values,
            }
        })
        .collect()
}
