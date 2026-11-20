//! Output formatting.
//!
//! Single Responsibility (SOLID-S): this module only turns numbers into the
//! exact string the assignment expects. It owns no business logic and is the
//! sole place that knows the human-facing format, so restyling the report
//! never touches statistics code.

use crate::models::HousePrices;
use crate::stats::StatisticsEngine;

/// Builds the human-readable report.
pub trait ReportFormatter {
    fn format(&self, prices: &HousePrices, engine: &dyn StatisticsEngine) -> String;
}

/// Formats the three required lines, mirroring the reference Python output:
///
/// ```text
/// Mean: 753.0
/// Median: 320.0
/// 4 of 5 houses cost less than the mean
/// ```
pub struct MeanMedianReportFormatter;

impl MeanMedianReportFormatter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MeanMedianReportFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl ReportFormatter for MeanMedianReportFormatter {
    fn format(&self, prices: &HousePrices, engine: &dyn StatisticsEngine) -> String {
        let mean = engine.mean(prices);
        let median = engine.median(prices);
        let below = engine.count_below_mean(prices);
        let total = prices.len();

        // `{:.1}` rounds to one decimal place, matching `np.round(x, 1)`.
        format!(
            "Mean: {:.1}\nMedian: {:.1}\n{} of {} houses cost less than the mean",
            mean, median, below, total
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::ParallelStatisticsEngine;
    use ndarray::Array1;

    #[test]
    fn report_matches_expected_output() {
        let prices = HousePrices::new(Array1::from(vec![
            320.0, 300.0, 2500.0, 310.0, 335.0,
        ]));
        let engine = ParallelStatisticsEngine::new();
        let formatter = MeanMedianReportFormatter::new();

        assert_eq!(
            formatter.format(&prices, &engine),
            "Mean: 753.0\nMedian: 320.0\n4 of 5 houses cost less than the mean"
        );
    }
}
