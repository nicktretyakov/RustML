//! Domain model for the survey data and the aggregated report.

use serde::Deserialize;

/// A raw row read verbatim from `survey.csv`.
///
/// `score` stays a `String` because some respondents skipped the rating
/// step and left the cell empty; the [`crate::cleaner::DropMissingCleaner`]
/// decides how to interpret that.
#[derive(Debug, Clone, Deserialize)]
pub struct RawRow {
    pub device: String,
    pub age_group: String,
    pub score: String,
}

impl RawRow {
    /// Parse the score cell.
    ///
    /// Empty / whitespace-only cells map to `None` (a missing rating);
    /// anything that fails to parse is also treated as missing so a stray
    /// non-numeric value can never silently corrupt the aggregates.
    pub fn score_value(&self) -> Option<f64> {
        let trimmed = self.score.trim();
        if trimmed.is_empty() {
            None
        } else {
            trimmed.parse::<f64>().ok()
        }
    }
}

/// A cleaned, fully-answered survey row: the unit of every downstream
/// aggregation.
#[derive(Debug, Clone)]
pub struct SurveyAnswer {
    pub device: String,
    pub age_group: String,
    pub score: f64,
}

/// One row of the "answers by device" breakdown.
#[derive(Debug, Clone)]
pub struct DeviceCount {
    pub device: String,
    pub count: usize,
}

/// Per-age-group descriptive statistics.
#[derive(Debug, Clone)]
pub struct AgeGroupStats {
    pub age_group: String,
    pub mean: f64,
    pub max: f64,
    /// Number of respondents in the group. Not rendered in the default
    /// report, but kept available for richer formatters / future analyses.
    #[allow(dead_code)]
    pub count: usize,
}

/// Everything the formatter needs to render the final report.
#[derive(Debug, Clone, Default)]
pub struct SurveyReport {
    pub missing: usize,
    pub kept: usize,
    pub device_counts: Vec<DeviceCount>,
    pub age_stats: Vec<AgeGroupStats>,
}
