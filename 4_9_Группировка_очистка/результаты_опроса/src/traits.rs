//! Small, focused traits -- one responsibility each (Interface Segregation
//! Principle) that the rest of the program depends on (Dependency Inversion
//! Principle).

use crate::models::{RawRow, SurveyAnswer, SurveyReport};

/// Reads raw survey rows from some source (a file, a network socket, ...).
pub trait SurveyReader {
    fn load(&self) -> Result<Vec<RawRow>, Box<dyn std::error::Error + Send + Sync>>;
}

/// Turns raw rows into clean answers, reporting how many were dropped.
pub trait DataCleaner {
    fn clean(&self, rows: Vec<RawRow>) -> (Vec<SurveyAnswer>, usize);
}

/// Aggregates clean answers into a report model.
pub trait SurveyAnalyzer {
    fn analyze(&self, answers: &[SurveyAnswer], missing: usize) -> SurveyReport;
}

/// Renders a report model to a string.
///
/// Open/Closed Principle: a new output format (JSON, Markdown, CSV, ...) is
/// a fresh implementation of this trait -- no existing code has to change.
pub trait ReportFormatter {
    fn format(&self, report: &SurveyReport) -> String;
}

/// Optional auxiliary machine-learning model trained over the cleaned
/// answers. Kept on a separate trait so a report-only caller never has to
/// depend on it (Interface Segregation).
pub trait SatisfactionModel {
    fn fit(&self, answers: &[SurveyAnswer]) -> Option<FitSummary>;
}

/// Human-readable summary of a trained model.
#[derive(Debug, Clone)]
pub struct FitSummary {
    pub n_samples: usize,
    pub n_features: usize,
    pub feature_names: Vec<String>,
    pub mse: f64,
    pub r2: f64,
}
