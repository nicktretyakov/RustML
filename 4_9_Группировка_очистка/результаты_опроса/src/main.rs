//! Entry point.
//!
//! The pipeline is composed from trait objects (Dependency Inversion): `main`
//! knows the traits, not the concrete types behind them. The report goes to
//! stdout; the optional machine-learning diagnostics go to stderr so a plain
//! `cargo run > report.txt` yields a pristine report.
//!
//! Run it with:
//!     cargo run                       # report on stdout, ML notes on stderr
//!     cargo run -- /path/to/survey.csv # read a different file
//!     cargo run > report.txt          # capture only the clean report

mod analyzer;
mod cleaner;
mod ml;
mod models;
mod reader;
mod reporter;
mod traits;

use std::path::PathBuf;

use analyzer::ParallelSurveyAnalyzer;
use cleaner::DropMissingCleaner;
use ml::LinearScoreRegressor;
use reader::CsvSurveyReader;
use reporter::PandasStyleFormatter;
use traits::{DataCleaner, ReportFormatter, SatisfactionModel, SurveyAnalyzer, SurveyReader};

/// Resolve the CSV path: explicit CLI arg, otherwise `survey.csv` shipped
/// next to `Cargo.toml` (works no matter what the current directory is).
fn resolve_csv_path() -> PathBuf {
    if let Some(arg) = std::env::args().nth(1) {
        return PathBuf::from(arg);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("survey.csv")
}

fn main() {
    let csv_path = resolve_csv_path();

    // Compose the pipeline from trait objects (DIP / LSP): any of these can
    // be swapped for another implementation without touching `main`.
    let reader: Box<dyn SurveyReader> = Box::new(CsvSurveyReader::new(&csv_path));
    let cleaner: Box<dyn DataCleaner> = Box::new(DropMissingCleaner::new());
    let analyzer: Box<dyn SurveyAnalyzer> = Box::new(ParallelSurveyAnalyzer::new());
    let formatter: Box<dyn ReportFormatter> = Box::new(PandasStyleFormatter::new());
    let model: Box<dyn SatisfactionModel> = Box::new(LinearScoreRegressor::new());

    // --- run the pipeline -------------------------------------------------
    let raw = reader.load().unwrap_or_else(|e| {
        eprintln!("error: could not read {}: {}", csv_path.display(), e);
        std::process::exit(1);
    });
    let (answers, missing) = cleaner.clean(raw);
    let report = analyzer.analyze(&answers, missing);

    // --- auxiliary ML diagnostics (stderr only) ---------------------------
    if let Some(fit) = model.fit(&answers) {
        eprintln!(
            "[ml] linear regression trained on {} samples, {} features [{}]",
            fit.n_samples,
            fit.n_features,
            fit.feature_names.join(", ")
        );
        eprintln!("[ml] MSE = {:.4}, R^2 = {:.4}", fit.mse, fit.r2);
    } else {
        eprintln!("[ml] not enough samples to train the regression model");
    }

    // --- the clean report (stdout) ----------------------------------------
    print!("{}", formatter.format(&report));
    println!();
}
