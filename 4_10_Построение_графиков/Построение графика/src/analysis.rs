//! Orchestration — the **Dependency-Inversion** layer.
//!
//! [`RiverAnalysis`] knows only the abstractions
//! ([`DataSource`](crate::data::DataSource), [`Preprocessor`](crate::preprocess::Preprocessor),
//! [`RegressionModel`](crate::ml::RegressionModel), [`PlotStrategy`](crate::plot::PlotStrategy))
//! injected through its constructor. Concrete types are chosen once, in `main`,
//! and never leak into this module — so swapping any layer (CSV → SQL, OLS →
//! ridge, bitmap → SVG) needs no change here.
//!
//! `run` consumes `self` so the owned trait objects can be moved cleanly into
//! the two worker closures executed concurrently by [`rayon::join`]:
//!
//! * thread A fits the regression `level_cm ~ rain_mm`,
//! * thread B renders the `level_cm` vs `hour` plot.
//!
//! They share no state, so they run in parallel.

use std::fmt;

use rayon::join;

use crate::data::{DataError, DataSource};
use crate::ml::{FitReport, MlError, RegressionModel};
use crate::plot::{PlotError, PlotStrategy};
use crate::preprocess::{Preprocessor, ProcessedData};
use crate::report_metric;

#[derive(Debug)]
pub enum AnalysisError {
    Data(DataError),
    Ml(MlError),
    Plot(PlotError),
}

impl From<DataError> for AnalysisError {
    fn from(e: DataError) -> Self {
        AnalysisError::Data(e)
    }
}
impl From<MlError> for AnalysisError {
    fn from(e: MlError) -> Self {
        AnalysisError::Ml(e)
    }
}
impl From<PlotError> for AnalysisError {
    fn from(e: PlotError) -> Self {
        AnalysisError::Plot(e)
    }
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalysisError::Data(e) => write!(f, "{e}"),
            AnalysisError::Ml(e) => write!(f, "{e}"),
            AnalysisError::Plot(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AnalysisError {}

/// Summary of a completed run — handed back to the caller (or `main`) so the
/// report can be printed without this layer knowing about stdout.
#[derive(Debug, Clone)]
pub struct RunReport {
    pub n_records: usize,
    pub processed: ProcessedData,
    pub fit: FitReport,
}

impl RunReport {
    /// Pretty-print the analysis using the `report_metric!` macro.
    pub fn print(&self) {
        println!("\n=== River level analysis ===");
        report_metric!("samples", self.n_records as f64);
        report_metric!(
            "hour range",
            self.processed.hour_max - self.processed.hour_min
        );
        report_metric!("min level (cm)", self.processed.level_min);
        report_metric!("max level (cm)", self.processed.level_max);
        report_metric!("mean level (cm)", self.processed.mean_level);
        println!("\n--- OLS regression: level_cm ~ cumulative_rain_mm ---");
        report_metric!("intercept (cm)", self.fit.intercept);
        report_metric!("slope (cm/mm)", self.fit.slope);
        report_metric!("R-squared", self.fit.r_squared);
        println!(
            "model:  level_cm = {intercept:.3} + {slope:.3} * cumulative_rain_mm",
            intercept = self.fit.intercept,
            slope = self.fit.slope
        );
    }
}

/// The orchestrator. Holds injected trait objects.
pub struct RiverAnalysis {
    source: Box<dyn DataSource>,
    preprocessor: Box<dyn Preprocessor>,
    model: Box<dyn RegressionModel>,
    plotter: Box<dyn PlotStrategy>,
}

impl RiverAnalysis {
    pub fn new(
        source: impl DataSource + 'static,
        preprocessor: impl Preprocessor + 'static,
        model: impl RegressionModel + 'static,
        plotter: impl PlotStrategy + 'static,
    ) -> Self {
        Self {
            source: Box::new(source),
            preprocessor: Box::new(preprocessor),
            model: Box::new(model),
            plotter: Box::new(plotter),
        }
    }

    /// Run the full pipeline: read → preprocess → (fit ‖ plot) → report.
    pub fn run(self, csv_path: &str, out_png: &str) -> Result<RunReport, AnalysisError> {
        // 1. Load
        let records = self.source.load(csv_path)?;
        // 2. Preprocess (rayon data-parallel)
        let processed = self.preprocessor.process(&records);

        // 3. Concurrent fit + plot via rayon::join. Each closure owns its
        //    trait object, so no shared mutable state and no Sync requirement.
        let features = processed.features.clone();
        let targets = processed.targets.clone();
        let points = processed.points.clone();
        let model = self.model;
        let plotter = self.plotter;

        let (fit_res, plot_res) = join(
            move || model.fit(&features, &targets),
            move || plotter.render(&points, out_png),
        );

        let fit = fit_res?;
        plot_res?;

        Ok(RunReport {
            n_records: records.len(),
            processed,
            fit,
        })
    }
}
