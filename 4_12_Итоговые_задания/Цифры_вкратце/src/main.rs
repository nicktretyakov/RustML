//! digits_rust — mean image of a scikit-learn digit.
//!
//! Pipeline (Dependency Inversion): `run_pipeline` depends only on the
//! `DigitDataset`, `ImageAnalyzer` and `Visualizer` traits, never on the
//! concrete structs, so any implementation can be swapped in.
//!
//! Run: `echo 4 | cargo run --release` (build with PYO3_PYTHON pointing at
//! the venv interpreter that has sklearn/numpy/matplotlib installed).
mod analysis;
mod dataset;
mod visualization;

use std::io::Read;

use pyo3::prelude::*;

use analysis::{ImageAnalyzer, MeanImageAnalyzer};
use dataset::{DigitDataset, SklearnDigits};
use visualization::{MatplotlibVisualizer, VisualReport, Visualizer};

/// Declarative macro: print a `"key: value"` diagnostic line uniformly.
macro_rules! kv {
    ($key:expr, $val:expr) => {
        println!("{}: {}", $key, $val)
    };
}

/// Declarative macro: print one `"Row i: [v0, v1, ...]"` line where each value
/// is rounded to one decimal place (mirrors `np.round(mean.reshape(8, 8), 1)`).
macro_rules! row_line {
    ($i:expr, $vals:expr) => {{
        let joined: String = $vals
            .iter()
            .map(|v| format!("{:.1}", v))
            .collect::<Vec<_>>()
            .join(", ");
        println!("Row {}: [{}]", $i, joined)
    }};
}

/// Read the target digit `d` from standard input (first whitespace-separated token).
fn read_digit() -> i64 {
    let mut s = String::new();
    let _ = std::io::stdin().read_to_string(&mut s);
    s.trim()
        .split_whitespace()
        .next()
        .and_then(|t| t.parse::<i64>().ok())
        .unwrap_or(0)
}

/// Print the description of the drawn chart, including the 8 pixel rows.
fn print_report(report: &VisualReport) {
    kv!("Charts drawn", report.charts_drawn);
    kv!("Title", report.title.as_str());
    kv!("Images", report.images);
    println!("Size: {} rows, {} columns", report.rows, report.cols);
    kv!("Color map", report.color_map.as_str());
    for r in 0..report.rows {
        let row: &[f64] = &report.mean[r * 8..r * 8 + 8];
        row_line!(r, row);
    }
}

/// Trait-based pipeline: depends only on abstractions (DIP), so any analyzer
/// or visualizer implementation can be substituted. The analyzer already
/// encapsulates its data source, so the pipeline takes only the collaborators
/// it actually uses.
fn run_pipeline<A, V>(analyzer: &A, visualizer: &V, d: i64, py: Python<'_>) -> PyResult<()>
where
    A: ImageAnalyzer,
    V: Visualizer,
{
    let mean = analyzer.average_image(d);
    let (row, col) = analyzer.brightest_pixel(&mean);
    let ink = analyzer.average_ink(&mean);

    kv!("Brightest pixel", format!("row {}, column {}", row, col));
    // `{:.2}` uses round-half-to-even, matching `numpy.round(_, 2)`.
    println!("Average ink: {:.2}", ink);

    let report = visualizer.render(&mean, d, py)?;
    print_report(&report);
    Ok(())
}

fn main() -> PyResult<()> {
    let d = read_digit();

    // Write the chart into the current working directory (portable: no hardcoded
    // absolute path). If the path ever gains a subdirectory, create it first so
    // matplotlib's `savefig` does not raise `FileNotFoundError`.
    let out_path = format!("average_{}.png", d);
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    Python::attach(|py| {
        let dataset = SklearnDigits::load(py)?;
        if dataset.n_samples() == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("empty digits dataset"));
        }
        let analyzer = MeanImageAnalyzer::new(&dataset);
        let visualizer = MatplotlibVisualizer::new(out_path);
        run_pipeline(&analyzer, &visualizer, d, py)
    })
}
