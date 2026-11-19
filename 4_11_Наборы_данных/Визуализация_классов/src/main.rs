//! Breast-cancer scatter visualization in Rust.
//!
//! Replaces the teammate's single-color scatter (all 569 samples drawn as one
//! blob) with **one colored series per diagnosis class** — `malignant` vs
//! `benign` — plotting **mean radius** (x) against **mean concave points**
//! (y), each class labelled with its name from `target_names`. This answers
//! the team question: *do malignant and benign samples form separate point
//! clouds?* — both visually and numerically.
//!
//! # How the required techniques map onto this one task
//!
//! | Requirement        | Where it lives                                                   |
//! |--------------------|------------------------------------------------------------------|
//! | Multithreading     | `rayon` in `ml::StandardScaler::fit` (per-column stats),         |
//! |                    | `ml::LogisticClassifier::train`/`accuracy` (per-sample grads),  |
//! |                    | `plotting::ParallelPartitioner` (per-class & per-row).          |
//! | OOP via traits     | `DatasetSource`, `FeatureProjector`, `Partitioner`,             |
//! |                    | `ScatterRenderer`, `DiagnosisClassifier`.                       |
//! | ML crates          | `ndarray` (feature matrix / target) + a hand-written            |
//! |                    | `StandardScaler` and `LogisticClassifier` quantifying           |
//! |                    | separability on the two plotted axes.                          |
//! | Declarative macros | `feature_index!` (name → column) in `dataset`,                  |
//! |                    | `scatter_series!` (series + legend) in `plotting`.             |
//! | SOLID              | SRP per module, OCP via traits, LSP via trait objects,          |
//! |                    | ISP (small focused traits), DIP (`main` depends on traits).     |

use std::error::Error;
use std::path::PathBuf;

use plotters::style::{register_font, FontStyle};

mod dataset;
mod ml;
mod plotting;

use dataset::{
    CancerDataset, CsvDatasetSource, DatasetSource, FeatureProjector, RadiusConcaveProjector,
};
use ml::{DiagnosisClassifier, LogisticClassifier, StandardScaler};
use plotting::{ParallelPartitioner, Partitioner, PngScatterRenderer, ScatterRenderer};

fn main() -> Result<(), Box<dyn Error>> {
    // plotters ships no built-in font: register a sans-serif TTF with its
    // ab_glyph rasterizer so the title, axis labels and legend render as
    // text. The bytes are read from a SYSTEM path at runtime (Linux / macOS /
    // Windows), so the project does NOT depend on a bundled font file.
    register_system_font()?;

    // 1) Acquire data — **DIP**: `main` depends on the `DatasetSource` trait.
    let data_path = PathBuf::from("data/breast_cancer.csv");
    let dataset = CsvDatasetSource::new(data_path).load()?;
    println!(
        "Loaded {} samples × {} features; classes: {:?}",
        dataset.n_samples(),
        dataset.n_features(),
        dataset.target_names
    );

    // 2) Quantify separability on JUST the two plotted axes. If a linear
    //    classifier reaches high accuracy using only (mean radius,
    //    mean concave points), the two classes form distinct clouds — which
    //    is exactly what the scatter plot below will show.
    let projector = RadiusConcaveProjector::new();
    let x2 = select_two_features(&dataset, projector.x_index(), projector.y_index())?;
    let scaler = StandardScaler::fit(&x2);
    let x2s = scaler.transform(&x2);
    let classifier = LogisticClassifier::train(&x2s, &dataset.target, 3000, 0.5);
    let acc = classifier.accuracy(&x2s, &dataset.target);
    println!(
        "{} accuracy on (mean radius, mean concave points): {:.3} — \
         {:.0}% of samples are linearly separable on these two axes alone.",
        classifier.name(),
        acc,
        acc * 100.0
    );

    // 3) Partition into per-class point clouds — **multithreaded** (rayon).
    let series = ParallelPartitioner.partition(&dataset, &projector);
    for s in &series {
        println!(
            "  class {} ({:?}): {} points",
            s.class_id,
            s.label,
            s.points.len()
        );
    }

    // 4) Render one scatter per class, each labelled with its target name.
    let out = PathBuf::from("cells_by_diagnosis.png");
    let renderer = PngScatterRenderer::new(
        out.clone(),
        "Cells by Diagnosis".to_string(),
        projector.x_label().to_string(),
        projector.y_label().to_string(),
    );
    renderer.render(&series)?;
    println!("Wrote scatter plot: {}", out.display());

    Ok(())
}

/// Extract columns `xi` and `yi` into an `(n × 2)` feature matrix so the
/// classifier can be trained on exactly the two axes being plotted.
fn select_two_features(
    dataset: &CancerDataset,
    xi: usize,
    yi: usize,
) -> Result<ndarray::Array2<f64>, Box<dyn Error>> {
    let n = dataset.n_samples();
    let mut v = Vec::with_capacity(n * 2);
    for i in 0..n {
        v.push(*dataset.features.get((i, xi)).unwrap());
        v.push(*dataset.features.get((i, yi)).unwrap());
    }
    Ok(ndarray::Array2::from_shape_vec((n, 2), v)?)
}

/// Register a sans-serif TTF with plotters' `ab_glyph` rasterizer, loading the
/// font bytes from a **system** path at runtime so the project needs no
/// bundled font file. `register_font` requires `&'static [u8]`, so the loaded
/// `Vec<u8>` is leaked into a static slice.
///
/// Tries common locations on Linux, macOS and Windows. If none is found we
/// warn and continue — scatter points and axes still draw, only text glyphs
/// may be missing.
fn register_system_font() -> Result<(), Box<dyn Error>> {
    const CANDIDATES: [&str; 7] = [
        // Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        // macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        // Windows
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ];

    for path in CANDIDATES {
        if let Ok(bytes) = std::fs::read(path) {
            let static_bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            register_font("sans-serif", FontStyle::Normal, static_bytes)
                .map_err(|_| -> Box<dyn Error> {
                    format!("font registration failed for {path}").into()
                })?;
            println!("Registered sans-serif font from {path}");
            return Ok(());
        }
    }

    eprintln!(
        "warning: no system TTF found among the candidate paths; \
         scatter points and axes will still render, but text glyphs may be missing"
    );
    Ok(())
}
