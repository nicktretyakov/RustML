//! Data-acquisition layer.
//!
//! **Single Responsibility**: this module only knows how to read the
//! breast-cancer CSV and expose it as an ML feature matrix + target vector.
//!
//! **Open/Closed**: a new source (HTTP download, parquet, in-memory) can be
//! added by implementing [`DatasetSource`] without changing any caller.
//!
//! **Interface Segregation**: [`FeatureProjector`] exposes only the two axes
//! a plot needs, nothing else about the dataset.

use std::error::Error;
use std::path::PathBuf;

use csv::ReaderBuilder;
use ndarray::{Array1, Array2};

/// The 30 feature names, in dataset column order (matches scikit-learn's
/// `load_breast_cancer().feature_names`).
pub const FEATURE_NAMES: [&str; 30] = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity", "mean concave points",
    "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error", "concave points error",
    "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity", "worst concave points",
    "worst symmetry", "worst fractal dimension",
];

/// Declarative macro: look up a feature's column index by name.
///
/// Used instead of magic numbers so `mean radius` / `mean concave points` are
/// resolved by their real names, exactly as scikit-learn does.
macro_rules! feature_index {
    ($name:expr) => {{
        FEATURE_NAMES
            .iter()
            .position(|n| *n == $name)
            .unwrap_or_else(|| {
                panic!("breast_cancer: feature `{}` is not in FEATURE_NAMES", $name)
            })
    }};
}

/// Plain data: the loaded dataset (SRP — a value object, no behaviour).
#[derive(Debug, Clone)]
pub struct CancerDataset {
    /// `(n_samples, 30)` feature matrix.
    pub features: Array2<f64>,
    /// `(n_samples,)` class id per row — `0 = malignant`, `1 = benign`.
    pub target: Array1<usize>,
    /// Class labels taken from the CSV header (`["malignant", "benign"]`).
    pub target_names: Vec<String>,
}

impl CancerDataset {
    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }
    pub fn feature_names() -> &'static [&'static str] {
        &FEATURE_NAMES
    }
}

/// **Dependency Inversion**: callers depend on this trait, not on a concrete
/// CSV reader.
pub trait DatasetSource {
    fn load(&self) -> Result<CancerDataset, Box<dyn Error>>;
}

/// Loads the scikit-learn `breast_cancer.csv` from disk.
pub struct CsvDatasetSource {
    path: PathBuf,
}

impl CsvDatasetSource {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl DatasetSource for CsvDatasetSource {
    fn load(&self) -> Result<CancerDataset, Box<dyn Error>> {
        // Self-contained: if the local file is missing, download it from the
        // official scikit-learn repository so `cargo run` works even when
        // `data/` was not copied alongside the source.
        ensure_data_file(self.path.as_path())?;

        let mut rdr = ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_path(&self.path)?;
        let mut records = rdr.records();

        // First record is the scikit-learn header:
        //   n_samples,n_features,name_class_0,name_class_1
        let header = records
            .next()
            .ok_or("breast_cancer.csv: file is empty")??;
        if header.len() < 4 {
            return Err("breast_cancer.csv: malformed header".into());
        }
        let target_names = vec![header[2].to_string(), header[3].to_string()];

        let mut features: Vec<f64> = Vec::new();
        let mut target: Vec<usize> = Vec::new();
        for rec in records {
            let rec = rec?;
            if rec.len() < 31 {
                return Err(format!(
                    "breast_cancer.csv: short row with {} fields (expected 31)",
                    rec.len()
                )
                .into());
            }
            for i in 0..30 {
                features.push(rec[i].trim().parse::<f64>()?);
            }
            target.push(rec[30].trim().parse::<usize>()?);
        }

        let n = target.len();
        let features = Array2::from_shape_vec((n, 30), features)?;
        let target = Array1::from_vec(target);
        Ok(CancerDataset {
            features,
            target,
            target_names,
        })
    }
}

/// Ensure the dataset file exists locally. If it does not, download it from
/// the official scikit-learn repository (so the program is self-contained and
/// `cargo run` works without copying `data/`).
fn ensure_data_file(path: &std::path::Path) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        return Ok(());
    }
    const URL: &str = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/breast_cancer.csv";
    eprintln!(
        "data file not found at {}; downloading from scikit-learn...",
        path.display()
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let response = ureq::get(URL).call()?;
    let mut reader = response.into_reader();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut bytes)?;
    std::fs::write(path, &bytes)?;
    eprintln!("downloaded {} bytes to {}", bytes.len(), path.display());
    Ok(())
}

/// **Interface Segregation**: a projector exposes only the two axes to plot.
pub trait FeatureProjector: Sync {
    fn x_index(&self) -> usize;
    fn y_index(&self) -> usize;
    fn x_label(&self) -> &'static str;
    fn y_label(&self) -> &'static str;

    /// Project one feature row onto the two plot axes.
    fn project(&self, features_row: &[f64]) -> (f64, f64) {
        (
            features_row[self.x_index()],
            features_row[self.y_index()],
        )
    }
}

/// Projects onto `mean radius` (x) × `mean concave points` (y),
/// exactly as required by the task.
pub struct RadiusConcaveProjector {
    x_idx: usize,
    y_idx: usize,
}

impl RadiusConcaveProjector {
    pub fn new() -> Self {
        Self {
            x_idx: feature_index!("mean radius"),
            y_idx: feature_index!("mean concave points"),
        }
    }
}

impl Default for RadiusConcaveProjector {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureProjector for RadiusConcaveProjector {
    fn x_index(&self) -> usize {
        self.x_idx
    }
    fn y_index(&self) -> usize {
        self.y_idx
    }
    fn x_label(&self) -> &'static str {
        "mean radius"
    }
    fn y_label(&self) -> &'static str {
        "mean concave points"
    }
}
