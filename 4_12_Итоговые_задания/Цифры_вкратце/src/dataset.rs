//! Data-access layer (Single Responsibility: only loads data).
//!
//! `sklearn.datasets.load_digits` is called **without** `as_frame`, so in
//! Python `.data` and `.target` are NumPy arrays. We only copy their contents
//! out into plain Rust vectors via `.tolist()` (dtype-agnostic and robust);
//! the values are bit-identical to the NumPy arrays.
use pyo3::prelude::*;

/// Interface (Interface Segregation) for anything that can hand us the
/// digits matrix together with its labels.
pub trait DigitDataset {
    /// Flat `n_samples * n_features` matrix, row-major.
    fn data(&self) -> &[f64];
    /// `n_samples` labels in `0..=9`.
    fn target(&self) -> &[i64];
    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize {
        64
    }
}

/// Concrete adapter over `sklearn.datasets.load_digits`.
///
/// Holds ownership of the copied matrix so it can be borrowed by every
/// analyzer without re-entering Python.
pub struct SklearnDigits {
    data: Vec<f64>,
    target: Vec<i64>,
    n_samples: usize,
}

impl SklearnDigits {
    pub fn load(py: Python<'_>) -> PyResult<Self> {
        // Make sure the embedded interpreter can see the venv's ML packages.
        Self::ensure_site_packages(py)?;

        let datasets = py.import("sklearn.datasets")?;
        // `as_frame` defaults to False -> `.data` and `.target` are NumPy arrays.
        let bunch = datasets.call_method1("load_digits", ())?;
        let data_obj = bunch.getattr("data")?;
        let target_obj = bunch.getattr("target")?;

        // Read the NumPy arrays out through `.tolist()` (works for any dtype).
        let data_nested: Vec<Vec<f64>> = data_obj.call_method1("tolist", ())?.extract()?;
        let target_list: Vec<i64> = target_obj.call_method1("tolist", ())?.extract()?;

        let n_samples = data_nested.len();
        let n_features = data_nested.first().map(|r| r.len()).unwrap_or(0);
        let mut data = Vec::with_capacity(n_samples * n_features);
        for row in data_nested {
            data.extend(row);
        }
        let mut target = target_list;
        target.truncate(n_samples);

        Ok(Self {
            data,
            target,
            n_samples,
        })
    }

    /// Best-effort: make sure sklearn / matplotlib / numpy are importable from
    /// the embedded interpreter by ensuring the interpreter's own site-packages
    /// directories are on `sys.path`. Discovered at runtime so it works on any
    /// CPython layout (system, miniconda, pyenv, uv, venv).
    fn ensure_site_packages(py: Python<'_>) -> PyResult<()> {
        let script = c"
import sys, os, site
path = sys.path
seen = set(path)
cands = []
try:
    cands += site.getsitepackages()
except Exception:
    pass
try:
    cands.append(site.getusersitepackages())
except Exception:
    pass
exedir = os.path.dirname(sys.executable)
pyver = 'python%d.%d' % sys.version_info[:2]
for base in (os.path.join(exedir, '..', 'lib'),
             os.path.join(exedir, '..', 'lib64'),
             os.path.join(sys.prefix, 'lib'),
             os.path.join(sys.prefix, 'lib64')):
    cands.append(os.path.join(base, pyver, 'site-packages'))
for c in cands:
    c = os.path.abspath(c)
    if os.path.isdir(c) and c not in seen:
        path.insert(0, c)
        seen.add(c)
";
        let globals = pyo3::types::PyDict::new(py);
        let _ = py.run(script, Some(&globals), Some(&globals));
        Ok(())
    }
}

impl DigitDataset for SklearnDigits {
    fn data(&self) -> &[f64] {
        &self.data
    }
    fn target(&self) -> &[i64] {
        &self.target
    }
    fn n_samples(&self) -> usize {
        self.n_samples
    }
}
