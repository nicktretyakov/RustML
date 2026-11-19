//! Presentation layer (Single Responsibility: only draws and describes).
//!
//! A mean-image vector is turned into an 8x8 grayscale matplotlib chart and a
//! `VisualReport` describing what was drawn. matplotlib is driven through
//! PyO3; the headless `Agg` backend is selected so a chart file is produced
//! without a display.
use pyo3::prelude::*;

/// Immutable description of the chart that was produced. A plain DTO (no
/// behaviour) so the caller can render it however it likes.
#[derive(Debug, Clone)]
pub struct VisualReport {
    pub charts_drawn: usize,
    pub title: String,
    pub images: usize,
    pub rows: usize,
    pub cols: usize,
    pub color_map: String,
    /// Raw 64 pixel means; the caller formats rows with 1-decimal rounding.
    pub mean: Vec<f64>,
}

/// Interface (Interface Segregation / Open-Closed) for any renderer.
pub trait Visualizer {
    fn render(&self, mean: &[f64], digit: i64, py: Python<'_>) -> PyResult<VisualReport>;
}

/// Renderer that drives matplotlib through PyO3 and saves the figure to disk.
pub struct MatplotlibVisualizer {
    pub out_path: String,
}

impl MatplotlibVisualizer {
    pub fn new(out_path: impl Into<String>) -> Self {
        Self {
            out_path: out_path.into(),
        }
    }
}

impl Visualizer for MatplotlibVisualizer {
    fn render(&self, mean: &[f64], digit: i64, py: Python<'_>) -> PyResult<VisualReport> {
        let title = format!("Average {}", digit);

        let mpl = py.import("matplotlib")?;
        mpl.call_method1("use", ("Agg",))?; // headless backend: no display needed
        let plt = py.import("matplotlib.pyplot")?;
        let np = py.import("numpy")?;

        // Mirror `np.array(mean).reshape(8, 8)` from the reference script.
        let flat: Vec<f64> = mean.to_vec();
        let arr = np.call_method1("array", (flat,))?;
        let reshaped = arr.call_method1("reshape", ((8, 8),))?;

        let subplots = plt.call_method1("subplots", ())?;
        let fig = subplots.get_item(0)?;
        let ax = subplots.get_item(1)?;

        // `imshow(X, cmap=...)`; cmap is the 2nd positional parameter, so the
        // colour map can be passed positionally as "gray".
        let reshaped_obj = reshaped.unbind();
        let _im = ax.call_method1("imshow", (reshaped_obj, "gray"))?;
        ax.call_method1("set_title", (title.as_str(),))?;

        fig.call_method1("savefig", (self.out_path.as_str(),))?;

        Ok(VisualReport {
            charts_drawn: 1,
            title,
            images: 1,
            rows: 8,
            cols: 8,
            color_map: "gray".to_string(),
            mean: mean.to_vec(),
        })
    }
}
