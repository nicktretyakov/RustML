//! Plotting layer.
//!
//! Reproduces the reference matplotlib snippet
//!
//! ```python
//! fig, ax = plt.subplots()
//! river.plot(x="hour", y="level_cm", ax=ax)
//! ax.set_title("River Level")
//! ax.set_ylabel("Level (cm)")
//! ```
//!
//! using the [`plotters`](https://docs.rs/plotters) crate. The [`PlotStrategy`]
//! trait is the open/closed seam: a `SvgPlot` or an interactive backend could
//! be added later without modifying the orchestrator.

use std::fmt;
use std::path::Path;

use plotters::prelude::*;

/// Anything able to render a sequence of `(x, y)` points to an image file.
pub trait PlotStrategy: Send {
    fn render(&self, points: &[(f64, f64)], out_path: &str) -> Result<(), PlotError>;
}

#[derive(Debug)]
pub enum PlotError {
    NoData,
    Render(String),
}

impl fmt::Display for PlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlotError::NoData => write!(f, "plot error: no points to draw"),
            PlotError::Render(m) => write!(f, "plot error: {m}"),
        }
    }
}

impl std::error::Error for PlotError {}

/// Bitmap line plot matching the upstream sensor's matplotlib look:
/// blue line, title "River Level", y-axis "Level (cm)", x-axis "hour".
#[derive(Debug, Clone, Copy)]
pub struct RiverLevelPlot {
    pub width: u32,
    pub height: u32,
}

impl RiverLevelPlot {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Default 800×500 canvas.
    pub fn default_size() -> Self {
        Self::new(800, 500)
    }
}

impl PlotStrategy for RiverLevelPlot {
    fn render(&self, points: &[(f64, f64)], out_path: &str) -> Result<(), PlotError> {
        if points.is_empty() {
            return Err(PlotError::NoData);
        }

        // Axis ranges with a little padding so the line never kisses the frame.
        let (hour_min, hour_max) = bounds(points.iter().map(|(h, _)| *h));
        let (level_min, level_max) = bounds(points.iter().map(|(_, l)| *l));
        let pad_x = ((hour_max - hour_min) * 0.02).max(0.5);
        let pad_y = ((level_max - level_min) * 0.05).max(1.0);

        let root = BitMapBackend::new(Path::new(out_path), (self.width, self.height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| PlotError::Render(format!("{e:?}")))?;

        let mut chart = ChartBuilder::on(&root)
            .caption("River Level", ("sans-serif", 36).into_font())
            .margin(12)
            .x_label_area_size(44)
            .y_label_area_size(64)
            .build_cartesian_2d(
                (hour_min - pad_x)..(hour_max + pad_x),
                (level_min - pad_y)..(level_max + pad_y),
            )
            .map_err(|e| PlotError::Render(format!("{e:?}")))?;

        chart
            .configure_mesh()
            .x_desc("hour")
            .y_desc("Level (cm)")
            .x_labels(12)
            .y_labels(8)
            .draw()
            .map_err(|e| PlotError::Render(format!("{e:?}")))?;

        chart
            .draw_series(LineSeries::new(
                points.iter().copied(),
                &BLUE,
            ))
            .map_err(|e| PlotError::Render(format!("{e:?}")))?;

        root.present()
            .map_err(|e| PlotError::Render(format!("{e:?}")))?;
        // Drop early to flush the bitmap backend.
        drop(root);
        Ok(())
    }
}

fn bounds(iter: impl Iterator<Item = f64>) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in iter {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}
