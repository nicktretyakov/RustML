//! Histogram plotting, isolated behind a trait.
//!
//! * SRP: the plotter only draws.
//! * DIP/OCP: callers depend on `HistogramPlotter`; a new backend (SVG, terminal...) is a
//!   new impl.
//! * Mirrors the matplotlib reference: `bins=5`, `alpha=0.5`, per-group label, shared
//!   x-axis labelled "Minutes", title "Commute Times", legend with "bike"/"bus".

use std::error::Error;

use plotters::prelude::*;

use crate::models::Group;

/// Abstraction over any histogram renderer.
pub trait HistogramPlotter {
    fn plot(&self, groups: &[Group], out_path: &str) -> Result<(), Box<dyn Error>>;
}

/// `plotters`-backed bitmap renderer.
pub struct PlottersHistogram {
    pub bins: usize,
    pub alpha: f64,
    pub title: String,
    pub x_label: String,
}

impl PlottersHistogram {
    pub fn new(bins: usize, alpha: f64) -> Self {
        Self {
            bins,
            alpha,
            title: "Commute Times".to_string(),
            x_label: "Minutes".to_string(),
        }
    }
}

impl HistogramPlotter for PlottersHistogram {
    fn plot(&self, groups: &[Group], out_path: &str) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(out_path, (960, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        // Shared x-axis span so both groups overlap on the same scale.
        let g_min = groups
            .iter()
            .filter_map(|g| g.values.iter().copied().min_by(f64::total_cmp))
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);
        let g_max = groups
            .iter()
            .filter_map(|g| g.values.iter().copied().max_by(f64::total_cmp))
            .max_by(f64::total_cmp)
            .unwrap_or(1.0);
        let g_max = if g_max > g_min { g_max } else { g_min + 1.0 };

        // Tallest bin across all groups -> y-axis ceiling.
        let max_count = groups
            .iter()
            .map(|g| *bin_counts(&g.values, self.bins).iter().max().unwrap_or(&0))
            .max()
            .unwrap_or(0)
            .max(1) as u32;

        let mut chart = ChartBuilder::on(&root)
            .caption(self.title.as_str(), ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(45)
            .y_label_area_size(55)
            .build_cartesian_2d(g_min..g_max, 0u32..max_count)?;

        chart
            .configure_mesh()
            .x_desc(self.x_label.as_str())
            .y_desc("Count")
            .draw()?;

        // Two semi-transparent colours so overlapping bars blend (alpha = 0.5).
        let palette = [
            RGBAColor(54, 100, 204, self.alpha), // bike  - blue
            RGBAColor(220, 90, 50, self.alpha),  // bus   - orange/red
        ];

        for (i, group) in groups.iter().enumerate() {
            let color = palette[i % palette.len()];
            let counts = bin_counts(&group.values, self.bins);
            let (lo, hi) = group_range(&group.values);
            let width = if hi > lo {
                (hi - lo) / self.bins as f64
            } else {
                1.0
            };

            chart
                .draw_series(counts.iter().enumerate().map(|(idx, &count)| {
                    let x0 = lo + idx as f64 * width;
                    let x1 = x0 + width;
                    let mid = (x0 + x1) / 2.0;
                    let half = width / 2.0;
                    Rectangle::new([(mid - half, 0u32), (mid + half, count)], color.filled())
                }))?
                .label(group.name.as_str())
                .legend(move |coord| {
                    EmptyElement::at(coord)
                        + Rectangle::new([(0, 0), (14, 14)], color.filled())
                });
        }

        chart
            .configure_series_labels()
            .border_style(BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }
}

/// (min, max) of a slice, safe for empty input.
fn group_range(values: &[f64]) -> (f64, f64) {
    let lo = values.iter().copied().min_by(f64::total_cmp).unwrap_or(0.0);
    let hi = values.iter().copied().max_by(f64::total_cmp).unwrap_or(1.0);
    (lo, hi)
}

/// Equal-width `bins` counts over the slice's own [min, max] — same convention as
/// matplotlib's `hist(x, bins=5)` (integer bins -> per-array range).
fn bin_counts(values: &[f64], bins: usize) -> Vec<u32> {
    let (lo, hi) = group_range(values);
    let width = if hi > lo {
        (hi - lo) / bins as f64
    } else {
        1.0
    };
    let mut counts = vec![0u32; bins];
    for &x in values {
        let mut idx = if width > 0.0 {
            ((x - lo) / width) as usize
        } else {
            0
        };
        if idx >= bins {
            idx = bins - 1; // clamp the right-edge value into the last bin
        }
        counts[idx] += 1;
    }
    counts
}
