//! Visualization layer.
//!
//! **SRP**: partitioning rows into point clouds and rendering pixels to a
//! file are separate concerns, each behind its own trait.
//!
//! **OCP**: a new [`Partitioner`] (e.g. a sequential one for tests) or a new
//! [`ScatterRenderer`] (e.g. SVG) can be swapped in without touching callers.
//!
//! **Multithreading**: [`ParallelPartitioner`] collects every class's point
//! cloud, and every point inside each cloud, concurrently with `rayon`.

use std::error::Error;
use std::path::PathBuf;

use plotters::prelude::*;
use rayon::prelude::*;

use crate::dataset::{CancerDataset, FeatureProjector};

/// One colored point cloud on the scatter plot.
pub struct ClassSeries {
    pub class_id: usize,
    pub label: String,
    pub points: Vec<(f64, f64)>,
}

/// Splits the dataset into per-class 2-D point clouds.
pub trait Partitioner {
    fn partition(
        &self,
        dataset: &CancerDataset,
        projector: &dyn FeatureProjector,
    ) -> Vec<ClassSeries>;
}

/// Multithreaded partitioner (`rayon`): the outer loop runs over classes in
/// parallel, and inside each class the rows are projected in parallel too.
pub struct ParallelPartitioner;

impl Partitioner for ParallelPartitioner {
    fn partition(
        &self,
        dataset: &CancerDataset,
        projector: &dyn FeatureProjector,
    ) -> Vec<ClassSeries> {
        let n_classes = dataset.target_names.len();
        let xi = projector.x_index();
        let yi = projector.y_index();

        // Group row indices by class (single sequential pass, O(n)).
        let mut idx_by_class: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (i, &c) in dataset.target.iter().enumerate() {
            idx_by_class[c].push(i);
        }

        idx_by_class
            .par_iter() // <-- parallel over classes
            .enumerate()
            .map(|(class_id, indices)| {
                let points: Vec<(f64, f64)> = indices
                    .par_iter() // <-- parallel over rows of this class
                    .map(|&i| {
                        (
                            *dataset.features.get((i, xi)).unwrap(),
                            *dataset.features.get((i, yi)).unwrap(),
                        )
                    })
                    .collect();
                ClassSeries {
                    class_id,
                    label: dataset.target_names[class_id].clone(),
                    points,
                }
            })
            .collect()
    }
}

/// Renders point clouds to an image file.
pub trait ScatterRenderer {
    fn render(&self, series: &[ClassSeries]) -> Result<(), Box<dyn Error>>;
}

/// Renders to a PNG via plotters' bitmap backend.
pub struct PngScatterRenderer {
    path: PathBuf,
    title: String,
    x_label: String,
    y_label: String,
}

impl PngScatterRenderer {
    pub fn new(path: PathBuf, title: String, x_label: String, y_label: String) -> Self {
        Self {
            path,
            title,
            x_label,
            y_label,
        }
    }
    pub fn path(&self) -> &std::path::Path {
        self.path.as_path()
    }
}

/// Declarative macro: add one colored scatter series to a plotters chart and
/// register its legend entry in one call. Removes the repetitive
/// `draw_series(...).label(...).legend(...)` boilerplate repeated per class.
macro_rules! scatter_series {
    ($chart:expr, $points:expr, $color:expr, $label:expr) => {{
        let points: &[(f64, f64)] = $points;
        let color: RGBColor = $color;
        $chart
            .draw_series(points.iter().map(|&(x, y)| Circle::new((x, y), 3, color.filled())))
            .map_err(|e| -> Box<dyn std::error::Error> {
                format!("draw_series failed: {e:?}").into()
            })?
            .label($label.to_string())
            .legend(move |coord| Circle::new(coord, 3, color.filled()));
    }};
}

/// Palette: malignant = red, benign = green (matches the usual "bad/good"
/// convention and keeps the two clouds visually unmistakable).
const CLASS_COLORS: [RGBColor; 2] = [RGBColor(231, 76, 60), RGBColor(46, 204, 113)];

impl ScatterRenderer for PngScatterRenderer {
    fn render(&self, series: &[ClassSeries]) -> Result<(), Box<dyn Error>> {
        let (xmin, xmax, ymin, ymax) = compute_bounds(series);

        let root = BitMapBackend::new(self.path.as_path(), (960, 640)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(self.title.as_str(), ("sans-serif", 28))
            .margin(20)
            .x_label_area_size(48)
            .y_label_area_size(64)
            .build_cartesian_2d(xmin..xmax, ymin..ymax)?;

        chart
            .configure_mesh()
            .x_desc(self.x_label.as_str())
            .y_desc(self.y_label.as_str())
            .draw()?;

        // One scatter series per class, each labelled with its target name.
        for s in series {
            let color = CLASS_COLORS[s.class_id % CLASS_COLORS.len()];
            scatter_series!(chart, &s.points, color, &s.label);
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        drop(root);
        Ok(())
    }
}

/// Compute axis bounds with a small 5 % margin so no point sits on an edge.
fn compute_bounds(series: &[ClassSeries]) -> (f64, f64, f64, f64) {
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for s in series {
        for &(x, y) in &s.points {
            if x < xmin {
                xmin = x;
            }
            if x > xmax {
                xmax = x;
            }
            if y < ymin {
                ymin = y;
            }
            if y > ymax {
                ymax = y;
            }
        }
    }
    let dx = (xmax - xmin) * 0.05 + 1e-6;
    let dy = (ymax - ymin) * 0.05 + 1e-6;
    (xmin - dx, xmax + dx, ymin - dy, ymax + dy)
}
