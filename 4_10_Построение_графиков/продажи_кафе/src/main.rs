//! Cafe sales analysis in Rust.
//!
//! Mirrors the pandas/matplotlib exercise:
//!   1. read `cafe.csv` (day, customers, sales);
//!   2. fill the empty day-9 `sales` cell with the mean of the remaining sales
//!      (mean imputation — a standard ML preprocessing step);
//!   3. render two subplot diagrams side by side:
//!        left  -> customers vs day line chart  ("Customers per Day")
//!        right -> sales histogram, 4 bins     ("Daily Sales")
//!
//! Design notes (the requested angle):
//!   * SOLID / OOP  — responsibilities are split across traits: `Loader`
//!     (loading, ISP), `Imputer` (ML preprocessing, strategy / DIP + OCP),
//!     `PlotPanel` (one subplot per type, SRP).
//!   * Macros       — `parallel_mean!` computes the mean of a slice.
//!   * Multithreading — the mean is reduced with `rayon`'s `par_iter`.
//!   * ML           — missing-value handling is framed as mean imputation,
//!     the simplest univariate imputer used before fitting a model.

use anyhow::Result;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::{register_font, FontStyle};
use rayon::prelude::*;
use serde::Deserialize;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Domain model
// ---------------------------------------------------------------------------

/// One row of the cafe ledger. `sales` is `Option` because day 9 is blank
/// (the cash register broke). The `csv` crate deserialises an empty field
/// into `None` automatically.
#[derive(Debug, Clone, Deserialize)]
struct CafeRow {
    day: u32,
    customers: u32,
    #[serde(default)]
    sales: Option<f64>,
}

// ---------------------------------------------------------------------------
// SOLID — single-responsibility trait #1: loading (Interface Segregation)
// ---------------------------------------------------------------------------

trait Loader {
    type Item;
    fn load(&self) -> Result<Vec<Self::Item>>;
}

/// Loads cafe rows from a CSV file. The `path` is borrowed, so the loader
/// owns no data and can be swapped for any other `Loader<CafeRow>`.
struct CsvLoader<'a> {
    path: &'a Path,
}

impl<'a> Loader for CsvLoader<'a> {
    type Item = CafeRow;

    fn load(&self) -> Result<Vec<CafeRow>> {
        let mut reader = csv::Reader::from_path(self.path)?;
        let rows: Vec<CafeRow> = reader
            .deserialize()
            .collect::<Result<_, _>>()
            .map_err(anyhow::Error::from)?;
        Ok(rows)
    }
}

// ---------------------------------------------------------------------------
// SOLID — single-responsibility trait #2: ML preprocessing (DIP + OCP)
// ---------------------------------------------------------------------------

/// An imputer replaces missing values in a slice of `Option<f64>`.
/// New strategies (median, most-frequent, k-NN, …) implement this trait
/// without changing the callers — Open/Closed.
trait Imputer {
    fn impute(&self, values: &mut [Option<f64>]);
}

/// Mean imputation: fill every `None` with the arithmetic mean of the
/// known values. This is the univariate imputer `sklearn` calls
/// `SimpleImputer(strategy="mean")`.
struct MeanImputer;

/// Custom macro: parallel mean of a `&[f64]` slice using `rayon`.
/// Reduces the slice across all CPU cores; empty slices yield 0.0.
macro_rules! parallel_mean {
    ($slice:expr) => {{
        let s: &[f64] = $slice;
        if s.is_empty() {
            0.0
        } else {
            s.par_iter().sum::<f64>() / s.len() as f64
        }
    }};
}

impl Imputer for MeanImputer {
    fn impute(&self, values: &mut [Option<f64>]) {
        // Gather the observed values, then reduce them in parallel.
        let known: Vec<f64> = values.iter().filter_map(|v| *v).collect();
        let mean = parallel_mean!(&known);
        for cell in values.iter_mut().filter(|v| v.is_none()) {
            *cell = Some(mean);
        }
    }
}

// ---------------------------------------------------------------------------
// SOLID — single-responsibility trait #3: one subplot per type (SRP)
// ---------------------------------------------------------------------------

/// Something that can draw itself onto one subplot area.
trait PlotPanel<DB: DrawingBackend> {
    fn draw(&self, area: &DrawingArea<DB, Shift>) -> Result<()>
    where
        <DB as DrawingBackend>::ErrorType: 'static;
}

/// Left subplot: customers per day, as a line chart with markers.
struct CustomersPanel<'a> {
    rows: &'a [CafeRow],
}

impl<'a, DB: DrawingBackend> PlotPanel<DB> for CustomersPanel<'a> {
    fn draw(&self, area: &DrawingArea<DB, Shift>) -> Result<()>
    where
        <DB as DrawingBackend>::ErrorType: 'static,
    {
        let max_customers = self.rows.iter().map(|r| r.customers).max().unwrap_or(0);
        let y_max = ((max_customers + 19) / 20) * 20; // round up to nearest 20
        let x_max = self.rows.len() as u32 + 1;

        let mut chart = ChartBuilder::on(area)
            .caption("Customers per Day", ("sans-serif", 24.0))
            .margin(10)
            .x_label_area_size(34)
            .y_label_area_size(44)
            .build_cartesian_2d(1u32..x_max, 0u32..y_max)?;

        chart
            .configure_mesh()
            .x_desc("day")
            .y_desc("Customers")
            .label_style(("sans-serif", 12.0))
            .draw()?;

        chart.draw_series(LineSeries::new(
            self.rows.iter().map(|r| (r.day, r.customers)),
            BLUE.stroke_width(2),
        ))?;

        chart.draw_series(
            self.rows
                .iter()
                .map(|r| Circle::new((r.day, r.customers), 3, BLUE.filled())),
        )?;

        Ok(())
    }
}

/// Right subplot: histogram of daily sales with a fixed number of bins.
struct SalesHistogramPanel<'a> {
    rows: &'a [CafeRow],
    bins: usize,
}

impl<'a, DB: DrawingBackend> PlotPanel<DB> for SalesHistogramPanel<'a> {
    fn draw(&self, area: &DrawingArea<DB, Shift>) -> Result<()>
    where
        <DB as DrawingBackend>::ErrorType: 'static,
    {
        let sales: Vec<f64> = self.rows.iter().map(|r| r.sales.unwrap()).collect();
        let min = sales.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = sales.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let width = (max - min) / self.bins as f64;

        // Equal-width binning, mirroring matplotlib's `hist(bins=4)` default
        // range [min, max]. The right edge is inclusive in the last bin.
        let mut counts = vec![0u32; self.bins];
        for &v in &sales {
            let mut idx = ((v - min) / width).floor() as usize;
            if idx >= self.bins {
                idx = self.bins - 1;
            }
            counts[idx] += 1;
        }
        let max_count = *counts.iter().max().unwrap_or(&0);

        let mut chart = ChartBuilder::on(area)
            .caption("Daily Sales", ("sans-serif", 24.0))
            .margin(10)
            .x_label_area_size(34)
            .y_label_area_size(44)
            .build_cartesian_2d(min..max, 0u32..(max_count + 1))?;

        chart
            .configure_mesh()
            .x_desc("Sales ($)")
            .y_desc("Days")
            .label_style(("sans-serif", 12.0))
            .draw()?;

        for (i, &count) in counts.iter().enumerate() {
            let lo = min + i as f64 * width;
            let hi = lo + width;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(lo, 0u32), (hi, count)],
                RED.filled(),
            )))?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    // 1. Load (single responsibility: delegated to the loader).
    let csv_path = resolve_input()?;
    let mut rows = CsvLoader { path: &csv_path }.load()?;
    println!(
        "Loaded {} cafe rows from {}",
        rows.len(),
        csv_path.display()
    );

    // 2. ML preprocessing: mean-impute the missing sales cell.
    let mut sales: Vec<Option<f64>> = rows.iter().map(|r| r.sales).collect();
    MeanImputer.impute(&mut sales);
    for (row, sales) in rows.iter_mut().zip(sales.into_iter()) {
        row.sales = sales;
    }

    // Report the table and the value used to plug the gap.
    let filled_value = rows
        .iter()
        .find(|r| r.day == 9)
        .and_then(|r| r.sales)
        .unwrap_or(f64::NAN);
    println!("\n day  customers    sales");
    println!("----------------------------");
    for r in &rows {
        println!("{:>4}  {:>9}  {:>8.2}", r.day, r.customers, r.sales.unwrap());
    }
    println!(
        "\nDay-9 sales was missing; filled with mean = {:.2}",
        filled_value
    );

    // 3. Render the two subplot diagrams side by side.
    register_default_font()?;
    let out = resolve_output();
    let root = BitMapBackend::new(&out, (1200, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let (left, right) = root.split_horizontally(600);

    CustomersPanel { rows: &rows }.draw(&left)?;
    SalesHistogramPanel { rows: &rows, bins: 4 }.draw(&right)?;

    root.present()?;
    println!("\nSaved diagram -> {}", out.display());
    Ok(())
}

/// Where to read `cafe.csv` from. Honours the `CAFE_CSV` env var, then the
/// current directory, then the crate root (set by cargo at build time).
fn resolve_input() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("CAFE_CSV") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Ok(p);
        }
    }
    for candidate in [
        PathBuf::from("cafe.csv"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cafe.csv"),
    ] {
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    anyhow::bail!(
        "cafe.csv not found. Place it next to the binary (or in the crate \
         root) or set CAFE_CSV=/path/to/cafe.csv."
    )
}

/// Where to write the PNG. Honours the `CAFE_OUT` env var, else writes
/// `cafe_plot_rust.png` in the current directory.
fn resolve_output() -> PathBuf {
    std::env::var("CAFE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("cafe_plot_rust.png"))
}

/// Register a sans-serif TTF for plotters' `ab_glyph` backend so the Latin
/// titles and axis labels rasterise without depending on fontconfig. Tries
/// the `CAFE_FONT` env var, then common Linux font locations.
fn register_default_font() -> Result<()> {
    let path = match std::env::var("CAFE_FONT") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            const CANDIDATES: &[&str] = &[
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/TTF/LiberationSans-Regular.ttf",
            ];
            CANDIDATES
                .iter()
                .map(PathBuf::from)
                .find(|p| p.exists())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "No sans-serif TTF found. Set CAFE_FONT=/path/to/font.ttf \
                         or install one (e.g. `apt install fonts-dejavu-core`)."
                    )
                })?
        }
    };

    let bytes = std::fs::read(&path)?;
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    register_font("sans-serif", FontStyle::Normal, leaked)
        .map_err(|_| anyhow::anyhow!("failed to register {} as sans-serif", path.display()))?;
    println!("Using font: {}", path.display());
    Ok(())
}
