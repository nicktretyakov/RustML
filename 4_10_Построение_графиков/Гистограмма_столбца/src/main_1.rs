//! Commute-time survey analyser.
//!
//! Demonstrates, in one small program:
//! * **macros**        — `spread_metric!` in `stats.rs` declares every metric.
//! * **crates**        — `csv` + `serde` (parse), `rayon` (parallel stats), `plotters` (PNG).
//! * **multithreading**— `StatsComputer` fans (group, metric) work out over `rayon`.
//! * **OOP**           — traits (`TripLoader`, `SpreadMetric`, `HistogramPlotter`) with
//!   trait objects (`Box<dyn ...>`), encapsulated structs, `impl` blocks.
//! * **SOLID**         — see the per-module doc comments.
//!
//! Answers: "whose commute time is more widely spread?" and draws a histogram that
//! mirrors the matplotlib reference (bins=5, alpha=0.5, label per group).

mod io;
mod models;
mod plot;
mod stats;

use io::{CsvTripLoader, TripLoader};
use plot::{HistogramPlotter, PlottersHistogram};
use stats::{group_by_mode, SpreadMetric, StatsComputer, Iqr, Range, StdDev};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Paths come from CLI args and default to the same names the matplotlib
    // version uses ("commute.csv" / "commute_histogram_rust.png"), so the binary
    // works from any directory on any machine — no hard-coded absolute paths.
    //   usage: commute_analyzer [commute.csv] [out.png]
    let csv_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "commute.csv".to_string());
    let png_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "commute_histogram_rust.png".to_string());

    // DIP: concrete loader behind a trait.
    let loader: Box<dyn TripLoader> = Box::new(CsvTripLoader::new());
    let trips = loader.load(&csv_path)?;
    println!("Loaded {} trips from {csv_path}", trips.len());

    let groups = group_by_mode(&trips);
    for g in &groups {
        println!("  group {:>5}: n = {}", g.name, g.values.len());
    }

    // OCP/DIP: inject metrics as trait objects. Add a metric = push one more `Box`.
    let metrics: Vec<Box<dyn SpreadMetric>> = vec![
        Box::new(StdDev),
        Box::new(Range),
        Box::new(Iqr),
    ];
    let computer = StatsComputer::new(metrics);

    // Multithreading: every (group, metric) pair runs on the rayon pool.
    let results = computer.compute(&groups);

    println!("\n=== Spread comparison ===");
    println!("  {:>5} | {:>6} | value", "group", "metric");
    println!("  -------+--------+--------");
    for r in &results {
        println!("  {:>5} | {:>6} | {:.2}", r.group, r.metric, r.value);
    }

    // Headline answer: wider distribution by standard deviation.
    let std_of = |name: &str| {
        results
            .iter()
            .find(|r| r.group == name && r.metric == "std")
            .map(|r| r.value)
    };
    if let (Some(bike_std), Some(bus_std)) = (std_of("bike"), std_of("bus")) {
        let wider = if bus_std > bike_std { "bus" } else { "bike" };
        println!(
            "\nWider spread (by std): {wider}  (bike={bike_std:.2}, bus={bus_std:.2})"
        );
    }

    // Plot: bins=5, alpha=0.5, per-group label — same parameters as the matplotlib version.
    let plotter = PlottersHistogram::new(5, 0.5);
    plotter.plot(&groups, &png_path)?;
    println!("\nSaved histogram -> {png_path}");

    Ok(())
}
