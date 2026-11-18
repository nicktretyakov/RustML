//! Commute-time survey analyser.
//!
//! macros (spread_metric!), crates (csv/serde/rayon/plotters), multithreading (rayon),
//! OOP (traits + Box<dyn ...>), SOLID — see per-module doc comments.

mod io;
mod models;
mod plot;
mod stats;

use io::{CsvTripLoader, TripLoader};
use plot::{HistogramPlotter, PlottersHistogram};
use stats::{group_by_mode, SpreadMetric, StatsComputer, Iqr, Range, StdDev};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Paths from CLI args, defaulting to the same names the matplotlib version uses,
    // so the binary works from any directory on any machine — no hard-coded paths.
    //   usage: commute_analyzer [commute.csv] [out.png]
    let csv_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "commute.csv".to_string());
    let png_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "commute_histogram_rust.png".to_string());

    let loader: Box<dyn TripLoader> = Box::new(CsvTripLoader::new());
    let trips = loader.load(&csv_path)?;
    println!("Loaded {} trips from {csv_path}", trips.len());

    let groups = group_by_mode(&trips);
    for g in &groups {
        println!("  group {:>5}: n = {}", g.name, g.values.len());
    }

    let metrics: Vec<Box<dyn SpreadMetric>> = vec![
        Box::new(StdDev),
        Box::new(Range),
        Box::new(Iqr),
    ];
    let computer = StatsComputer::new(metrics);
    let results = computer.compute(&groups);

    println!("\n=== Spread comparison ===");
    println!("  {:>5} | {:>6} | value", "group", "metric");
    println!("  -------+--------+--------");
    for r in &results {
        println!("  {:>5} | {:>6} | {:.2}", r.group, r.metric, r.value);
    }

    let std_of = |name: &str| {
        results
            .iter()
            .find(|r| r.group == name && r.metric == "std")
            .map(|r| r.value)
    };
    if let (Some(bike_std), Some(bus_std)) = (std_of("bike"), std_of("bus")) {
        let wider = if bus_std > bike_std { "bus" } else { "bike" };
        println!("\nWider spread (by std): {wider}  (bike={bike_std:.2}, bus={bus_std:.2})");
    }

    let plotter = PlottersHistogram::new(5, 0.5);
    plotter.plot(&groups, &png_path)?;
    println!("\nSaved histogram -> {png_path}");

    Ok(())
}
