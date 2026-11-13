use std::io::{self, Write};
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    print!("Enter panel name: ");
    io::stdout().flush()?;
    let mut panel = String::new();
    io::stdin().read_line(&mut panel)?;
    let panel = panel.trim();

    let hours: Vec<f64> = vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let watts: Vec<f64> = vec![40.0, 95.0, 160.0, 210.0, 255.0, 280.0, 290.0];

    let root = BitMapBackend::new("plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("{} Output", panel), ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(6.0..12.0, 0.0..300.0)?;

    chart.configure_mesh()
        .x_desc("Hour")
        .y_desc("Power (W)")
        .draw()?;

    chart.draw_series(
        hours.iter().zip(watts.iter()).map(|(&x, &y)| Circle::new((x, y), 5, RED.filled()))
    )?;

    // Draw line connecting points (equivalent to plt.plot line)
    chart.draw_series(LineSeries::new(
        hours.iter().zip(watts.iter()).map(|(&x, &y)| (x, y)),
        &RED,
    ))?;

    root.present()?;
    Ok(())
}
