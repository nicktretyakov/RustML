use plotters::prelude::*;

macro_rules! scatter {
    ($chart:expr, $data:expr, $color:expr) => {{
        $chart.draw_series(
            $data
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 6, $color.filled())),
        )?
    }};
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // По одному значению для каждого филиала городской библиотеки
    let data: Vec<(f64, f64)> = vec![
        (30., 410.),
        (45., 690.),
        (38., 520.),
        (52., 800.),
        (24., 300.),
        (60., 930.),
        (41., 610.),
    ];

    let root = BitMapBackend::new("library_branches.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Library Branches", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(20f64..65f64, 250f64..1000f64)?;

    chart
        .configure_mesh()
        .x_desc("Hours open per week")
        .y_desc("Loans per week")
        .draw()?;

    scatter!(chart, data, RED);

    root.present()?;
    Ok(())
}
