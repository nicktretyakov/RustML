use plotters::prelude::*;
use plotters::series::DashedLineSeries;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Данные
    let hours:    [f32; 7] = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
    let measured: [f32; 7] = [17.5, 19.0, 20.5, 21.5, 22.0, 21.0, 20.0];
    let target:   [f32; 7] = [20.0; 7];

    // "Figure"
    let root = BitMapBackend::new("room_temperature.png", (800, 600))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Room Temperature", ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(6.0f32..18.0f32, 16.0f32..24.0f32)?;

    chart
        .configure_mesh()
        .x_desc("Hour")
        .y_desc("Temperature (C)")
        .draw()?;

    // --- "Измените только две строки ниже" ---

    // plt.plot(hours, measured, "rs-", label="Measured")
    let measured_iter = hours.iter().copied().zip(measured.iter().copied());
    chart
        .draw_series(LineSeries::new(measured_iter.clone(), &RED))?
        .label("Measured")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
    // Квадратные маркеры (аналог "s" в matplotlib)
    chart.draw_series(PointSeries::of_element(
        measured_iter,
        3,
        &RED,
        &|c, _s, st| {
            Rectangle::new(
                [(c.0 - 0.15, c.1 - 0.15), (c.0 + 0.15, c.1 + 0.15)],
                st.filled(),
            )
        },
    ))?;

    // plt.plot(hours, target, color="black", linestyle="--", label="Target")
    chart
        .draw_series(DashedLineSeries::new(
            hours.iter().copied().zip(target.iter().copied()),
            5,                      // длина штриха
            3,                      // длина пробела
            BLACK.stroke_width(2),
        ))?
        .label("Target")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

    // --- "Не изменяйте строки ниже" ---

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
