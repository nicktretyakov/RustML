use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Данные
    let heights = [2, 5, 9, 14, 18, 21];
    let weeks   = [1, 2, 3, 4, 5, 6];

    // Создаём файл для графика
    let root = BitMapBackend::new("plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Определяем диапазоны осей
    let x_min = *weeks.iter().min().unwrap() as f64;
    let x_max = *weeks.iter().max().unwrap() as f64;
    let y_min = 0f64;
    let y_max = *heights.iter().max().unwrap() as f64;

    let mut chart = ChartBuilder::on(&root)
        .caption("Рост", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Недели")
        .y_desc("Высота")
        .draw()?;

    // Аналог plt.plot(weeks, heights) — линия с точками
    chart
        .draw_series(LineSeries::new(
            weeks.iter().zip(heights.iter()).map(|(&x, &y)| (x as f64, y as f64)),
            &BLUE,
        ))?
        .label("heights")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(
            weeks
                .iter()
                .zip(heights.iter())
                .map(|(&x, &y)| Circle::new((x as f64, y as f64), 4, BLUE.filled())),
        )?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("График сохранён в plot.png");
    Ok(())
}
