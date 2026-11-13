use ndarray::{array, Axis};
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Количество осадков за день (мм), 6 недель по 7 дней
    let rain = array![
        0, 0, 3, 12, 5, 0, 0,
        1, 0, 0, 0, 2, 8, 0,
        14, 25, 6, 0, 0, 3, 0,
        0, 0, 0, 4, 0, 0, 1,
        0, 7, 18, 9, 0, 0, 2,
        5, 0, 0, 0, 11, 3, 0
    ];

    let weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

    let by_weekday = rain
        .view()
        .into_shape((6, 7))?
        .mean_axis(Axis(0))
        .ok_or("Не удалось вычислить среднее")?;

    // Аналог argmax
    let (max_idx, _) = by_weekday
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    println!("Wettest: {}", weekdays[max_idx]);

    // --- Построение графиков ---
    let root = BitMapBackend::new("rain_analysis.png", (1000, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((1, 2));
    let left_area = &areas[0];
    let right_area = &areas[1];

    // 1. Столбчатая диаграмма средних значений по дням недели
    //    Явно фиксируем тип оси как i32 (0i32..7)
    let mut chart = ChartBuilder::on(left_area)
        .caption("Rain by Weekday", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0i32..7, 0i32..20)?;

    chart
        .configure_mesh()
        .x_desc("Weekday")
        .y_desc("Average rain (mm)")
        .x_labels(7)
        // Явно указываем тип параметра замыкания: &i32
        .x_label_formatter(&|x: &i32| weekdays[*x as usize].to_string())
        .draw()?;

    chart.draw_series(by_weekday.iter().enumerate().map(|(i, &y)| {
        let x0 = i as i32;
        let x1 = x0 + 1;
        Rectangle::new([(x0, 0), (x1, y as i32)], BLUE.filled())
    }))?;

    // 2. Гистограмма распределения суточных осадков (5 интервалов)
    let bins = 5usize;
    let max_val = *rain.iter().max().unwrap() as f64;
    let bin_width = (max_val / bins as f64).ceil() as i32;
    let mut counts = vec![0i32; bins];

    for &v in rain.iter() {
        let bin = ((v as f64) / bin_width as f64).floor() as usize;
        let bin = bin.min(bins - 1);
        counts[bin] += 1;
    }

    let y_max = *counts.iter().max().unwrap() + 2;

    let mut chart = ChartBuilder::on(right_area)
        .caption("Daily Rain", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0i32..bins as i32, 0i32..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Rain (mm)")
        .y_desc("Days")
        .draw()?;

    chart.draw_series(counts.iter().enumerate().map(|(i, &c)| {
        let x0 = i as i32;
        let x1 = x0 + 1;
        Rectangle::new([(x0, 0), (x1, c)], GREEN.filled())
    }))?;

    root.present()?;
    println!("График сохранён в файл 'rain_analysis.png'");

    Ok(())
}
