use plotters::prelude::*;

// Макрос для компактного описания пар (длина, вес)
macro_rules! fish {
    ($($len:expr => $weight:expr),* $(,)?) => {
        vec![$(($len as f64, $weight as f64)),*]
    };
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Каждая строка соответствует одной рыбе в порядке вылова: длина (cm), вес (g)
    let fish = fish![
        22 => 180,
        30 => 390,
        25 => 250,
        35 => 620,
        28 => 330,
        40 => 870,
        19 => 120,
        33 => 510,
    ];

    // Создаём область рисования (аналог фигуры matplotlib)
    let root = BitMapBackend::new("fish_catch.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Строим систему координат с подписями (аналог plt.title/xlabel/ylabel)
    let mut chart = ChartBuilder::on(&root)
        .caption("Fish Catch", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(15.0..45.0, 100.0..900.0)?;

    chart
        .configure_mesh()
        .x_desc("Length (cm)")
        .y_desc("Weight (g)")
        .draw()?;

    // Аналог plt.scatter(fish[:, 0], fish[:, 1])
    chart.draw_series(
        fish.iter()
            .map(|&(len, weight)| Circle::new((len, weight), 5, RED.filled())),
    )?;

    // Аналог plt.show() — сохраняем результат
    root.present()?;
    println!("График сохранён в fish_catch.png");

    Ok(())
}
