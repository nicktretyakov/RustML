use plotters::prelude::*;

// 1 = пешком, 2 = на велосипеде, 3 = на автобусе, 4 = на машине
const ANSWERS: [u32; 48] = [
    1, 3, 1, 4, 3, 3, 4, 3, 4, 4, 3, 4, 4, 4, 3, 3,
    2, 2, 2, 1, 1, 3, 1, 1, 1, 2, 3, 3, 1, 3, 2, 1,
    3, 4, 3, 3, 1, 1, 1, 2, 3, 1, 3, 4, 4, 3, 2, 4,
];

/// Возвращает: сколько значений соответствует каждому коду от 1 до 4
fn count_codes(values: &[u32]) -> [usize; 4] {
    let mut counts = [0usize; 4];
    for &v in values {
        if (1..=4).contains(&v) {
            counts[(v - 1) as usize] += 1;
        }
    }
    counts
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let counts = count_codes(&ANSWERS);
    let total = ANSWERS.len() as f64;

    // доли в процентах, округление до 1 знака (аналог np.round(..., 1))
    let shares: Vec<f64> = counts
        .iter()
        .map(|&c| (c as f64 / total * 1000.0).round() / 10.0)
        .collect();

    println!("counts = {:?}", counts);
    println!(
        "shares = [{}]",
        shares
            .iter()
            .map(|v| format!("{:.1}", v))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let labels = vec!["Walk", "Bike", "Bus", "Car"];

    // Рисуем график
    let root = BitMapBackend::new("getting_to_school.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_y = shares.iter().cloned().fold(0.0_f64, f64::max).ceil() as u32 + 10;

    let mut chart = ChartBuilder::on(&root)
        .caption("Getting to School", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        // сегментированная ось X: 0..4 (по категории на каждую единицу)
        .build_cartesian_2d((0u32..4).into_segmented(), 0u32..max_y)?;

    chart
        .configure_mesh()
        .x_label_formatter(&|x| match x {
            SegmentValue::CenterOf(i) => {
                labels.get(*i as usize).copied().unwrap_or("").to_string()
            }
            _ => String::new(),
        })
        .y_desc("Students (%)")
        .draw()?;

    // Рисуем по столбику на категорию: от Exact(i) до Exact(i+1)
    chart.draw_series(shares.iter().enumerate().map(|(i, &v)| {
        Rectangle::new(
            [
                (SegmentValue::Exact(i as u32), 0u32),
                (SegmentValue::Exact(i as u32 + 1), v as u32),
            ],
            BLUE.mix(0.6).filled(),
        )
    }))?;

    root.present()?;
    println!("Plot saved to getting_to_school.png");

    Ok(())
}
