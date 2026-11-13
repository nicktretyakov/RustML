use ferray_random::default_rng_seeded;
use plotters::prelude::*;
use std::error::Error;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error>> {
    // ----- Ввод количества корзин -----
    print!("bins = ");
    io::stdout().flush()?;
    let mut buf = String::new();
    io::stdin().read_line(&mut buf)?;
    let bins: usize = buf.trim().parse()?;

    // ----- Генерация весов -----
    let mut rng = default_rng_seeded(11);            // np.random.default_rng(11)
    let arr = rng.normal(150.0, 20.0, 120)?;         // rng.normal(150, 20, size=120)
    let weights: Vec<f64> = arr.as_slice().unwrap().to_vec();

    // ----- Ручной hist (равные интервалы [min, max], как в numpy) -----
    let w_min = weights.iter().cloned().fold(f64::INFINITY, f64::min);
    let w_max = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_w = (w_max - w_min) / bins as f64;

    let edges: Vec<f64> = (0..=bins).map(|i| w_min + i as f64 * bin_w).collect();
    let mut counts = vec![0u32; bins];

    for &w in &weights {
        let mut idx = ((w - w_min) / bin_w).floor() as isize;
        if idx < 0              { idx = 0; }
        if idx >= bins as isize { idx = bins as isize - 1; }
        counts[idx as usize] += 1;
    }

    // ----- Самая заполненная корзина (первое вхождение максимума, как argmax) -----
    let (i, &c) = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &v)| v)
        .unwrap();

    println!(
        "Fullest bin: {:.1} to {:.1} g, {} apples",
        edges[i], edges[i + 1], c
    );

    // ----- Отрисовка и показ (аналог ax.hist + plt.show) -----
    let path = "apple_weights.png";
    draw_hist(&edges, &counts, path)?;
    opener::open(path)?;
    Ok(())
}

fn draw_hist(
    edges: &[f64],
    counts: &[u32],
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let ymax = *counts.iter().max().unwrap() as i32;
    let xmin = edges[0];
    let xmax = *edges.last().unwrap();

    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Apple Weights", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(xmin..xmax, 0..ymax)?;

    chart
        .configure_mesh()
        .x_desc("Weight (g)")
        .y_desc("Apples")
        .draw()?;

    chart.draw_series(
        counts.iter().enumerate().map(|(i, &c)| {
            Rectangle::new(
                [(edges[i], 0), (edges[i + 1], c as i32)],
                BLUE.mix(0.6).filled(),
            )
        }),
    )?;

    root.present()?;
    Ok(())
}
