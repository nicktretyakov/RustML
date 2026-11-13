use plotters::prelude::*;
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scan = array![
        [255u8, 255, 255, 255, 255],
        [255,  30,   0,   0,   0],
        [255,   0,   0,   0,   0],
        [255, 255, 255, 200,   0],
        [255,   0,   0,   0,   0],
        [255,   0,   0,   0,  15],
        [255, 255, 255, 255, 255]
    ];
    let image = scan.mapv(|v| 255 - v);
    let (h, w) = image.dim();

    let root = BitMapBackend::new("letter_e.png", (w as u32 * 40, h as u32 * 40 + 40))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Letter E", ("sans-serif", 24))
        .margin(5)
        .build_cartesian_2d(0..w as i32, 0..h as i32)?;

    chart.draw_series((0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| {
        let v = image[[y, x]];
        Rectangle::new(
            [(x as i32, y as i32), (x as i32 + 1, y as i32 + 1)],
            RGBColor(v, v, v).filled(),
        )
    }))?;

    root.present()?;
    Ok(())
}
