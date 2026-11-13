use plotters::prelude::full_palette::PURPLE; // <-- добавлено
use plotters::prelude::*;

macro_rules! dot_legend {
    ($color:expr) => {
        move |(x, y)| Circle::new((x + 10, y), 5, $color.filled())
    };
}

macro_rules! line_legend {
    ($color:expr) => {
        move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], $color)
    };
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let days: Vec<f64> = (1..=7).map(|d| d as f64).collect();
    let ana: [f64; 7] = [8.420, 11.250, 9.800, 12.400, 7.600, 10.900, 13.100];
    let ben: [f64; 7] = [6.200, 7.400, 10.100, 10.000, 9.500, 12.000, 11.300];

    let root = BitMapBackend::new("weekly_steps.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Weekly Steps", ("sans-serif", 30).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.5f64..7.5f64, 0f64..14f64)?;

    chart
        .configure_mesh()
        .x_desc("Day")
        .y_desc("Steps (thousands)")
        .x_labels(7)
        .x_label_formatter(&|x| format!("{}", x.round() as i32))
        .y_labels(8)
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            days.iter().zip(ana.iter()).map(|(&d, &a)| (d, a)),
            &PURPLE,
        ))?
        .label("Ana")
        .legend(line_legend!(PURPLE));

    chart.draw_series(
        days.iter()
            .zip(ana.iter())
            .map(|(&d, &a)| Circle::new((d, a), 4, PURPLE.filled())),
    )?;

    chart
        .draw_series(
            days.iter()
                .zip(ben.iter())
                .filter(|(_, &b)| b >= 10.0)
                .map(|(&d, &b)| Circle::new((d, b), 6, GREEN.filled())),
        )?
        .label("Ben met goal")
        .legend(dot_legend!(GREEN));

    chart
        .draw_series(
            days.iter()
                .zip(ben.iter())
                .filter(|(_, &b)| b < 10.0)
                .map(|(&d, &b)| Circle::new((d, b), 6, RED.filled())),
        )?
        .label("Ben missed")
        .legend(dot_legend!(RED));

    // Пунктирная линия цели y = 10
    let goal_dashes: Vec<_> = {
        let mut v = Vec::new();
        let mut x: f64 = 1.0; // <-- явный тип
        while x < 7.0 {
            let x2 = (x + 0.2).min(7.0);
            v.push(PathElement::new(
                vec![(x, 10.0), (x2, 10.0)],
                BLACK.stroke_width(2),
            ));
            x += 0.35;
        }
        v
    };
    chart
        .draw_series(goal_dashes)?
        .label("Goal")
        .legend(line_legend!(BLACK.stroke_width(2)));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
