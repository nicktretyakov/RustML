use plotters::prelude::*;
use std::io::{self, Write};

/// Макрос: рисует линию по паре срезов и добавляет её в легенду.
macro_rules! add_series {
    ($chart:expr, $xs:expr, $ys:expr, $color:expr, $label:expr) => {
        $chart
            .draw_series(LineSeries::new(
                $xs.iter().zip($ys.iter()).map(|(&x, &y)| (x, y)),
                $color,
            ))?
            .label($label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], $color)
            });
    };
}

fn read_player(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ввод имён игроков
    let player1 = read_player("player1: ")?;
    let player2 = read_player("player2: ")?;

    // Данные (как в оригинале)
    let rounds:  [i32; 6] = [1, 2, 3, 4, 5, 6];
    let scores1: [i32; 6] = [8, 12, 9, 15, 14, 18];
    let scores2: [i32; 6] = [10, 9, 13, 12, 17, 16];

    // Заголовок / оси — аналог plt.title / plt.xlabel / plt.ylabel
    let root = BitMapBackend::new("game_scores.png", (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Game Scores", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..7, 0..20)?;

    chart
        .configure_mesh()
        .x_desc("Round")
        .y_desc("Points")
        .draw()?;

    // Добавление обеих линий через макрос
    add_series!(chart, rounds, scores1, &RED,  player1);
    add_series!(chart, rounds, scores2, &BLUE, player2);

    // Легенда (аналог plt.legend())
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    // Аналог plt.show(): открыть сохранённый файл в просмотрщике

    Ok(())
}
