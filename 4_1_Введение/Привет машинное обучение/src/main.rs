use linfa::dataset::Records;
use linfa_nn::{distance::L2Dist, CommonNearestNeighbour, NearestNeighbour};
use plotters::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;

const CLASS_NAMES: [&str; 3] = ["setosa", "versicolor", "virginica"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Загрузка датасета Iris через linfa-datasets
    let mut rng = SmallRng::seed_from_u64(0);
    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.75);

    let train_records = train.records().clone(); // Array2<f64>
    let train_targets = train.targets().clone(); // Array1<usize>

    // 2. Построение индекса ближайших соседей (K-D tree) на train-выборке
    let nn = CommonNearestNeighbour::KdTree
        .from_batch(&train_records, L2Dist)
        .unwrap();

    // 3. KNN-предсказание (k = 5)
    let k = 5;
    let mut guesses = Vec::with_capacity(test.nsamples());

    for point in test.records().rows() {
        let neighbors = nn.k_nearest(point, k).unwrap();

        let mut votes = [0usize; 3];
        for (_, idx) in neighbors {
            votes[train_targets[idx]] += 1;
        }

        let prediction = votes
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap();

        guesses.push(prediction);
    }

    // 4. Оценка качества
    let test_targets = test.targets();
    let right = guesses
        .iter()
        .zip(test_targets.iter())
        .filter(|(g, t)| **g == **t)
        .count();

    println!("Studied {} flowers", train.nsamples());
    println!("Named {} of {} new flowers correctly", right, test.nsamples());

    // 5. Визуализация
    let root = BitMapBackend::new("iris_linfa.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let petal_lengths: Vec<f64> = test.records().column(2).to_vec();
    let petal_widths: Vec<f64> = test.records().column(3).to_vec();

    let x_min = petal_lengths.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = petal_lengths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = petal_widths.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = petal_widths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "The model's guesses for new flowers (Linfa)",
            ("sans-serif", 20),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - 0.2)..(x_max + 0.2),
            (y_min - 0.2)..(y_max + 0.2),
        )?;

    chart
        .configure_mesh()
        .x_desc("petal length (cm)")
        .y_desc("petal width (cm)")
        .draw()?;

    let colors = [BLUE, GREEN, MAGENTA];

    for class_idx in 0..3 {
        let points: Vec<(f64, f64)> = test
            .records()
            .rows()
            .into_iter()
            .zip(guesses.iter())
            .filter(|(_, g)| **g == class_idx)
            .map(|(row, _)| (row[2], row[3]))
            .collect();

        if !points.is_empty() {
            chart
                .draw_series(
                    points
                        .iter()
                        .map(|(x, y)| Circle::new((*x, *y), 5, colors[class_idx].filled())),
                )?
                .label(format!("guessed {}", CLASS_NAMES[class_idx]))
                .legend(move |(x, y)| Circle::new((x + 10, y), 5, colors[class_idx].filled()));
        }
    }

    let mistakes: Vec<(f64, f64)> = test
        .records()
        .rows()
        .into_iter()
        .zip(guesses.iter())
        .zip(test_targets.iter())
        .filter(|((_, g), t)| **g != **t)
        .map(|((row, _), _)| (row[2], row[3]))
        .collect();

    if !mistakes.is_empty() {
        chart
            .draw_series(
                mistakes
                    .iter()
                    .map(|(x, y)| Cross::new((*x, *y), 8, RED.stroke_width(2))),
            )?
            .label("mistake")
            .legend(|(x, y)| Cross::new((x + 10, y), 5, RED.stroke_width(2)));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Saved plot to iris_linfa.png");

    Ok(())
}
