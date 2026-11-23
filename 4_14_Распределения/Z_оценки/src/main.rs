use ndarray::prelude::*;
use linfa::prelude::*;
use linfa_preprocessing::linear_scaling::LinearScaler;
use rayon::prelude::*;
use std::io;

/// Общий трейт для любой игры: хранит набор результатов.
/// Принцип разделения интерфейсов (ISP) – только необходимый метод.
trait GameData {
    fn scores(&self) -> &Array1<f64>;
}

/// Игра-головоломка.
struct Puzzle {
    scores: Array1<f64>,
}

impl GameData for Puzzle {
    fn scores(&self) -> &Array1<f64> {
        &self.scores
    }
}

/// Гоночная игра.
struct Racing {
    scores: Array1<f64>,
}

impl GameData for Racing {
    fn scores(&self) -> &Array1<f64> {
        &self.scores
    }
}

/// Вычисляет z-оценку с помощью `LinearScaler` из `linfa`.
/// Принцип инверсии зависимостей (DIP): функция зависит от абстракции `GameData`,
/// а не от конкретных структур.
fn z_score_linfa(scores: &Array1<f64>, value: f64) -> f64 {
    // Преобразуем тренировочные данные в матрицу (n_samples × 1).
    let train = scores.to_owned().insert_axis(Axis(1)); // Array2<f64>
    // Обучаем стандартный scaler (вычитает среднее и делит на std).
    let scaler = LinearScaler::standard()
        .fit(&Dataset::from(train))
        .expect("Failed to fit scaler");

    // Создаём Dataset из одного значения.
    let single = array![[value]]; // 1×1
    let transformed = scaler.transform(single);
    // Извлекаем z-оценку.
    transformed[[0, 0]]
}

fn main() {
    // Инициализация данных (макрос `array!` из ndarray).
    let puzzle_scores = array![
        640., 720., 580., 810., 690., 750., 600., 700., 660., 730., 620., 780.
    ];
    let racing_scores = array![
        12400., 15800., 13900., 11200., 16500., 14100.,
        12800., 15100., 13300., 14700., 12100., 14600.
    ];

    let puzzle = Puzzle { scores: puzzle_scores };
    let racing = Racing { scores: racing_scores };

    // Чтение результатов новичка.
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let p: f64 = input.trim().parse().unwrap();

    input.clear();
    io::stdin().read_line(&mut input).unwrap();
    let r: f64 = input.trim().parse().unwrap();

    // Параллельное вычисление z-оценок с помощью `rayon::join`.
    let (z_puzzle, z_racing) = rayon::join(
        || z_score_linfa(puzzle.scores(), p),
        || z_score_linfa(racing.scores(), r),
    );

    // Округление до 2 знаков после запятой.
    let z_puzzle_rounded = (z_puzzle * 100.0).round() / 100.0;
    let z_racing_rounded = (z_racing * 100.0).round() / 100.0;

    println!("Puzzle z: {:.2}", z_puzzle_rounded);
    println!("Racing z: {:.2}", z_racing_rounded);

    // Сравнение впечатляющести результатов.
    if z_puzzle > z_racing {
        println!("More impressive: Puzzle");
    } else {
        println!("More impressive: Racing");
    }
}
