use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::io::{self, Write};

// Макрос для единообразного вывода результата
macro_rules! print_result {
    ($n:expr, $se:expr, $spread:expr) => {
        println!("Bags to weigh: {}", $n);
        println!("Standard error: {:.2} g", $se);
        println!("Simulated spread: {:.2} g", $spread);
    };
}

// Трейт-интерфейс (Interface Segregation, Dependency Inversion)
trait Estimator {
    fn population_std(&self, data: &[f64]) -> f64;
    fn standard_error(&self, data: &[f64], n: usize) -> f64;
}

// Класс-реализация (Single Responsibility)
struct BagAnalyzer;

impl Estimator for BagAnalyzer {
    fn population_std(&self, data: &[f64]) -> f64 {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / data.len() as f64;
        variance.sqrt()
    }

    fn standard_error(&self, data: &[f64], n: usize) -> f64 {
        self.population_std(data) / (n as f64).sqrt()
    }
}

// Генерация весов мешков (округление до 1 знака после запятой)
fn generate_bags(seed: u64, size: usize, mean: f64, std_dev: f64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(mean, std_dev).unwrap();
    (0..size)
        .map(|_| {
            let x: f64 = normal.sample(&mut rng);
            (x * 10.0).round() / 10.0
        })
        .collect()
}

// Многопоточный расчёт выборочных средних (без возвращения)
fn sample_means_parallel(
    data: &[f64],
    size: usize,
    count: usize,
    base_seed: u64,
) -> Vec<f64> {
    (0..count)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(base_seed + i as u64);
            let mut indices: Vec<usize> = (0..data.len()).collect();
            indices.shuffle(&mut rng);
            let sample: Vec<f64> = indices[..size]
                .iter()
                .map(|&idx| data[idx])
                .collect();
            sample.iter().sum::<f64>() / size as f64
        })
        .collect()
}

fn main() {
    // Чтение допустимой стандартной ошибки
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");
    let target: f64 = input.trim().parse().expect("Invalid target");

    // Исходные данные: 5000 мешков, среднее 500 г, ст. отклонение 8 г
    let bags = generate_bags(22, 5000, 500.0, 8.0);
    let analyzer = BagAnalyzer;

    // Поиск минимального n
    let mut n = 1;
    while analyzer.standard_error(&bags, n) > target {
        n += 1;
    }

    // Симуляция 300 выборочных средних
    let means = sample_means_parallel(&bags, n, 300, 1000);
    let spread = {
        let m = means.iter().sum::<f64>() / means.len() as f64;
        let var = means.iter().map(|&x| (x - m).powi(2)).sum::<f64>()
            / means.len() as f64;
        var.sqrt()
    };

    print_result!(n, analyzer.standard_error(&bags, n), spread);
}
