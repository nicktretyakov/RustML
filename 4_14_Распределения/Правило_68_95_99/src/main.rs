use linfa::dataset::Dataset;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::{Distribution as RandDistribution, Normal as RandNormal};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal as StatrsNormal};
use std::io;

// ────────────────────────────────────────────────────────────
// 1. Макрос для единообразного вывода процентов
// ────────────────────────────────────────────────────────────
macro_rules! print_percent {
    ($label:expr, $value:expr) => {
        println!("{}: {:.1}%", $label, $value * 100.0);
    };
}

// ────────────────────────────────────────────────────────────
// 2. Трейты (SOLID: SRP, DIP)
// ────────────────────────────────────────────────────────────
trait Statistics {
    fn mean(&self) -> f64;
    fn std_dev(&self) -> f64;
    fn count_below(&self, threshold: f64) -> usize;
    fn proportion_within_std(&self, k: f64) -> f64;
}

trait Distribution {
    fn cdf(&self, x: f64) -> f64;
}

trait Predictor {
    fn predict_cdf(&self, x: f64) -> f64;
}

// ────────────────────────────────────────────────────────────
// 3. Реализация Statistics для данных о лампах
// ────────────────────────────────────────────────────────────
struct LifeData {
    values: Vec<f64>,
}

impl LifeData {
    fn new(values: Vec<f64>) -> Self {
        Self { values }
    }
}

impl Statistics for LifeData {
    fn mean(&self) -> f64 {
        self.values.par_iter().sum::<f64>() / self.values.len() as f64
    }

    fn std_dev(&self) -> f64 {
        let mean = self.mean();
        let variance = self
            .values
            .par_iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / self.values.len() as f64;
        variance.sqrt()
    }

    fn count_below(&self, threshold: f64) -> usize {
        self.values.par_iter().filter(|&&x| x < threshold).count()
    }

    fn proportion_within_std(&self, k: f64) -> f64 {
        let mean = self.mean();
        let std = self.std_dev();
        let lower = mean - k * std;
        let upper = mean + k * std;
        let count = self
            .values
            .par_iter()
            .filter(|&&x| x > lower && x < upper)
            .count();
        count as f64 / self.values.len() as f64
    }
}

// ────────────────────────────────────────────────────────────
// 4. Нормальное распределение и ML-обёртка
// ────────────────────────────────────────────────────────────
struct NormalDistribution {
    mean: f64,
    std_dev: f64,
}

impl Distribution for NormalDistribution {
    fn cdf(&self, x: f64) -> f64 {
        let normal = StatrsNormal::new(self.mean, self.std_dev).unwrap();
        normal.cdf(x)
    }
}

impl Predictor for NormalDistribution {
    fn predict_cdf(&self, x: f64) -> f64 {
        self.cdf(x)
    }
}

// ────────────────────────────────────────────────────────────
// 5. Основная программа
// ────────────────────────────────────────────────────────────
fn main() {
    // 5.1 Генерация 5000 значений
    let mut rng = thread_rng();
    let normal = RandNormal::new(1200.0, 100.0).unwrap();
    let life: Vec<f64> = (0..5000).map(|_| normal.sample(&mut rng)).collect();

    // 5.2 Оборачиваем в Linfa Dataset (правильные типы!)
    let n = life.len();
    let records: Array2<f64> =
        Array2::from_shape_vec((n, 1), life.clone()).expect("shape mismatch");
    let targets: Array1<f64> = Array1::zeros(n);
    let _dataset = Dataset::new(records, targets);

    let life_data = LifeData::new(life);

    // 5.3 Статистики
    let mean = life_data.mean();
    let std_dev = life_data.std_dev();
    let within1 = life_data.proportion_within_std(1.0);
    let within2 = life_data.proportion_within_std(2.0);

    // 5.4 Читаем h
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let h: f64 = input.trim().parse().unwrap();

    // 5.5 Доли ниже h
    let below_counted =
        life_data.count_below(h) as f64 / life_data.values.len() as f64;

    // 5.6 Нормальная кривая (ML-модель)
    let normal_dist = NormalDistribution { mean, std_dev };
    let below_normal = normal_dist.predict_cdf(h);

    // 5.7 Вывод
    print_percent!("Within 1 std", within1);
    print_percent!("Within 2 std", within2);
    println!("Below {} hours, counted: {:.1}%", h, below_counted * 100.0);
    println!("Below {} hours, normal curve: {:.1}%", h, below_normal * 100.0);
}
