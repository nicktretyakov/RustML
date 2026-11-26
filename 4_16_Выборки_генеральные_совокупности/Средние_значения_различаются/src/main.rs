// src/main.rs
use rand::prelude::*;
use rand::seq::index;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

// ============================================================
//  Трейты (SOLID: SRP, OCP, LSP, ISP, DIP)
// ============================================================

/// Интерфейс отбора выборок — позволяет подменить алгоритм,
/// не меняя вызывающий код (Dependency Inversion).
trait Sampler: Send + Sync {
    fn sample_means(&self, data: &[f64], size: usize, count: usize) -> Vec<f64>;
}

/// Интерфейс оценки параметров распределения (ML-компонент).
trait Estimator: Send + Sync {
    /// Возвращает (среднее, стандартное отклонение) по выборке.
    fn fit(&self, data: &[f64]) -> (f64, f64);
}

// ============================================================
//  Реализации
// ============================================================

/// Отбор без возвращения + параллельное вычисление средних.
struct RandomSampler;

impl Sampler for RandomSampler {
    fn sample_means(&self, data: &[f64], size: usize, count: usize) -> Vec<f64> {
        assert!(
            size <= data.len(),
            "размер выборки больше генеральной совокупности"
        );

        (0..count)
            .into_par_iter() // многопоточность через rayon
            .map(|i| {
                // Свой детерминированный ГПСЧ на каждую выборку
                let mut rng = StdRng::seed_from_u64(20 + i as u64);
                let idx = index::sample(&mut rng, data.len(), size);
                let sum: f64 = idx.iter().map(|j| data[j]).sum();
                sum / size as f64
            })
            .collect()
    }
}

/// Оценка максимального правдоподобия для нормального распределения.
/// Это простейший ML-метод (density estimation).
struct NormalMleEstimator;

impl Estimator for NormalMleEstimator {
    fn fit(&self, data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        (mean, var.sqrt())
    }
}

// ============================================================
//  Пайплайн анализа (DIP: зависит от абстракций)
// ============================================================

struct Report {
    size: usize,
    count: usize,
    min: f64,
    max: f64,
    std: f64,
}

struct AnalysisPipeline {
    sampler: Box<dyn Sampler>,
    estimator: Box<dyn Estimator>,
}

impl AnalysisPipeline {
    fn new(sampler: Box<dyn Sampler>, estimator: Box<dyn Estimator>) -> Self {
        Self { sampler, estimator }
    }

    fn run(&self, data: &[f64], size: usize, count: usize) -> Report {
        let means = self.sampler.sample_means(data, size, count);
        let (_, std) = self.estimator.fit(&means);

        let min = means.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Report {
            size,
            count,
            min,
            max,
            std,
        }
    }
}

// ============================================================
//  Макросы
// ============================================================

/// Генерация массива высот бамбука (аналог rng.normal(95, 15, size=1800)).
/// Turbofish `Normal::<f64>` фиксирует тип литералов $mu/$sigma (лечит E0689).
macro_rules! generate_heights {
    ($n:expr, $mu:expr, $sigma:expr, $seed:expr) => {{
        let mut rng = StdRng::seed_from_u64($seed);
        let normal = Normal::<f64>::new($mu, $sigma).unwrap();
        (0..$n)
            .map(|_| {
                let v: f64 = normal.sample(&mut rng);
                (v * 10.0).round() / 10.0
            })
            .collect::<Vec<f64>>()
    }};
}

/// Печать отчёта в формате, совместимом с исходным Python-скриптом.
macro_rules! print_report {
    ($r:expr) => {
        println!(
            "{} means of {} plants: {:.2} to {:.2} cm",
            $r.count, $r.size, $r.min, $r.max
        );
        println!("Spread of the means (std): {:.2} cm", $r.std);
    };
}

// ============================================================
//  main
// ============================================================

fn main() {
    // Размер выборки: аргумент командной строки или 10 по умолчанию
    let size: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // Генеральная совокупность: 1800 растений ~ N(95, 15)
    let heights = generate_heights!(1800, 95.0, 15.0, 20);

    // Сборка пайплайна: сэмплер + ML-оценщик
    let pipeline = AnalysisPipeline::new(Box::new(RandomSampler), Box::new(NormalMleEstimator));

    // 200 выборок, отчёт
    let report = pipeline.run(&heights, size, 200);
    print_report!(report);
}
