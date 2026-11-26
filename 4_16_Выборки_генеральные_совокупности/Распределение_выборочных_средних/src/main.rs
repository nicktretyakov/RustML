use plotters::prelude::*;
use rand::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use rayon::prelude::*;
use std::error::Error;
use std::io::{self, Write};

// ============================================================
// SRP: генерация размеров файлов
// ============================================================
trait FileSizer: Sync {
    fn generate(&self, n: usize) -> Vec<f64>;
}

struct SkewedFileSizer {
    seed: u64,
}

impl FileSizer for SkewedFileSizer {
    fn generate(&self, n: usize) -> Vec<f64> {
        let mut rng = Pcg64::seed_from_u64(self.seed);
        (0..n)
            .map(|_| {
                let u: f64 = rng.gen();
                let v = u.powi(4) * 300.0 + 5.0;
                (v * 100.0).round() / 100.0
            })
            .collect()
    }
}

// ============================================================
// DIP / OCP: стратегия выборочного среднего
// ============================================================
trait Sampler: Sync + Send {
    fn sample_mean(&self, data: &[f64], size: usize, seed: u64) -> f64;
}

struct WithoutReplacementSampler;

impl Sampler for WithoutReplacementSampler {
    fn sample_mean(&self, data: &[f64], size: usize, seed: u64) -> f64 {
        let mut rng = Pcg64::seed_from_u64(seed);
        let mut idx: Vec<usize> = (0..data.len()).collect();
        // Частичный Fisher–Yates: только `size` перестановок
        for i in 0..size {
            let j = rng.gen_range(i..data.len());
            idx.swap(i, j);
        }
        let sum: f64 = idx[..size].iter().map(|&i| data[i]).sum();
        sum / size as f64
    }
}

// ============================================================
// ISP: описательные статистики через макрос
// ============================================================
trait Describe {
    fn mean(&self) -> f64;
    fn median(&self) -> f64;
    fn mean_minus_median(&self) -> f64 {
        self.mean() - self.median()
    }
}

macro_rules! impl_describe_f64_slice {
    () => {
        impl Describe for [f64] {
            fn mean(&self) -> f64 {
                self.iter().sum::<f64>() / self.len() as f64
            }
            fn median(&self) -> f64 {
                let mut v = self.to_vec();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = v.len();
                if n % 2 == 0 {
                    (v[n / 2 - 1] + v[n / 2]) / 2.0
                } else {
                    v[n / 2]
                }
            }
        }
    };
}
impl_describe_f64_slice!();

// ============================================================
// ML: sklearn‑style Estimator (fit / predict)
// ============================================================
trait Estimator: Send + Sync {
    fn fit(&mut self, x: &[f64]);
    fn predict(&self) -> f64;
}

struct SampleMeanEstimator {
    mu_hat: f64,
}

impl SampleMeanEstimator {
    fn new() -> Self {
        Self { mu_hat: 0.0 }
    }
}

impl Estimator for SampleMeanEstimator {
    fn fit(&mut self, x: &[f64]) {
        self.mu_hat = x.mean();
    }
    fn predict(&self) -> f64 {
        self.mu_hat
    }
}

// ============================================================
// Гистограмма (аналог axs[0] / axs[1])
// ============================================================
fn save_histogram(
    path: &str,
    data: &[f64],
    bins: usize,
    title: &str,
    xlabel: &str,
) -> Result<(), Box<dyn Error>> {
    let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let w = (max - min) / bins as f64;

    let mut counts = vec![0u64; bins];
    for &x in data {
        let mut b = ((x - min) / w) as usize;
        if b >= bins {
            b = bins - 1;
        }
        counts[b] += 1;
    }

    let ymax = (*counts.iter().max().unwrap() as f64) * 1.15 + 1.0;
    let xlo = min;
    let xhi = min + w * bins as f64;

    let root = BitMapBackend::new(path, (720, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(12)
        .x_label_area_size(40)
        .y_label_area_size(45)
        .build_cartesian_2d(xlo..xhi, 0f64..ymax)?;

    chart
        .configure_mesh()
        .x_desc(xlabel)
        .y_desc("Count")
        .draw()?;

    chart.draw_series(counts.iter().enumerate().map(|(i, &c)| {
        let x0 = xlo + i as f64 * w;
        Rectangle::new(
            [(x0, 0.0), (x0 + w, c as f64)],
            BLUE.mix(0.7).filled(),
        )
    }))?;

    root.present()?;
    Ok(())
}

// ============================================================
// Точка входа
// ============================================================
fn main() -> Result<(), Box<dyn Error>> {
    // --- чтение размера выборки ---
    print!("Sample size: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let size: usize = input.trim().parse()?;
    if size == 0 {
        return Err("Sample size must be positive".into());
    }

    // --- генерация 6000 файлов (DIP) ---
    let sizer: Box<dyn FileSizer> = Box::new(SkewedFileSizer { seed: 21 });
    let files = sizer.generate(6000);

    // --- 500 выборочных средних параллельно (rayon) ---
    let sampler = WithoutReplacementSampler;
    let means: Vec<f64> = (0..500u64)
        .into_par_iter()
        .map(|i| sampler.sample_mean(&files, size, 21 + i * 7_919))
        .collect();

    // --- ML‑шаг: оценка мат. ожидания по средним ---
    let mut est = SampleMeanEstimator::new();
    est.fit(&means);
    let _mu_hat = est.predict(); // можно сохранить/логировать

    // --- отчёт ---
    println!(
        "Files: mean - median = {:.2} MB",
        files.as_slice().mean_minus_median()
    );
    println!(
        "Sample means: mean - median = {:.2} MB",
        means.as_slice().mean_minus_median()
    );

    // --- диаграммы ---
    save_histogram("axs0.png", &files, 12, "All Files", "File size (MB)")?;
    save_histogram(
        "axs1.png",
        &means,
        12,
        "Sample Means",
        "Mean file size (MB)",
    )?;

    Ok(())
}
