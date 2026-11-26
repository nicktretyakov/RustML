use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::sync::Mutex;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// CLI — макрос-крейт clap (derive)
// ─────────────────────────────────────────────────────────────────────────────
#[derive(Parser, Debug)]
#[command(name = "sheep_farm", about = "Статистическая симуляция веса руна")]
struct Args {
    /// Размер выборки
    #[arg(short, long, default_value_t = 30)]
    size: usize,
    /// Всего овец в хозяйстве
    #[arg(long, default_value_t = 3000)]
    sheep: usize,
    /// Сид ГПСЧ
    #[arg(long, default_value_t = 19)]
    seed: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Доменная модель
// ─────────────────────────────────────────────────────────────────────────────
#[derive(Debug, Clone, Copy)]
pub struct Fleece {
    #[allow(dead_code)]
    sheep_id: usize,
    weight_g: f64,
}

#[derive(Error, Debug)]
pub enum FarmError {
    #[error("размер выборки {0} больше популяции {1}")]
    SampleTooBig(usize, usize),
    #[error("нормальное распределение: {0}")]
    Normal(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// SOLID · I(SP): узкий интерфейс источника данных
// ─────────────────────────────────────────────────────────────────────────────
pub trait FleeceSource: Send + Sync {
    fn len(&self) -> usize;
    fn weight(&self, i: usize) -> f64;
    fn mean_all(&self) -> f64;
}

// SRP: отвечает только за генерацию данных
pub struct SimulatedFarm {
    fleeces: Vec<Fleece>,
}

impl SimulatedFarm {
    pub fn generate(n: usize, mean: f64, std: f64, seed: u64) -> Result<Self, FarmError> {
        const CHUNK: usize = 512;
        let n_chunks = n.div_ceil(CHUNK);

        // Многопоточная генерация: каждый чанк получает свой детерминированный RNG.
        let fleeces: Vec<Fleece> = (0..n_chunks)
            .into_par_iter()
            .flat_map_iter(|ci| {
                let mut rng = ChaCha8Rng::seed_from_u64(
                    seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(ci as u64),
                );
                let dist = Normal::new(mean, std).expect("валидное нормальное");
                let start = ci * CHUNK;
                let end = (start + CHUNK).min(n);
                (start..end).map(move |i| Fleece {
                    sheep_id: i,
                    // np.round(rng.normal(...))  — округление до целого
                    weight_g: np_round(dist.sample(&mut rng)),
                })
            })
            .collect();

        Ok(Self { fleeces })
    }
}

impl FleeceSource for SimulatedFarm {
    fn len(&self) -> usize {
        self.fleeces.len()
    }
    fn weight(&self, i: usize) -> f64 {
        self.fleeces[i].weight_g
    }
    fn mean_all(&self) -> f64 {
        // Многопоточный подсчёт среднего
        self.fleeces.par_iter().map(|f| f.weight_g).sum::<f64>()
            / self.fleeces.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SOLID · O(CP): новые стратегии сэмплирования — новый impl, код Analysis не трогаем
// ─────────────────────────────────────────────────────────────────────────────
pub trait Sampler: Send + Sync {
    fn sample_indices(&self, population: usize, size: usize) -> Result<Vec<usize>, FarmError>;
}

pub struct RandomSampler {
    rng: Mutex<ChaCha8Rng>,
}

impl RandomSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Mutex::new(ChaCha8Rng::seed_from_u64(seed ^ 0xDEAD_BEEF_CAFE_F00D)),
        }
    }
}

impl Sampler for RandomSampler {
    fn sample_indices(&self, population: usize, size: usize) -> Result<Vec<usize>, FarmError> {
        if size > population {
            return Err(FarmError::SampleTooBig(size, population));
        }
        let mut rng = self.rng.lock().unwrap();
        // Частичный Фишер–Йетс (как np.random.Generator.choice без replacement)
        let mut idx: Vec<usize> = (0..population).collect();
        for i in 0..size {
            let j = rng.gen_range(i..population);
            idx.swap(i, j);
        }
        idx.truncate(size);
        Ok(idx)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ML · оценка среднего + доверительный интервал
// ─────────────────────────────────────────────────────────────────────────────
pub trait Estimator: Send + Sync {
    fn estimate(&self, sample: &[f64]) -> f64;
}

pub struct MeanEstimator;

impl Estimator for MeanEstimator {
    fn estimate(&self, sample: &[f64]) -> f64 {
        sample.par_iter().sum::<f64>() / sample.len() as f64
    }
}

/// «ML»-часть: статистический вывод — оценка матожидания и 95 % ДИ.
pub struct ConfidenceInterval {
    level: f64,
}

impl ConfidenceInterval {
    pub fn new(level: f64) -> Self {
        Self { level }
    }
    pub fn interval(&self, sample: &[f64]) -> (f64, f64) {
        let n = sample.len() as f64;
        let mean = sample.par_iter().sum::<f64>() / n;
        let var = sample
            .par_iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let se = (var / n).sqrt();
        let t = StudentsT::new(0.0, 1.0, n - 1.0).expect("валидное t-распределение");
        let q = t.inverse_cdf(1.0 - (1.0 - self.level) / 2.0);
        (mean - q * se, mean + q * se)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SOLID · D(IP): Analysis зависит от абстракций, не от конкретики
// ─────────────────────────────────────────────────────────────────────────────
pub struct AnalysisResult {
    pub size: usize,
    pub total: usize,
    pub sample_mean: f64,
    pub pop_mean: f64,
    pub diff: f64,
    pub ci: Option<(f64, f64)>,
}

pub struct Analysis<'a, S: Sampler, E: Estimator> {
    source: &'a dyn FleeceSource,
    sampler: S,
    estimator: E,
    ci: Option<ConfidenceInterval>,
}

impl<'a, S: Sampler, E: Estimator> Analysis<'a, S, E> {
    pub fn new(source: &'a dyn FleeceSource, sampler: S, estimator: E) -> Self {
        Self { source, sampler, estimator, ci: Some(ConfidenceInterval::new(0.95)) }
    }

    pub fn run(&self, size: usize) -> Result<AnalysisResult, FarmError> {
        let idx = self.sampler.sample_indices(self.source.len(), size)?;
        let weights: Vec<f64> = idx.iter().map(|&i| self.source.weight(i)).collect();
        let sample_mean = self.estimator.estimate(&weights);
        let pop_mean = self.source.mean_all();
        Ok(AnalysisResult {
            size,
            total: self.source.len(),
            sample_mean,
            pop_mean,
            diff: sample_mean - pop_mean,
            ci: self.ci.as_ref().map(|c| c.interval(&weights)),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Пользовательский макрос отчёта
// ─────────────────────────────────────────────────────────────────────────────
macro_rules! report {
    ($r:expr) => {{
        let r = $r;
        println!("Sample of {} sheep: {:.2} g", r.size, np_round(r.sample_mean * 100.0) / 100.0);
        println!("All {} sheep: {:.2} g", r.total, np_round(r.pop_mean * 100.0) / 100.0);
        println!("Off by: {:.2} g", np_round(r.diff * 100.0) / 100.0);
        if let Some((lo, hi)) = r.ci {
            println!("95% CI for population mean: [{:.2}, {:.2}] g",
                     np_round(lo * 100.0) / 100.0, np_round(hi * 100.0) / 100.0);
        }
    }};
}

// ─────────────────────────────────────────────────────────────────────────────
// Банковское округление (как np.round)
// ─────────────────────────────────────────────────────────────────────────────
fn np_round(x: f64) -> f64 {
    let t = x.trunc();
    let frac = x - t;
    if (frac.abs() - 0.5).abs() < 1e-12 {
        if (t as i64) % 2 == 0 { t } else { t + x.signum() }
    } else {
        x.round()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
fn main() -> Result<(), FarmError> {
    let args = Args::parse();

    let farm = SimulatedFarm::generate(args.sheep, 4500.0, 800.0, args.seed)?;

    let analysis = Analysis::new(
        &farm,
        RandomSampler::new(args.seed),
        MeanEstimator,
    );

    let result = analysis.run(args.size)?;
    report!(result);

    Ok(())
}
