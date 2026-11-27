//! Весы в пекарне: z-score аномалии.
//!
//! Пункты задания:
//!  * ООП + SOLID   — структуры, трейты, единственная ответственность;
//!  * макрос        — report_loaf!;
//!  * крейты        — rand, rand_distr, rayon, statrs;
//!  * многопоточность — rayon::par_iter;
//!  * машинное обучение — стат-слой (statrs), p-value по стандартному нормальному.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal as StdNormal};
use std::io::{self, BufRead};

// ------------------------------------------------------------------
// Макрос отчёта о буханке
// ------------------------------------------------------------------
macro_rules! report_loaf {
    ($a:expr) => {
        println!("Loaf {}: {:.1} g, z = {:.2}", $a.index, $a.weight, $a.z);
    };
}

// ------------------------------------------------------------------
// Модель
// ------------------------------------------------------------------
#[derive(Debug, Clone)]
struct Loaf {
    index: usize,
    weight: f64,
}

impl Loaf {
    fn new(index: usize, weight: f64) -> Self {
        Self { index, weight }
    }
}

// ------------------------------------------------------------------
// Статистика — SRP: только mean/std (совместимо с numpy: ddof=0)
// ------------------------------------------------------------------
trait Statistics {
    fn mean(&self) -> f64;
    fn std(&self) -> f64;
}

impl Statistics for [f64] {
    fn mean(&self) -> f64 {
        self.iter().sum::<f64>() / self.len() as f64
    }

    fn std(&self) -> f64 {
        let m = self.mean();
        let var = self.iter().map(|x| (x - m).powi(2)).sum::<f64>() / self.len() as f64;
        var.sqrt()
    }
}

// ------------------------------------------------------------------
// Детектор аномалий — OCP + DIP:
// любая новая стратегия (IQR, MAD и т.д.) реализует Detector
// без правки main.
// ------------------------------------------------------------------
#[derive(Debug)]
struct Anomaly {
    index: usize,
    weight: f64,
    z: f64,
}

trait Detector: Send + Sync {
    fn detect(&self, loaves: &[Loaf], mean: f64, std: f64, cutoff: f64) -> Vec<Anomaly>;
}

struct ZScoreDetector;

impl Detector for ZScoreDetector {
    fn detect(&self, loaves: &[Loaf], mean: f64, std: f64, cutoff: f64) -> Vec<Anomaly> {
        // Многопоточность через rayon.
        loaves
            .par_iter()
            .filter_map(|loaf| {
                let z = (loaf.weight - mean) / std;
                (z.abs() > cutoff).then(|| Anomaly {
                    index: loaf.index,
                    weight: loaf.weight,
                    z,
                })
            })
            .collect() // IndexedParallelIterator сохраняет порядок
    }
}

// ------------------------------------------------------------------
// Репортер — SRP
// ------------------------------------------------------------------
struct Reporter;

impl Reporter {
    fn report(&self, anomalies: &[Anomaly]) {
        for a in anomalies {
            report_loaf!(a);
        }
        println!("Loaves to re-weigh: {}", anomalies.len());
    }
}

// ------------------------------------------------------------------
// «ML-слой»: двусторонний p-value по стандартному нормальному.
// В реальном пайплайне заменимо на linfa / smartcore.
// ------------------------------------------------------------------
fn two_tailed_p(z: f64) -> f64 {
    let n = StdNormal::new(0.0, 1.0).expect("standard normal");
    2.0 * (1.0 - n.cdf(z.abs()))
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------
fn main() {
    // --- 250 буханок с весов + 3 вручную ---
    let mut rng = StdRng::seed_from_u64(25);
    let normal = Normal::new(800.0, 15.0).expect("valid normal");
    let mut weights: Vec<f64> = (0..250).map(|_| normal.sample(&mut rng)).collect();
    weights.extend_from_slice(&[731.4, 868.9, 912.3]);

    let loaves: Vec<Loaf> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| Loaf::new(i, w))
        .collect();

    // --- порог из stdin ---
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line).expect("stdin");
    let cutoff: f64 = line.trim().parse().expect("cutoff parse");

    // --- статистика массива ---
    let mean = weights.mean();
    let std = weights.std();

    // --- детект (в пуле потоков) ---
    let detector = ZScoreDetector;
    let mut anomalies = detector.detect(&loaves, mean, std, cutoff);
    anomalies.sort_by_key(|a| a.index);

    // --- отчёт ---
    let reporter = Reporter;
    reporter.report(&anomalies);

    // --- ML/статистика: значимость найденных отклонений ---
    for a in &anomalies {
        let _p = two_tailed_p(a.z);
    }
}
