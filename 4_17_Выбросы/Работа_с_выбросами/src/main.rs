//! Анализ выбросов в скоростях велопоездок через IQR‑границы.
//!
//! SOLID / ООП:
//!   * SRP  — каждая ответственность в своём типе (`IqrFence`, `DropPolicy`,
//!            `CapPolicy`, `Summary`, `Report`).
//!   * OCP  — новые стратегии = новый `impl OutlierPolicy`, без правок `Analyzer`.
//!   * LSP  — `DropPolicy` и `CapPolicy` взаимозаменяемы за трейтом.
//!   * ISP  — узкие трейты `Fence`, `Stats`, `OutlierPolicy`.
//!   * DIP  — `Analyzer` зависит от абстракций (`dyn Fence` / `dyn OutlierPolicy`).
//!
//! Макросы: `array!` (конструктор), `#[derive(Debug)]`.
//! Многопоточность: rayon во всех per-ride операциях.
//! ML/числовой стек: ndarray + ndarray-stats (низкоуровневый слой linfa).

use ndarray::array;
use ndarray::Array1;
use ndarray_stats::interpolate::Linear;
use ndarray_stats::Quantile1dExt;
use noisy_float::types::{n64, N64};
use rayon::prelude::*;
use std::fmt;

// ---------------- SRP: вычисление границ ----------------
pub trait Fence {
    fn bounds(&self, speeds: &[f64]) -> (f64, f64);
}

pub struct IqrFence {
    pub k: f64,
}

impl Default for IqrFence {
    fn default() -> Self {
        Self { k: 1.5 }
    }
}

impl Fence for IqrFence {
    fn bounds(&self, speeds: &[f64]) -> (f64, f64) {
        // ndarray-stats 0.5: `quantiles_mut` требует `A: Ord + Clone`.
        // `f64` не реализует `Ord` (из-за NaN), а `N64` из noisy_float — да,
        // поэтому приводим массив к `Array1<N64>`.
        let mut arr: Array1<N64> = Array1::from_iter(speeds.iter().map(|&v| n64(v)));

        let qs = array![n64(0.25), n64(0.75)];
        let q = arr
            .quantiles_mut(&qs, &Linear)
            .expect("non-empty input");

        // q[0], q[1] — это N64; извлекаем примитивный f64 через `.raw()`.
        let (q1, q3) = (q[0].raw(), q[1].raw());
        let iqr = q3 - q1;
        (q1 - self.k * iqr, q3 + self.k * iqr)
    }
}

// ---------------- SRP: статистика (population mean/std, как numpy) ----------------
pub trait Stats {
    fn mean_std(&self) -> (f64, f64);
}

impl Stats for [f64] {
    fn mean_std(&self) -> (f64, f64) {
        let n = self.len() as f64;
        // Однопроходный параллельный reduce: (Σx, Σx²).
        let (s, s2) = self
            .par_iter()
            .map(|&v| (v, v * v))
            .reduce(|| (0.0_f64, 0.0_f64), |a, b| (a.0 + b.0, a.1 + b.1));
        let mean = s / n;
        // Population variance (ddof = 0), как np.std.
        let var = (s2 / n - mean * mean).max(0.0);
        (mean, var.sqrt())
    }
}

// ---------------- OCP / LSP: стратегии обработки выбросов ----------------
pub trait OutlierPolicy {
    fn apply(&self, speeds: &[f64], low: f64, high: f64) -> Vec<f64>;
    fn changed(&self, before: &[f64], after: &[f64]) -> usize;
}

pub struct DropPolicy;

impl OutlierPolicy for DropPolicy {
    fn apply(&self, speeds: &[f64], low: f64, high: f64) -> Vec<f64> {
        speeds
            .par_iter()
            .copied()
            .filter(|&v| v >= low && v <= high)
            .collect()
    }
    fn changed(&self, _before: &[f64], _after: &[f64]) -> usize {
        0
    }
}

pub struct CapPolicy;

impl OutlierPolicy for CapPolicy {
    fn apply(&self, speeds: &[f64], low: f64, high: f64) -> Vec<f64> {
        speeds.par_iter().map(|&v| v.clamp(low, high)).collect()
    }
    fn changed(&self, before: &[f64], after: &[f64]) -> usize {
        before
            .par_iter()
            .zip(after.par_iter())
            .filter(|(a, b)| **a != **b)
            .count()
    }
}

// ---------------- SRP: строка отчёта ----------------
#[derive(Debug)]
struct Summary {
    label: &'static str,
    rides: usize,
    mean: f64,
    std: f64,
}

impl Summary {
    fn from(label: &'static str, speeds: &[f64]) -> Self {
        let (mean, std) = speeds.mean_std();
        Self { label, rides: speeds.len(), mean, std }
    }
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} rides, mean {:.2}, std {:.2}",
            self.label, self.rides, self.mean, self.std
        )
    }
}

// ---------------- DIP: анализатор работает через абстракции ----------------
struct Analyzer<'a> {
    fence: &'a dyn Fence,
    drop_policy: &'a dyn OutlierPolicy,
    cap_policy: &'a dyn OutlierPolicy,
}

impl<'a> Analyzer<'a> {
    fn run(&self, speeds: &[f64]) -> Report {
        let (low, high) = self.fence.bounds(speeds);
        let dropped = self.drop_policy.apply(speeds, low, high);
        let capped = self.cap_policy.apply(speeds, low, high);
        let changed = self.cap_policy.changed(speeds, &capped);
        Report {
            keep: Summary::from("Keep", speeds),
            drop: Summary::from("Drop", &dropped),
            cap: Summary::from("Cap", &capped),
            changed,
            total: speeds.len(),
        }
    }
}

struct Report {
    keep: Summary,
    drop: Summary,
    cap: Summary,
    changed: usize,
    total: usize,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.keep)?;
        writeln!(f, "{}", self.drop)?;
        writeln!(f, "{}", self.cap)?;
        write!(f, "Capping changed {} of {} rides", self.changed, self.total)
    }
}

fn main() {
    // Исходные скорости — 15 поездок (две из них аномально низкая/высокая).
    let speeds: Vec<f64> = array![
        24.5, 22.8, 26.1, 23.9, 3.1, 25.4, 21.7, 24.2, 27.3, 78.4,
        23.1, 25.8, 22.4, 24.9, 26.6
    ]
    .to_vec();

    // Внедрение зависимостей (DIP): конкретные реализации подставляются снаружи.
    let fence = IqrFence::default();
    let drop_policy = DropPolicy;
    let cap_policy = CapPolicy;

    let analyzer = Analyzer {
        fence: &fence,
        drop_policy: &drop_policy,
        cap_policy: &cap_policy,
    };

    print!("{}", analyzer.run(&speeds));
}
