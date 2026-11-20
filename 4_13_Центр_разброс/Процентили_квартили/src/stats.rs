//! Модуль `stats` — расчёт квартилей и IQR (SRP, OCP).
//!
//! Trait [`QuartileEstimator`] абстрагирует алгоритм квартилей: новые стратегии
//! (linear, lower, nearest, midhinge) добавляются реализацией trait-а без
//! правок движка (OCP). Базовая реализация [`LinearQuartileEstimator`]
//! соответствует методу линейной интерполяции `numpy.percentile` (default).

use crate::data::DataSource;
use ndarray::Array1;
use statrs::statistics::{Data, Distribution};

/// Результат расчёта квартилей и IQR.
#[derive(Debug, Clone, Copy)]
pub struct QuartileSummary {
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub iqr: f64,
    /// Среднее значение выборки (ML: базовая описательная статистика).
    pub mean: f64,
    /// Стандартное отклонение выборки (ML: разброс).
    pub std: f64,
}

/// Алгоритм оценки квартилей.
pub trait QuartileEstimator {
    fn estimate(&self, data: &dyn DataSource) -> QuartileSummary;
}

/// Линейная интерполяция (по умолчанию, как `numpy.percentile`).
pub struct LinearQuartileEstimator;

impl LinearQuartileEstimator {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LinearQuartileEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl QuartileEstimator for LinearQuartileEstimator {
    fn estimate(&self, data: &dyn DataSource) -> QuartileSummary {
        // Реализация метода 'linear' из numpy.percentile: позиция i=(n-1)*q,
        // интерполяция между floor(i) и ceil(i). Это даёт точное совпадение с
        // исходным Python-кодом np.percentile(use, [25, 50, 75]).
        let mut arr: Vec<f64> = data.usage().to_vec();
        arr.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        let q1 = numpy_percentile(&arr, 0.25);
        let median = numpy_percentile(&arr, 0.50);
        let q3 = numpy_percentile(&arr, 0.75);
        let iqr = q3 - q1;
        // ML-крейт statrs: описательная статистика (mean, std) для оценки
        // разброса и потенциальных выбросов расхода воды.
        let d = Data::new(arr.clone());
        let mean = d.mean().unwrap_or(f64::NAN);
        let std = d.std_dev().unwrap_or(f64::NAN);
        QuartileSummary { q1, median, q3, iqr, mean, std }
    }
}

/// Утилита: получить отсортированную копию выборки (используется в тестах/CLI).
#[allow(dead_code)]
pub fn sorted(data: &dyn DataSource) -> Array1<f64> {
    let mut v = data.usage().to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    Array1::from_vec(v)
}

/// Алгоритм 'linear' из `numpy.percentile`:
/// позиция `i = (n - 1) * q`, значение = линейная интерполяция между
/// элементами с индексами `floor(i)` и `ceil(i)`.
fn numpy_percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    assert!(n > 0, "sample must not be empty");
    let pos = (n as f64 - 1.0) * q;
    let lo = pos.floor();
    let hi = pos.ceil();
    if lo == hi {
        sorted[lo as usize]
    } else {
        let frac = pos - lo;
        let a = sorted[lo as usize];
        let b = sorted[hi as usize];
        a + (b - a) * frac
    }
}
