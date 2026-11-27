// src/main.rs
use ndarray::Array1;               // ML-экосистема (linfa/smartcore)
use rayon::prelude::*;             // многопоточность
use std::io::{self, BufRead};

// ============================================================
// SOLID: I (Interface Segregation) + D (Dependency Inversion)
// Узкие трейты-абстракции, от которых зависит main.
// ============================================================
pub trait FenceCalculator: Send + Sync {
    fn fences(&self, data: &Array1<f64>) -> (f64, f64);
}

pub trait OutlierDetector: Send + Sync {
    fn unusual(&self, data: &Array1<f64>, low: f64, high: f64) -> Vec<f64>;
}

// ============================================================
// SOLID: S (Single Responsibility) — только считает границы IQR
// SOLID: O (Open/Closed) — новые реализации добавляются без правки main
// ============================================================
#[derive(Default, Clone, Copy, Debug)]
pub struct IqrFenceCalculator;

impl IqrFenceCalculator {
    /// Перцентиль с линейной интерполяцией (совместимо с numpy.percentile)
    fn percentile(sorted: &[f64], p: f64) -> f64 {
        match sorted.len() {
            0 => f64::NAN,
            1 => sorted[0],
            n => {
                let rank = p / 100.0 * (n - 1) as f64;
                let lo = rank.floor() as usize;
                let hi = rank.ceil() as usize;
                let w = rank - lo as f64;
                sorted[lo] * (1.0 - w) + sorted[hi] * w
            }
        }
    }
}

impl FenceCalculator for IqrFenceCalculator {
    fn fences(&self, data: &Array1<f64>) -> (f64, f64) {
        // Многопоточная сортировка (rayon)
        let mut sorted: Vec<f64> = data.to_vec();
        sorted.par_sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q1 = Self::percentile(&sorted, 25.0);
        let q3 = Self::percentile(&sorted, 75.0);
        let iqr = q3 - q1;
        (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    }
}

// ============================================================
// SOLID: S — только отбирает выбросы, сохраняя исходный порядок
// ============================================================
#[derive(Default, Clone, Copy, Debug)]
pub struct ParallelOutlierDetector;

impl OutlierDetector for ParallelOutlierDetector {
    fn unusual(&self, data: &Array1<f64>, low: f64, high: f64) -> Vec<f64> {
        let values: Vec<f64> = data.to_vec();
        // Параллельный фильтр по индексам
        let mut idxs: Vec<usize> = (0..values.len())
            .into_par_iter()
            .filter(|&i| values[i] < low || values[i] > high)
            .collect();
        idxs.sort_unstable();                      // сохранить исходный порядок
        idxs.into_iter().map(|i| values[i]).collect()
    }
}

// ============================================================
// Макрос: чтение n чисел (по одному в строке)
// ============================================================
macro_rules! read_values {
    ($n:expr) => {{
        let stdin = io::stdin();
        let mut handle = stdin.lock();
        let mut values: Vec<f64> = Vec::with_capacity($n);
        for _ in 0..$n {
            let mut line = String::new();
            handle.read_line(&mut line).expect("read_line failed");
            values.push(line.trim().parse::<f64>().expect("parse f64"));
        }
        Array1::from(values)
    }};
}

// Макросы форматирования (эмуляция numpy round / печати int-массива)
macro_rules! fence_str {
    ($v:expr) => {{
        let x: f64 = $v;
        if x.fract() == 0.0 { format!("{:.1}", x) } else { format!("{}", x) }
    }};
}

macro_rules! value_str {
    ($v:expr) => {{
        let x: f64 = $v;
        if x.fract() == 0.0 { format!("{}", x as i64) } else { format!("{}", x) }
    }};
}

// ============================================================
// SOLID: D — main зависит только от трейтов, не от конкретики
// ============================================================
fn main() {
    let mut buf = String::new();
    io::stdin().read_line(&mut buf).expect("read n");
    let n: usize = buf.trim().parse().expect("parse n");

    let minutes = read_values!(n);

    // Dependency inversion: конкретика скрыта за dyn-трейтом
    let calc: Box<dyn FenceCalculator> = Box::new(IqrFenceCalculator);
    let (low, high) = calc.fences(&minutes);

    // Не изменяем поведение исходника
    println!(
        "Fences: {} to {} minutes",
        fence_str!(low),
        fence_str!(high)
    );

    let detector: Box<dyn OutlierDetector> = Box::new(ParallelOutlierDetector);
    let unusual = detector.unusual(&minutes, low, high);
    let rendered: Vec<String> = unusual.iter().map(|&v| value_str!(v)).collect();
    println!("Unusual: [{}]", rendered.join(", "));
}
