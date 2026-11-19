//! Модуль машинного обучения: линейная регрессия через градиентный спуск.
//!
//! ML-компонент задачи. Здесь простая одномерная линейная регрессия
//! `profit = w * units + b`, обучаемая пакетным градиентным спуском.
//! Это демонстрирует, как Rust может быть использован для ML без внешних
//! ML-фреймворков: математика пишется прямо на коде, а многопоточность
//! (через rayon) ускоряет умножение матриц/векторов.
//!
//! SOLID: класс инкапсулирует только алгоритм обучения; трейт `Regressor`
//! позволяет подставлять другие модели (Dependency Inversion).

use rayon::prelude::*;

/// Трейт модели регрессии (Dependency Inversion).
pub trait Regressor {
    fn fit(&mut self, x: &[f64], y: &[f64]);
    fn predict(&self, x: f64) -> f64;
    fn weights(&self) -> (f64, f64); // (w, b)
}

/// Линейная регрессия, обучаемая градиентным спуском.
pub struct LinearRegression {
    pub w: f64,
    pub b: f64,
    pub lr: f64,
    pub epochs: usize,
}

impl LinearRegression {
    pub fn new(lr: f64, epochs: usize) -> Self {
        Self {
            w: 0.0,
            b: 0.0,
            lr,
            epochs,
        }
    }
}

impl Regressor for LinearRegression {
    /// Пакетный градиентный спуск.
    /// w -= lr * (1/n) * sum((w*x_i + b - y_i) * x_i)
    /// b -= lr * (1/n) * sum(w*x_i + b - y_i)
    fn fit(&mut self, x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len());
        let n = x.len() as f64;
        for _ in 0..self.epochs {
            // Параллельный расчёт градиентов по каждому примеру.
            let (gw, gb) = x
                .par_iter()
                .zip(y.par_iter())
                .map(|(xi, yi)| {
                    let pred = self.w * xi + self.b;
                    let err = pred - yi;
                    (err * xi, err)
                })
                .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

            self.w -= self.lr * gw / n;
            self.b -= self.lr * gb / n;
        }
    }

    #[inline]
    fn predict(&self, x: f64) -> f64 {
        self.w * x + self.b
    }

    fn weights(&self) -> (f64, f64) {
        (self.w, self.b)
    }
}

/// Помощник: масштабирует вектор к нулевому среднему и единичному std (z-score).
/// Ускоряет сходимость градиентного спуска.
pub fn standardize(x: &[f64]) -> (Vec<f64>, f64, f64) {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
    let std = var.sqrt().max(1e-8);
    let scaled = x.iter().map(|v| (v - mean) / std).collect();
    (scaled, mean, std)
}
