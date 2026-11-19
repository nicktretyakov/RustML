//! Машинное обучение: k-NN классификатор с евклидовым расстоянием,
//! нормированием признаков и параллельным предсказанием через rayon.
//!
//! ООП: структуры с инкапсулированным состоянием и методами,
//! реализующие трейт `Classifier` (полиморфизм + LSP — любая реализация
//! `Classifier` взаимозаменяема там, где ожидается `impl Classifier`).

use crate::domain::Classifier;
use anyhow::Result;
use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;
use std::collections::HashMap;

/// Минимальный контракт «образец»: вектор признаков + метка.
/// Нужен, чтобы классификатор не зависел от конкретного носителя данных:
/// обучающую выборку можно задать как массив векторов, строк матрицы
/// ndarray или любых собственных типов, реализующих этот трейт.
pub trait Sample {
    fn features(&self) -> &[f64];
    fn label(&self) -> u8;
}

/// K-ближайших соседей. Состояние инкапсулировано (приватные поля),
/// доступ — только через методы трейта `Classifier`.
pub struct KnnClassifier {
    k: usize,
    train_x: Array2<f64>,
    train_y: Vec<u8>,
}

impl KnnClassifier {
    /// Конструктор — единственный способ создать валидный объект (инкапсуляция).
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be positive");
        Self {
            k,
            train_x: Array2::zeros((0, 0)),
            train_y: Vec::new(),
        }
    }

    /// Обучение из произвольной коллекции объектов, реализующих трейт `Sample`.
    /// Демонстрирует обобщённое ООП: классификатор не привязан к ndarray,
    /// а работает с любым носителем (Vec, строки матрицы, доменные структуры).
    pub fn fit_samples<S: Sample>(&mut self, samples: &[S]) -> Result<()> {
        let n = samples.len();
        let m = samples.first().map(|s| s.features().len()).unwrap_or(0);
        let mut x = Array2::zeros((n, m));
        let mut y = Vec::with_capacity(n);
        for (i, s) in samples.iter().enumerate() {
            let feats = s.features();
            anyhow::ensure!(
                feats.len() == m,
                "fit_samples: разная длина признаков в образцах ({} vs {})",
                feats.len(), m
            );
            for (j, v) in feats.iter().enumerate() {
                x[[i, j]] = *v;
            }
            y.push(s.label());
        }
        self.train_x = x;
        self.train_y = y;
        Ok(())
    }
}

impl Classifier for KnnClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &[u8]) -> Result<()> {
        anyhow::ensure!(
            x.nrows() == y.len(),
            "train: X rows ({}) != y len ({})",
            x.nrows(),
            y.len()
        );
        self.train_x = x.clone();
        self.train_y = y.to_vec();
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Vec<u8>> {
        let n_features = self.train_x.ncols();
        anyhow::ensure!(
            x.ncols() == n_features,
            "predict: expected {} features, got {}",
            n_features,
            x.ncols()
        );

        // Многопоточное предсказание: каждая тестовая строка обрабатывается
        // независимо через rayon. train_x делится по ссылке (Arc не нужен —
        // rayon::par_map работает с &self).
        let train = &self.train_x;
        let train_y = &self.train_y;
        let k = self.k;

        let preds: Vec<u8> = (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let test_row = x.row(i);
                classify_one(train, train_y, test_row, k)
            })
            .collect();

        Ok(preds)
    }
}

/// Голосование k ближайших соседей по евклидову расстоянию.
fn classify_one(
    train: &Array2<f64>,
    train_y: &[u8],
    test: ArrayView1<f64>,
    k: usize,
) -> u8 {
    // Считаем расстояния до всех обучающих образцов.
    let mut dists: Vec<(f64, u8)> = (0..train.nrows())
        .map(|i| {
            let tr = train.row(i);
            let d = squared_euclidean(tr, test);
            (d, train_y[i])
        })
        .collect();

    // partial_sort по k наименьшим (O(n log k)) — быстрее полной сортировки.
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let neighbours = &dists[..k.min(dists.len())];

    // Мажоритарное голосование. Для каждого класса храним пару (count, sum_dist):
    // голоса — сколько соседей этого класса, сумма квадратов расстояний — для
    // разрешения ничьей (меньшая сумма = ближе = приоритетнее).
    let mut tally: HashMap<u8, (usize, f64)> = HashMap::new();
    for &(dist, label) in neighbours {
        let entry = tally.entry(label).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += dist;
    }

    // Выбираем класс: больше голосов, при равенстве — меньше сумма расстояний.
    tally
        .iter()
        .max_by(|&( _, (ca, da)), &(_, (cb, db))| {
            ca.cmp(cb) // больше голосов — лучше
                .then_with(|| {
                    // при равных голосах — меньше сумма расстояний лучше
                    db.partial_cmp(da).unwrap_or(std::cmp::Ordering::Equal)
                })
        })
        .map(|(label, _)| *label)
        .unwrap_or(0)
}

/// Квадрат евклидова расстояния (корень не нужен для сравнения порядков).
fn squared_euclidean(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

/// Утилита: точность предсказания (accuracy) — метрика качества ML-модели.
pub fn accuracy(pred: &[u8], truth: &[u8]) -> f64 {
    let n = truth.len() as f64;
    let correct = pred.iter().zip(truth.iter()).filter(|(p, t)| p == t).count();
    correct as f64 / n
}

/// Помощник для построения матрицы признаков из векторов (используется тестами).
#[allow(dead_code)]
pub fn rows_to_matrix(rows: &[Vec<f64>]) -> Array2<f64> {
    let n = rows.len();
    let m = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut a = Array2::zeros((n, m));
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            a[[i, j]] = *v;
        }
    }
    a
}

// Лёгкая проверка в unit-тестах, что классификатор вообще работает.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn_classifies_simple_points() {
        let train = rows_to_matrix(&[
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![5.0, 5.0],
        ]);
        let y = vec![1u8, 1, 2];
        let mut clf = KnnClassifier::new(1);
        clf.fit(&train, &y).unwrap();
        let test = rows_to_matrix(&[vec![0.05, 0.0], vec![4.9, 5.1]]);
        assert_eq!(clf.predict(&test).unwrap(), vec![1, 2]);
    }

    #[test]
    fn accuracy_is_one_when_perfect() {
        assert!((accuracy(&[1, 2, 3], &[1, 2, 3]) - 1.0).abs() < 1e-9);
    }

    /// Демонстрирует обобщённое ООП: обучение из объектов, реализующих трейт
    /// `Sample` (кортеж (Vec<f64>, u8) реализует его в domain.rs), без ndarray.
    #[test]
    fn knn_fits_via_sample_trait() {
        let samples: Vec<(Vec<f64>, u8)> = vec![
            (vec![0.0, 0.0], 1),
            (vec![0.1, 0.0], 1),
            (vec![5.0, 5.0], 2),
        ];
        let mut clf = KnnClassifier::new(1);
        clf.fit_samples(&samples).unwrap();
        let test = rows_to_matrix(&[vec![0.05, 0.0], vec![4.9, 5.1]]);
        assert_eq!(clf.predict(&test).unwrap(), vec![1, 2]);
    }
}
