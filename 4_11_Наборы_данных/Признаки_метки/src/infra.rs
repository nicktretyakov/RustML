//! Инфраструктурные реализации доменных трейтов: CSV-репозиторий,
//! скейлер признаков, потокобезопасный счётчик меток, репортёр.
//!
//! Конкретные реализации — точка расширения (OCP): добавить паркетный
//! репозиторий или робастный скейлер можно, не трогая main и трейты.

use crate::domain::{Counter, Dataset, DatasetRepository, Preprocessor, Reporter};
use anyhow::{anyhow, Context, Result};
use ndarray::Array2;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// ──────────────────────────────────────────────────────────────────────
// CSV-репозиторий: читает признаки (64 пикселя) и метки из двух файлов.
// Важно: X содержит РОВНО 64 признака — метка в него не подмешивается,
// что и исправляет баг коллеги, у которого выходило (1797, 65).
// ──────────────────────────────────────────────────────────────────────
pub struct CsvDatasetRepository {
    features_path: PathBuf,
    labels_path: PathBuf,
    n_features: usize,
}

impl CsvDatasetRepository {
    pub fn new(features_path: impl Into<PathBuf>, labels_path: impl Into<PathBuf>, n_features: usize) -> Self {
        Self {
            features_path: features_path.into(),
            labels_path: labels_path.into(),
            n_features,
        }
    }
}

impl DatasetRepository for CsvDatasetRepository {
    fn load(&self) -> Result<Dataset> {
        // --- Признаки ---
        let mut feat_reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(BufReader::new(File::open(&self.features_path)?));

        let feat_records: Vec<Vec<f64>> = feat_reader
            .deserialize()
            .collect::<Result<Vec<Vec<f64>>, _>>()
            .context("parse features CSV")?;

        let n_rows = feat_records.len();
        let n_cols = feat_records.first().map(|r| r.len()).unwrap_or(0);
        if n_cols != self.n_features {
            return Err(anyhow!(
                "X должен содержать {} признака, а содержит {}. Метка не должна попадать в X.",
                self.n_features, n_cols
            ));
        }

        let mut x = Array2::zeros((n_rows, n_cols));
        for (i, row) in feat_records.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                x[[i, j]] = *v;
            }
        }

        // --- Метки ---
        let mut lbl_reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(BufReader::new(File::open(&self.labels_path)?));

        #[derive(serde::Deserialize)]
        struct LabelRow {
            label: u8,
        }
        let y: Vec<u8> = lbl_reader
            .deserialize::<LabelRow>()
            .map(|r| r.map(|row| row.label))
            .collect::<Result<Vec<u8>, _>>()
            .context("parse labels CSV")?;

        if y.len() != n_rows {
            return Err(anyhow!(
                "X rows ({}) != y len ({}): признаки и метки рассинхронизированы",
                n_rows, y.len()
            ));
        }

        Ok(Dataset { x, y })
    }
}

// ──────────────────────────────────────────────────────────────────────
// Скейлер: нормирование признаков в [0, 1] (пиксели 0..16 -> 0..1).
// ──────────────────────────────────────────────────────────────────────
pub struct MinMaxScaler {
    min: f64,
    max: f64,
}

impl MinMaxScaler {
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }
}

impl Preprocessor for MinMaxScaler {
    fn fit_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let span = (self.max - self.min).max(1e-9);
        Ok(x.mapv(|v| (v - self.min) / span))
    }
}

// ──────────────────────────────────────────────────────────────────────
// Потокобезопасный счётчик меток: распределение считается параллельно
// (rayon) и агрегируется в общий Mutex<HashMap> (Arc для разделения).
// ──────────────────────────────────────────────────────────────────────
pub struct ParallelLabelCounter {
    distribution: Arc<Mutex<HashMap<u8, usize>>>,
}

impl ParallelLabelCounter {
    pub fn from_labels(labels: &[u8]) -> Self {
        // Чанками по ~256 элементов считаем локальные гистограммы в rayon,
        // затем сливаем в общий Mutex<Map> — классический map-reduce.
        let shared = Arc::new(Mutex::new(HashMap::<u8, usize>::new()));

        labels
            .par_chunks(256)
            .for_each(|chunk| {
                let mut local = HashMap::<u8, usize>::new();
                for &l in chunk {
                    *local.entry(l).or_insert(0) += 1;
                }
                let mut guard = shared.lock().expect("mutex poisoned");
                for (k, v) in local {
                    *guard.entry(k).or_insert(0) += v;
                }
            });

        Self { distribution: shared }
    }
}

impl Counter for ParallelLabelCounter {
    fn count_for(&self, label: u8) -> Result<usize> {
        let guard = self.distribution.lock().expect("mutex poisoned");
        guard
            .get(&label)
            .copied()
            .ok_or_else(|| anyhow!("цифра {} отсутствует в датасете", label))
    }

    fn distribution(&self) -> Result<Vec<(u8, usize)>> {
        let guard = self.distribution.lock().expect("mutex poisoned");
        let mut v: Vec<(u8, usize)> = guard.iter().map(|(k, c)| (*k, *c)).collect();
        v.sort_by_key(|(k, _)| *k);
        Ok(v)
    }
}

// ──────────────────────────────────────────────────────────────────────
// Репортёр: печатает результат в требуемом формате.
// ──────────────────────────────────────────────────────────────────────
pub struct ConsoleReporter {
    x_shape: (usize, usize),
    y_shape: usize,
    counter: Arc<dyn Counter>,
}

impl ConsoleReporter {
    pub fn new(x_shape: (usize, usize), y_shape: usize, counter: Arc<dyn Counter>) -> Self {
        Self { x_shape, y_shape, counter }
    }
}

impl Reporter for ConsoleReporter {
    fn report(&self, d: u8) -> Result<()> {
        // Строго требуемый формат вывода (совпадает с Python-версией).
        println!("X: ({}, {})", self.x_shape.0, self.x_shape.1);
        println!("y: ({},)", self.y_shape);
        let n = self.counter.count_for(d)?;
        println!("Samples of digit {}: {}", d, n);
        Ok(())
    }
}
