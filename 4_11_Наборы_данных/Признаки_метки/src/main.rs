//! digits_ml — Rust-реализация задачи scikit-learn digits.
//!
//! Демонстрирует:
//!   • крейты: csv, ndarray, rayon, serde, rand, anyhow, clap;
//!   • декларативный макрос `declare_counter!` (свой DSL над классами цифр);
//!   • ООП: структуры с инкапсуляцией, трейты, полиморфизм через trait-объекты;
//!   • многопоточность: rayon (параллельный подсчёт и предсказание),
//!     Arc + Mutex для потокобезопасной агрегации гистограммы меток;
//!   • SOLID: трейты в `domain` (DIP/ISP/OCP), узкие обязанности (SRP),
//!     взаимозаменяемые реализации (LSP);
//!   • машинное обучение: k-NN классификатор, train/test split, accuracy.
//!
//! Запуск:
//!   cargo run --release -- --digit 3 \
//!       --features data/digits_features.csv \
//!       --labels   data/digits_labels.csv

mod domain;
mod infra;
mod ml;

use crate::domain::{Classifier, Counter, DatasetRepository, Preprocessor, Reporter};
use crate::infra::{
    ConsoleReporter, CsvDatasetRepository, MinMaxScaler, ParallelLabelCounter,
};
use crate::ml::{accuracy, KnnClassifier};

use anyhow::Result;
use clap::Parser;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::PathBuf;
use std::sync::Arc;

/// CLI-аргументы. Программа «считывает» цифру d (по условию задачи).
#[derive(Parser, Debug)]
#[command(name = "digits_ml", about = "digits dataset + k-NN in Rust (SOLID, threads, ML)")]
struct Cli {
    /// Цифра d, для которой считаем число образцов.
    #[arg(long = "digit")]
    digit: u8,

    /// CSV-файл с признаками (64 колонки, БЕЗ метки).
    #[arg(long = "features", default_value = "data/digits_features.csv")]
    features: PathBuf,

    /// CSV-файл с метками (одна колонка `label`).
    #[arg(long = "labels", default_value = "data/digits_labels.csv")]
    labels: PathBuf,

    /// Число соседей в k-NN.
    #[arg(long = "k", default_value_t = 5)]
    k: usize,

    /// Доля тестовой выборки (0..1).
    #[arg(long = "test-size", default_value_t = 0.2)]
    test_size: f64,
}

/// Декларативный макрос: создаёт статический список классов цифр и
/// константную функцию-проверку, что d ∈ [0..10). Показывает собственный DSL
/// поверх домена (макросы Rust — отдельный механизм от трейтов/ОШОП).
macro_rules! declare_counter {
    ($name:ident, [$($d:literal),* $(,)?]) => {
        struct $name;
        impl $name {
            /// Все классы, описанные макросом.
            const CLASSES: &'static [u8] = &[$($d),*];
            /// true, если метка входит в объявленный список классов.
            const fn contains(label: u8) -> bool {
                let mut i = 0;
                while i < Self::CLASSES.len() {
                    if Self::CLASSES[i] == label { return true; }
                    i += 1;
                }
                false
            }
        }
    };
}

declare_counter!(DigitsClasses, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

/// Разделение выборки на train/test по индексам (детерминированное перемешивание).
fn train_test_split(n: usize, test_size: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    let split = ((n as f64) * test_size).round() as usize;
    let test = idx[..split].to_vec();
    let train = idx[split..].to_vec();
    (train, test)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let d = cli.digit;
    anyhow::ensure!(DigitsClasses::contains(d), "d должна быть цифрой 0..=9");

    // ── DIP: main зависит только от трейтов, не от конкретных типов ──
    let repo: Box<dyn DatasetRepository> = Box::new(CsvDatasetRepository::new(
        &cli.features,
        &cli.labels,
        64, // ровно 64 признака-пикселя
    ));
    let dataset = repo.load()?;

    let x_shape = (dataset.x.nrows(), dataset.x.ncols());
    let y_shape = dataset.y.len();

    // ── Предобработка (нормирование признаков) ──
    let scaler = MinMaxScaler::new(0.0, 16.0); // пиксели digits в диапазоне 0..16
    let x_norm = scaler.fit_transform(&dataset.x)?;

    // ── Многопоточный счётчик меток (Arc + Mutex + rayon map-reduce) ──
    let counter: Arc<dyn Counter> = Arc::new(ParallelLabelCounter::from_labels(&dataset.y));

    // ── Репортёр печатает требуемые строки (формат 1-в-1 с Python) ──
    let reporter = ConsoleReporter::new(x_shape, y_shape, counter.clone());
    reporter.report(d)?;

    // ── ML: обучаем k-NN и оцениваем accuracy на отложенной выборке ──
    let (train_idx, test_idx) = train_test_split(y_shape, cli.test_size, 42);

    let mut train_x = ndarray::Array2::zeros((train_idx.len(), x_norm.ncols()));
    let mut train_y = Vec::with_capacity(train_idx.len());
    for (i, &gi) in train_idx.iter().enumerate() {
        train_x.row_mut(i).assign(&x_norm.row(gi));
        train_y.push(dataset.y[gi]);
    }
    let mut test_x = ndarray::Array2::zeros((test_idx.len(), x_norm.ncols()));
    let mut test_y = Vec::with_capacity(test_idx.len());
    for (i, &gi) in test_idx.iter().enumerate() {
        test_x.row_mut(i).assign(&x_norm.row(gi));
        test_y.push(dataset.y[gi]);
    }

    let mut clf: Box<dyn Classifier> = Box::new(KnnClassifier::new(cli.k));
    clf.fit(&train_x, &train_y)?;
    let preds = clf.predict(&test_x)?;
    let acc = accuracy(&preds, &test_y);

    // Альтернативный путь обучения — через обобщённый трейт `Sample`:
    // строим доменные объекты DigitSample и обучаем через fit_samples.
    // Демонстрирует обобщённое ООП: тот же классификатор, другой носитель данных.
    let train_samples: Vec<domain::DigitSample> = train_idx
        .iter()
        .map(|&gi| domain::DigitSample::new(x_norm.row(gi).to_vec(), dataset.y[gi]))
        .collect();
    let mut clf2 = KnnClassifier::new(cli.k);
    clf2.fit_samples(&train_samples)?;
    let preds2 = clf2.predict(&test_x)?;
    let acc2 = accuracy(&preds2, &test_y);

    eprintln!();
    eprintln!("── ML (k-NN, k={}, test_size={:.0}%) ──", cli.k, cli.test_size * 100.0);
    eprintln!("train: {} samples", train_y.len());
    eprintln!("test : {} samples", test_y.len());
    eprintln!("accuracy (matrix fit)     : {:.4}", acc);
    eprintln!("accuracy (Sample<T> fit)  : {:.4}", acc2);

    eprintln!();
    eprintln!("── распределение классов (считано параллельно) ──");
    for (label, count) in counter.distribution()? {
        let bar = "█".repeat((count as f64 / 183.0 * 40.0).round() as usize);
        eprintln!("  digit {}: {:>3}  {}", label, count, bar);
    }

    Ok(())
}
