//! Главная точка входа программы анализа набора данных Breast Cancer.
//!
//! Программа демонстрирует применение принципов SOLID, ООП, макросов
//! и многопоточности на Rust для решения задачи машинного обучения.
//!
//! Данные разрешаются в следующем порядке (Chain of Responsibility):
//!   1. Переменная окружения `BC_DATA_PATH`.
//!   2. Файл `breast_cancer.csv` в текущей директории.
//!   3. Встроенный в бинарник CSV (никогда не падает).
//!
//! Запуск: `echo 3 | cargo run --release` (или `cargo run --release` с вводом).

mod dataset;
mod stats;
mod reporter;
mod app;

use anyhow::Result;
use std::io;

use crate::app::App;
use crate::dataset::{ColumnarDataset, load_with_fallback};
use crate::reporter::ConsoleReporter;
use crate::stats::ParallelMeanCalculator;

fn main() -> Result<()> {
    println!("=== Breast Cancer Dataset Analyzer (Rust + ML crates) ===");
    println!();

    // Читаем n из стандартного ввода.
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let n: usize = input
        .trim()
        .parse()
        .map_err(|e| anyhow::anyhow!("Не удалось распарсить n как целое число: {e}"))?;

    // Разрешаем источник данных с fallback на встроенный CSV.
    let (row, source_info) = load_with_fallback()?;
    eprintln!("(источник данных: {source_info})");

    // DIP: App зависит от абстракций (трейтов), а не от конкретных типов.
    // Загружаем row-major CSV, затем транспонируем в column-major для
    // эффективного slice-доступа при параллельных вычислениях (rayon).
    let dataset = ColumnarDataset::from_row_major(&row);
    let calculator = ParallelMeanCalculator;
    let reporter = ConsoleReporter::new();

    let app = App::new(dataset, calculator, reporter);
    app.run(n)?;

    Ok(())
}
