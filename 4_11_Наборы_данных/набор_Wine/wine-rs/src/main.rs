//! Точка входа программы анализа вин.
//!
//! Архитектура (SOLID):
//!
//! * `FeatureSource` (трейт) ← `WineDataset` (linfa)  — DIP + ISP + OCP.
//! * `StatsComputer` (трейт) ← `ParallelStats` (rayon) — DIP + OCP.
//! * `main` зависит только от трейтов и не знает о linfa/rayon напрямую,
//!   кроме как через функции-конструкторы.
//!
//! Параллелизм: `rayon` внутри `ParallelStats` использует все ядра.
//! ML-крейт: `linfa::Dataset` подтверждает загрузку данных в привычном
//! для Rust-экосистемы формате.

mod dataset;
mod macros;
mod stats;
mod traits;

use std::io::{self, BufRead};

use dataset::{target_name, WineDataset};
use stats::ParallelStats;
use traits::{FeatureSource, StatsComputer};

fn main() {
    // 1. Чтение названия характеристики из stdin.
    let stdin = io::stdin();
    let feature = stdin
        .lock()
        .lines()
        .next()
        .unwrap_or(Ok(String::new()))
        .unwrap_or_default()
        .trim()
        .to_string();

    if feature.is_empty() {
        eprintln!("ошибка: не задано название характеристики");
        std::process::exit(1);
    }

    // 2. Загрузка данных: путь берётся из переменной окружения (по умолчанию
    //    data/wine.csv относительно запуска), что упрощает тесты (OCP/DIP).
    let csv_path = std::env::var("WINE_CSV_PATH").unwrap_or_else(|_| "data/wine.csv".to_string());

    let ds = match WineDataset::from_csv(&csv_path) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("ошибка загрузки {}: {}", csv_path, e);
            std::process::exit(2);
        }
    };

    // Сохраняем имена признаков до передачи owned-данных в вычислитель.
    let feature_names: Vec<String> = ds.feature_names().to_vec();

    // Проверка, что запрошенная характеристика существует.
    if !feature_names.iter().any(|n| n == &feature) {
        eprintln!(
            "ошибка: характеристика '{}' не найдена. Доступные: {}",
            feature,
            feature_names.join(", ")
        );
        std::process::exit(3);
    }

    // DIP: передаём owned-данные в вычислитель, не раскрывая источник.
    let (records, targets, feature_index, _) = ds.into_parts();
    let stats = ParallelStats::new(records, targets, feature_index);

    // 3. Факт №1: класс с наибольшим количеством вин.
    let (top_class, top_count) = stats
        .most_frequent_class()
        .expect("должен быть хотя бы один класс");

    // 4. Факт №2: класс с наибольшим средним значением характеристики.
    let (hi_class, _hi_mean) = stats
        .class_with_highest_mean(&feature)
        .expect("должен быть хотя бы один класс");

    // 5. Факт №3: сколько вин этого класса выше среднего по всем винам.
    let above = stats
        .count_above_overall_mean(hi_class, &feature)
        .expect("должны быть значения");

    // 6. Вывод через декларативный макрос (формат = эталону sklearn).
    report!(most_wines, target_name(top_class), top_count);
    report!(highest, feature, target_name(hi_class));
    report!(above, target_name(hi_class), feature, above);
}
