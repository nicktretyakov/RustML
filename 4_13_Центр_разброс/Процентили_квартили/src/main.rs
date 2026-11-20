//! water_quartiles — анализ квартилей расхода воды.
//!
//! Архитектура построена по принципам SOLID:
//! - **SRP**: каждый модуль отвечает за одну задачу (ввод, данные, статистика,
//!   классификация, отчёт).
//! - **OCP**: новые способы расчёта квартилей или источники данных добавляются
//!   реализацией соответствующих trait-ов без изменения существующего кода.
//! - **LSP**: любые реализации `QuartileEstimator` взаимозаменяемы.
//! - **ISP**: trait-ы минимальны (`DataSource`, `QuartileEstimator`,
//!   `Classifier`, `Reporter`).
//! - **DIP**: высокоуровневый движок (`AnalysisEngine`) зависит только от
//!   trait-ов, а не от конкретных типов.
//!
//! Крейты для машинного обучения/статистики:
//! - [`ndarray`] — многомерные массивы (аналог `numpy`),
//! - [`statrs`] — описательная статистика и распределения.

mod data;
mod stats;
mod classify;
mod report;
mod macros;
mod engine;

use std::error::Error;

use data::StreetConsumption;
use stats::LinearQuartileEstimator;
use classify::ThresholdClassifier;
use report::StdoutReporter;
use engine::AnalysisEngine;

fn main() -> Result<(), Box<dyn Error>> {
    // Данные о расходе воды (литры) в 20 домах на улице.
    let usage = vec![
        312.0_f64, 285.0, 410.0, 198.0, 356.0, 274.0, 520.0, 301.0, 289.0, 345.0,
        262.0, 398.0, 230.0, 610.0, 327.0, 295.0, 368.0, 251.0, 334.0, 280.0,
    ];
    let street = StreetConsumption::new(usage);

    let estimator = LinearQuartileEstimator::new();
    let classifier = ThresholdClassifier;
    let reporter = StdoutReporter;

    let mut engine = AnalysisEngine::new(estimator, classifier, reporter);
    let result = engine.analyse(&street)?;

    let client_liters = macros::read_client_liters()?;
    engine.classify_and_report(&result, client_liters)?;
    Ok(())
}
