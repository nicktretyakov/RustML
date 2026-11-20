//! Модуль `engine` — высокоуровневый движок анализа (DIP, OCP).
//!
//! [`AnalysisEngine`] зависит только от trait-ов `QuartileEstimator`,
//! `Classifier` и `Reporter`, а не от их конкретных реализаций. Это ядро,
//! собирающее конвейер: данные -> квартили -> классификация -> отчёт.

use std::error::Error;

use crate::classify::Classifier;
use crate::data::DataSource;
use crate::report::Reporter;
use crate::stats::{QuartileEstimator, QuartileSummary};

/// Ошибка анализа (пустая выборка и т.п.).
#[derive(Debug)]
pub struct AnalysisError(String);

impl std::fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "analysis error: {}", self.0)
    }
}

impl Error for AnalysisError {}

/// Движок анализа расхода воды.
pub struct AnalysisEngine<E, C, R>
where
    E: QuartileEstimator,
    C: Classifier,
    R: Reporter,
{
    estimator: E,
    classifier: C,
    reporter: R,
}

impl<E, C, R> AnalysisEngine<E, C, R>
where
    E: QuartileEstimator,
    C: Classifier,
    R: Reporter,
{
    pub fn new(estimator: E, classifier: C, reporter: R) -> Self {
        Self { estimator, classifier, reporter }
    }

    /// Рассчитывает квартили по выборке и печатает сводку.
    pub fn analyse(&mut self, data: &dyn DataSource) -> Result<QuartileSummary, AnalysisError> {
        if data.size() == 0 {
            return Err(AnalysisError("sample is empty".into()));
        }
        let summary = self.estimator.estimate(data);
        self.reporter
            .report_summary(&summary)
            .map_err(|e| AnalysisError(e.to_string()))?;
        Ok(summary)
    }

    /// Классифицирует значение клиента и печатает результат.
    pub fn classify_and_report(
        &mut self,
        summary: &QuartileSummary,
        client_value: f64,
    ) -> Result<u8, AnalysisError> {
        let quarter = self.classifier.classify(client_value, summary);
        self.reporter
            .report_client(quarter)
            .map_err(|e| AnalysisError(e.to_string()))?;
        Ok(quarter)
    }
}
