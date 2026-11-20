//! Модуль `classify` — определение квартиля клиента (SRP, ISP).
//!
//! Trait [`Classifier`] изолирует правило отнесения значения к квартилю.
//! Реализация [`ThresholdClassifier`] точно повторяет логику Python-версии:
//! 1 — меньше Q1, 2 — меньше медианы, 3 — меньше Q3, иначе 4.

use crate::stats::QuartileSummary;

/// Классификатор квартиля клиента.
pub trait Classifier {
    fn classify(&self, value: f64, summary: &QuartileSummary) -> u8;
}

/// Пороговая классификация по Q1/median/Q3.
pub struct ThresholdClassifier;

impl Classifier for ThresholdClassifier {
    fn classify(&self, value: f64, s: &QuartileSummary) -> u8 {
        if value < s.q1 {
            1
        } else if value < s.median {
            2
        } else if value < s.q3 {
            3
        } else {
            4
        }
    }
}
