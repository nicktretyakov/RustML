//! Drops rows whose score is missing and counts the casualties.

use crate::models::{RawRow, SurveyAnswer};
use crate::traits::DataCleaner;

/// Keeps only fully-answered rows; records how many were dropped.
///
/// Single Responsibility: this type only knows the rule "a row is usable
/// when it has a numeric score" -- nothing about where the rows came from
/// or how the survivors will be aggregated.
#[derive(Default)]
pub struct DropMissingCleaner;

impl DropMissingCleaner {
    pub fn new() -> Self {
        Self
    }
}

impl DataCleaner for DropMissingCleaner {
    fn clean(&self, rows: Vec<RawRow>) -> (Vec<SurveyAnswer>, usize) {
        let mut missing = 0usize;
        let kept: Vec<SurveyAnswer> = rows
            .into_iter()
            .filter_map(|row| match row.score_value() {
                Some(score) => Some(SurveyAnswer {
                    device: row.device,
                    age_group: row.age_group,
                    score,
                }),
                None => {
                    missing += 1;
                    None
                }
            })
            .collect();
        (kept, missing)
    }
}
