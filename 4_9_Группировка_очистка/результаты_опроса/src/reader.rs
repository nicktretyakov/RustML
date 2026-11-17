//! CSV-backed implementation of [`SurveyReader`].

use std::fs::File;
use std::path::PathBuf;

use csv::ReaderBuilder;

use crate::models::RawRow;
use crate::traits::SurveyReader;

/// Reads `survey.csv` (or any CSV with the same header) from disk.
pub struct CsvSurveyReader {
    path: PathBuf,
}

impl CsvSurveyReader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl SurveyReader for CsvSurveyReader {
    fn load(&self) -> Result<Vec<RawRow>, Box<dyn std::error::Error + Send + Sync>> {
        let file = File::open(&self.path)?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .flexible(false)
            .from_reader(file);
        let mut rows = Vec::new();
        for result in rdr.deserialize::<RawRow>() {
            rows.push(result?);
        }
        Ok(rows)
    }
}
