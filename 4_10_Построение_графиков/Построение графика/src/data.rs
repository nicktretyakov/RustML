//! Data access layer.
//!
//! **SRP** — `CsvDataSource` does exactly one thing: turn a CSV file into a
//! `Vec<RiverRecord>`. **OCP / DIP** — callers depend on the [`DataSource`]
//! trait, so a future `SqlDataSource` or `MockDataSource` can be substituted
//! without touching the analysis pipeline.

use std::fmt;

use crate::model::RiverRecord;

/// Anything that can produce a vector of river-level readings.
///
/// The `Send` bound lets the source be moved across threads (used by the
/// orchestrator's concurrent fit/plot step).
pub trait DataSource: Send {
    fn load(&self, path: &str) -> Result<Vec<RiverRecord>, DataError>;
}

/// Errors raised while reading raw data.
#[derive(Debug)]
pub enum DataError {
    Io(String),
    Parse(String),
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::Io(msg) => write!(f, "data I/O error: {msg}"),
            DataError::Parse(msg) => write!(f, "data parse error: {msg}"),
        }
    }
}

impl std::error::Error for DataError {}

impl From<std::io::Error> for DataError {
    fn from(err: std::io::Error) -> Self {
        DataError::Io(err.to_string())
    }
}

impl From<csv::Error> for DataError {
    fn from(err: csv::Error) -> Self {
        // csv::Error is either an I/O or a (de)serialisation error; route both
        // through the appropriate variant. `err.to_string()` carries the full
        // message regardless of kind.
        let msg = err.to_string();
        match err.kind() {
            csv::ErrorKind::Io(_) => DataError::Io(msg),
            _ => DataError::Parse(msg),
        }
    }
}

/// Reads `river.csv`-style files via the `csv` + `serde` crates.
#[derive(Debug, Default, Clone, Copy)]
pub struct CsvDataSource;

impl CsvDataSource {
    pub fn new() -> Self {
        Self
    }
}

impl DataSource for CsvDataSource {
    fn load(&self, path: &str) -> Result<Vec<RiverRecord>, DataError> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(false)
            .from_path(path)?;
        let mut out = Vec::new();
        for (idx, result) in reader.deserialize().enumerate() {
            let record: RiverRecord =
                result.map_err(|e| DataError::Parse(format!("row {idx}: {e}")))?;
            out.push(record);
        }
        if out.is_empty() {
            return Err(DataError::Parse(format!(
                "no data rows found in '{path}'"
            )));
        }
        Ok(out)
    }
}
