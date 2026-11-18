//! CSV loading, isolated behind a trait.
//!
//! * SRP: `CsvTripLoader` does only I/O.
//! * DIP: callers depend on `TripLoader` (abstraction), not on the concrete loader.
//! * OCP: a new source (JSON, DB...) is a new `TripLoader` impl, no caller change.

use std::error::Error;

use crate::models::Trip;

/// Abstraction over any source of `Trip`s.
pub trait TripLoader {
    fn load(&self, path: &str) -> Result<Vec<Trip>, Box<dyn Error>>;
}

/// Loads trips from a CSV file with a header `mode,minutes`.
pub struct CsvTripLoader;

impl CsvTripLoader {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CsvTripLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl TripLoader for CsvTripLoader {
    fn load(&self, path: &str) -> Result<Vec<Trip>, Box<dyn Error>> {
        // Wrap errors with the path so a wrong/missing file is reported as
        //   Error: cannot open CSV "commute.csv": ... No such file or directory ...
        // instead of a bare `Error(Io(Os { ... }))` that does not name the file.
        let mut reader = csv::Reader::from_path(path)
            .map_err(|e| format!("cannot open CSV {path:?}: {e}"))?;
        let trips: Vec<Trip> = reader
            .deserialize()
            .collect::<Result<_, _>>()
            .map_err(|e| format!("failed to parse a row in {path:?}: {e}"))?;
        Ok(trips)
    }
}
