//! Input reading.
//!
//! Single Responsibility (SOLID-S): this module only decodes the incoming
//! byte stream into a `HousePrices` value. It is the only place that knows
//! the on-the-wire format (`count` then `count` integer lines), so swapping
//! the source (file, network, test fixture) touches nothing else.

use crate::models::HousePrices;
use ndarray::Array1;
use std::io::{self, BufRead};

/// Abstraction over any producer of house-price data.
///
/// Dependency Inversion (SOLID-D): high-level orchestration depends on this
/// trait, not on a concrete reader.
pub trait InputSource {
    fn read_prices(&self) -> io::Result<HousePrices>;
}

/// Reads prices from standard input.
///
/// Format:
/// ```text
/// n            <- count of houses
/// p0           <- price of house 0 (integer, thousands)
/// p1
/// ...
/// p{n-1}
/// ```
pub struct StdinInputSource;

impl StdinInputSource {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StdinInputSource {
    fn default() -> Self {
        Self::new()
    }
}

impl InputSource for StdinInputSource {
    fn read_prices(&self) -> io::Result<HousePrices> {
        let stdin = io::stdin();
        let mut lines = stdin.lock().lines();

        let first = lines
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "missing count line"))??;
        let n: usize = first
            .trim()
            .parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

        let mut values: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            let line = lines
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "missing price line"))??;
            let v: f64 = line
                .trim()
                .parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
            values.push(v);
        }

        Ok(HousePrices::new(Array1::from(values)))
    }
}

/// Test-friendly source backed by an in-memory string. Lets unit tests feed
/// input without touching real stdin. Open/Closed (SOLID-O): new sources are
/// added by implementing the trait, without editing existing ones.
#[allow(dead_code)] // wired in the cfg(test) integration test in main.rs
pub struct BufferedInputSource {
    text: String,
}

impl BufferedInputSource {
    #[allow(dead_code)] // wired in the cfg(test) integration test in main.rs
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl InputSource for BufferedInputSource {
    fn read_prices(&self) -> io::Result<HousePrices> {
        let mut lines = self.text.lines();
        let first = lines
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "missing count line"))?;
        let n: usize = first
            .trim()
            .parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let mut values: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            let line = lines
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "missing price line"))?;
            let v: f64 = line
                .trim()
                .parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
            values.push(v);
        }
        Ok(HousePrices::new(Array1::from(values)))
    }
}
