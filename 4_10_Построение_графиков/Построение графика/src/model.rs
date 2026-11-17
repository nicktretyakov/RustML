//! Domain model: a single river-level sensor reading.
//!
//! The `#[derive(...)]` attribute is a *procedural macro* provided by `serde`
//! and the standard library. `RiverRecord` mirrors the columns of `river.csv`
//! (`hour`, `level_cm`, `rain_mm`) so the `csv` crate can deserialise each row
//! directly into a typed value — the Rust analogue of a pandas `DataFrame` row.

use serde::{Deserialize, Serialize};

/// One hourly measurement from the river-level sensor.
///
/// * `hour`     — hours since the start of the observation window (0..=23).
/// * `level_cm` — water level at the gauge, in centimetres.
/// * `rain_mm`  — rainfall recorded during the preceding hour, in millimetres.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiverRecord {
    pub hour: f64,
    pub level_cm: f64,
    pub rain_mm: f64,
}

impl RiverRecord {
    /// Total water "input" attributable to a single hour, used by the ML model
    /// as the raw signal that the preprocessor turns into the cumulative-rain
    /// feature. Exposed as a method so the feature definition lives next to the
    /// data it describes (cohesion / SRP).
    #[inline]
    pub fn feature_rain(&self) -> f64 {
        self.rain_mm
    }
}
