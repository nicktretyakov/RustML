//! Domain model: a `Trip` and a named numeric `Group`.
//!
//! SRP: these types only carry data, they do not load, compute or plot.

use serde::Deserialize;

/// Travel mode. Serialised from the CSV column `mode` (lowercase: "bike"/"bus").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Bike,
    Bus,
}

impl Mode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Mode::Bike => "bike",
            Mode::Bus => "bus",
        }
    }
}

/// A single survey row.
#[derive(Debug, Clone, Deserialize)]
pub struct Trip {
    #[serde(deserialize_with = "deserialize_mode")]
    pub mode: Mode,
    pub minutes: f64,
}

/// Tolerant deserialiser: accept "bike"/"bus" in any case, ignore surrounding whitespace.
/// Keeps parsing robust without leaking that concern into the model.
fn deserialize_mode<'de, D>(de: D) -> Result<Mode, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let raw = String::deserialize(de)?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "bike" => Ok(Mode::Bike),
        "bus" => Ok(Mode::Bus),
        other => Err(serde::de::Error::custom(format!("unknown mode: {other:?}"))),
    }
}

/// A named group of numeric observations (e.g. all bike minutes).
#[derive(Debug, Clone)]
pub struct Group {
    pub name: String,
    pub values: Vec<f64>,
}
