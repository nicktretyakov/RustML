//! Domain model.
//!
//! Single Responsibility (SOLID-S): this module only describes the
//! `HousePrices` value object. It knows nothing about how prices are read,
//! how statistics are computed, or how results are printed.

use ndarray::Array1;

/// A street of houses, each carrying a price expressed in thousands.
///
/// Wraps an `ndarray` one-dimensional array (`Array1<f64>`), the same kind of
/// typed n-dimensional container that powers Rust ML libraries, so the model
/// can be handed directly to algorithms further down the pipeline.
#[derive(Debug, Clone)]
pub struct HousePrices {
    data: Array1<f64>,
}

impl HousePrices {
    /// Build from an owned vector of prices.
    #[allow(dead_code)] // convenience constructor, exercised under cfg(test)
    pub fn from_vec(values: Vec<f64>) -> Self {
        Self {
            data: Array1::from(values),
        }
    }

    /// Build from an existing `Array1`.
    pub fn new(data: Array1<f64>) -> Self {
        Self { data }
    }

    /// Borrow the underlying tensor view.
    pub fn data(&self) -> &Array1<f64> {
        &self.data
    }

    /// Number of houses on the street.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// `true` when the street has no houses.
    #[allow(dead_code)] // query helper, exercised under cfg(test)
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
