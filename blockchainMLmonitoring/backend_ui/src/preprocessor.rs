// src/preprocessor.rs
use linfa::dataset::DatasetBase;
use ndarray::{Array1, Array2};

pub fn preprocess(records: Array2<f64>, targets: Array1<f64>) -> DatasetBase<Array2<f64>, Array1<f64>> {
    DatasetBase::new(records, targets)
}