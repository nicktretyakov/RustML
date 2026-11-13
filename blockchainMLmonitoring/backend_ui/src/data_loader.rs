// src/data_loader.rs

use ndarray::{Array1, Array2};

// Загрузка "сырых" данных (хардкод для демо)
// В реальном проекте здесь можно добавить загрузку из CSV, Parquet, базы и т.д. (например, через Polars)
pub fn load_raw_data() -> (Array2<f64>, Array1<f64>) {
    // Признаки: [age, has_large_text]
    let records = Array2::from(vec![
        [25.0, 1.0],
        [35.0, 0.0],
        [45.0, 1.0],
        [30.0, 0.0],
        [28.0, 1.0],
    ]);

    // Цели: произвольные значения для регрессии (порядок элемента)
    let targets = Array1::from(vec![0.0, 1.0, 0.5, 1.5, 0.8]);

    (records, targets)
}

// Пример загрузки из CSV с Polars (опционально, раскомментируйте и добавьте зависимость polars = { version = "0.39", features = ["ndarray"] })
/*
use polars::prelude::*;
use ndarray::{Array2, Array1};

pub fn load_from_csv(path: &str) -> Result<(Array2<f64>, Array1<f64>), PolarsError> {
    let df = CsvReadOptions::default()
        .with_try_parse_dates(true)
        .map(|opts| opts.infer_schema(None).with_missing_is_null(true))
        .try_into_reader_with_file_path(Some(path.into()) ... })?
        .finish()?;

    // Предполагаем, что последняя колонка — target, остальные — features
    let feature_cols = df.get_column_names();
    let target_col = feature_cols.last().unwrap().to_string();
    let feature_cols = &feature_cols[..feature_cols.len() - 1];

    let features_df = df.select(feature_cols)?;
    let targets_df = df.select([target_col])?;

    let features = features_df.to_ndarray::<Float64Type>()?.to_owned();
    let targets = targets_df.to_ndarray::<Float64Type>()?.into_raw_vec();
    let targets = Array1::from_vec(targets);

    Ok((features, targets))
}
*/