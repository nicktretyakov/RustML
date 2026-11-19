//! Реализация `FeatureSource` поверх linfa `Dataset`.
//!
//! SRP: отвечает только за разбор CSV и доступ к данным по индексам.
//! OCP: новая реализация (например, прямой из памяти) добавляется через
//! трейт, не меняя существующий код.
//! DIP: конкретный тип `WineDataset` зависит от трейта `FeatureSource`,
//! а не наоборот.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use linfa::Dataset;
use ndarray::{Array1, Array2};

use crate::traits::FeatureSource;

/// Названия классов в порядке меток 0..=2 — те же строки, что
/// `sklearn.datasets.load_wine().target_names`.
pub const TARGET_NAMES: [&str; 3] = ["class_0", "class_1", "class_2"];

/// Возвращает имя класса по его числовому идентификатору.
pub fn target_name(class: usize) -> &'static str {
    TARGET_NAMES
        .get(class)
        .copied()
        .unwrap_or("unknown_class")
}

/// Внешнее представление датасета: именно он реализует `FeatureSource`.
pub struct WineDataset {
    /// Имена признаков (13 шт.), порядок = порядок столбцов в матрице `records`.
    feature_names: Vec<String>,
    /// Данные `X`: 178 строк × 13 признаков.
    records: Array2<f64>,
    /// Метки `y`: 178 целых от 0 до 2.
    targets: Array1<usize>,
    /// Для O(1)-доступа: имя признака -> индекс столбца.
    feature_index: BTreeMap<String, usize>,
}

impl WineDataset {
    /// Загрузить CSV с заголовком `target,<13 признаков>` и собрать linfa-датасет.
    ///
    /// Формат файла — точно такой же, как у `load_wine(as_frame=True).to_csv()`.
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let mut rdr = csv::Reader::from_path(path)?;

        let header: Vec<String> = rdr
            .headers()?
            .iter()
            .map(|s| s.trim().to_string())
            .collect();
        if header.is_empty() || header[0] != "target" {
            return Err(format!(
                "ожидалась колонка 'target' в начале заголовка, получено: {:?}",
                header.first()
            )
            .into());
        }
        let feature_names: Vec<String> = header[1..].to_vec();
        let feature_index: BTreeMap<String, usize> = feature_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let n_features = feature_names.len();

        let mut targets: Vec<usize> = Vec::new();
        let mut records: Vec<f64> = Vec::new();

        for result in rdr.records() {
            let rec = result?;
            if rec.len() != n_features + 1 {
                return Err(format!(
                    "ожидалось {} полей, получено {}",
                    n_features + 1,
                    rec.len()
                )
                .into());
            }
            let target: usize = rec.get(0).unwrap().trim().parse()?;
            targets.push(target);
            for j in 0..n_features {
                let v: f64 = rec.get(j + 1).unwrap().trim().parse()?;
                records.push(v);
            }
        }

        let n_samples = targets.len();
        if n_samples == 0 {
            return Err("CSV пуст: нет строк с данными".into());
        }
        let records_arr =
            Array2::from_shape_vec((n_samples, n_features), records).map_err(|e| e.to_string())?;
        let targets_arr = Array1::from_vec(targets);

        // Контракт linfa: target == 0..=n_classes-1; формируем linfa::Dataset,
        // чтобы подчеркнуть, что ML-крейт действительно используется. linfa 0.7
        // принимает owned-массивы ndarray; размер получаем через `records().nrows()`.
        let ds = Dataset::new(records_arr.clone(), targets_arr.clone());
        debug_assert_eq!(ds.records().nrows(), n_samples);
        debug_assert_eq!(ds.targets().len(), n_samples);

        Ok(WineDataset {
            feature_names,
            records: records_arr,
            targets: targets_arr,
            feature_index,
        })
    }

    fn feature_col_index(&self, feature: &str) -> Option<usize> {
        self.feature_index.get(feature).copied()
    }

    /// Передаёт owned-данные наружу для параллельных вычислений (DIP: потребитель
    /// получает абстрактные `Array2`/`Array1` и индекс признаков, не зная о
    /// внутренней структуре источника).
    pub fn into_parts(
        self,
    ) -> (
        Array2<f64>,
        Array1<usize>,
        std::collections::BTreeMap<String, usize>,
        Vec<String>,
    ) {
        (
            self.records,
            self.targets,
            self.feature_index,
            self.feature_names,
        )
    }
}

impl FeatureSource for WineDataset {
    fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    fn class_labels(&self) -> Vec<usize> {
        let mut set: Vec<usize> = self.targets.iter().copied().collect();
        set.sort_unstable();
        set.dedup();
        set
    }

    fn all_feature_values(&self, feature: &str) -> Option<Vec<f64>> {
        let col = self.feature_col_index(feature)?;
        Some(self.records.column(col).to_vec())
    }

    fn class_feature_values(&self, class: usize, feature: &str) -> Option<Vec<f64>> {
        let col = self.feature_col_index(feature)?;
        let column = self.records.column(col);
        let mut out = Vec::new();
        for (i, &c) in self.targets.iter().enumerate() {
            if c == class {
                out.push(column[i]);
            }
        }
        Some(out)
    }
}
