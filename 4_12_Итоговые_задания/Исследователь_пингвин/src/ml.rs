//! Модуль машинного обучения: KNN-классификатор видов пингвина.
//!
//! **SOLID — DIP + OCP**: трейт [`Classifier`] абстрагирует ML-модель,
//! `KnnSpeciesClassifier` — конкретная реализация на крейте `smartcore`.
//! Любая другая модель (логистическая регрессия, случайный лес) добавляется
//! новой реализацией трейта, не трогая `main`.
//!
//! **Машинное обучение**: признаки пингвина (4 числовых измерения) → вид.
//! Используется `KNNClassifier` с k=5 и евклидовым расстоянием.

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};

use crate::stats::CleanedData;

/// Универсальный классификатор (DIP): обучается на очищенных данных и
/// предсказывает целевую переменную.
pub trait Classifier {
    type Input;
    type Output;

    fn fit(&mut self, data: &CleanedData) -> anyhow::Result<()>;
    fn predict(&self, input: &Self::Input) -> anyhow::Result<Self::Output>;
    fn accuracy(&self, data: &CleanedData) -> anyhow::Result<f64>;
}

/// Метка вида, которая кодируется числом для KNN (smartcore принимает
/// целочисленные метки).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpeciesId(pub usize);

/// Сопоставление строкового вида ↔ числового идентификатора.
#[derive(Debug, Clone, Default)]
pub struct LabelEncoder {
    pub labels: Vec<String>,
}

impl LabelEncoder {
    pub fn fit(species: impl IntoIterator<Item = String>) -> Self {
        use std::collections::BTreeSet;
        let set: BTreeSet<String> = species.into_iter().collect();
        Self {
            labels: set.into_iter().collect(),
        }
    }

    pub fn encode(&self, species: &str) -> Option<SpeciesId> {
        self.labels.iter().position(|l| l == species).map(SpeciesId)
    }

    pub fn decode(&self, id: SpeciesId) -> Option<&str> {
        self.labels.get(id.0).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }
}

/// KNN-классификатор видов пингвина. Обёртка над `smartcore::KNNClassifier`,
/// скрывающая детали матричного API за объектно-ориентированным фасадом.
pub struct KnnSpeciesClassifier {
    k: usize,
    encoder: LabelEncoder,
    model: Option<KNNClassifier<f64, usize, DenseMatrix<f64>, Vec<usize>, Euclidian<f64>>>,
}

impl KnnSpeciesClassifier {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            encoder: LabelEncoder::default(),
            model: None,
        }
    }
}

impl Classifier for KnnSpeciesClassifier {
    type Input = [f64; 4];
    type Output = String;

    fn fit(&mut self, data: &CleanedData) -> anyhow::Result<()> {
        // Собираем матрицу признаков (N × 4) и вектор меток (N).
        let rows = data.rows();
        if rows.is_empty() {
            anyhow::bail!("cannot train on empty dataset");
        }

        self.encoder = LabelEncoder::fit(rows.iter().map(|r| r.species.clone()));

        let n = rows.len();
        let mut buf = Vec::with_capacity(n * 4);
        let mut y: Vec<usize> = Vec::with_capacity(n);
        for r in rows {
            let features = r.numeric_features().ok_or_else(|| {
                anyhow::anyhow!("row has missing numeric features after cleaning")
            })?;
            buf.extend_from_slice(&features);
            y.push(self.encoder.encode(&r.species).map(|s| s.0).unwrap_or(0));
        }

        let x: DenseMatrix<f64> = DenseMatrix::new(n, 4, buf, false);

        let params = KNNClassifierParameters::default()
            .with_k(self.k)
            .with_distance(Euclidian::<f64>::default());
        let model = KNNClassifier::fit(&x, &y, params)?;
        self.model = Some(model);
        Ok(())
    }

    fn predict(&self, input: &Self::Input) -> anyhow::Result<Self::Output> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model is not fitted yet"))?;
        let x: DenseMatrix<f64> = DenseMatrix::new(1, 4, input.to_vec(), false);
        let y: Vec<usize> = model.predict(&x)?;
        let id = SpeciesId(*y.first().unwrap_or(&0));
        Ok(self
            .encoder
            .decode(id)
            .unwrap_or("unknown")
            .to_string())
    }

    fn accuracy(&self, data: &CleanedData) -> anyhow::Result<f64> {
        let rows = data.rows();
        if rows.is_empty() {
            return Ok(0.0);
        }
        let mut correct = 0usize;
        for r in rows {
            let features = match r.numeric_features() {
                Some(f) => f,
                None => continue,
            };
            let pred = self.predict(&features)?;
            if pred == r.species {
                correct += 1;
            }
        }
        Ok(correct as f64 / rows.len() as f64)
    }
}

// `DenseMatrix` реализует `Array2<f64>` и `Vec<usize>` — `Array1<usize>`,
// что требуется параметрами `KNNClassifier`. Гарантируется типами smartcore.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{Dataset, PenguinRow};
    use crate::stats::DropStrategy;

    fn row(species: &str, b1: f64, b2: f64, f1: f64, mass: f64) -> PenguinRow {
        PenguinRow {
            species: species.into(),
            island: "T".into(),
            bill_length_mm: Some(b1),
            bill_depth_mm: Some(b2),
            flipper_length_mm: Some(f1),
            body_mass_g: Some(mass),
            sex: Some("male".into()),
            year: 2007,
        }
    }

    #[test]
    fn classifier_trains_and_predicts() {
        let rows = vec![
            row("Adelie", 39.0, 18.0, 181.0, 3750.0),
            row("Adelie", 39.5, 17.0, 182.0, 3800.0),
            row("Gentoo", 46.0, 14.0, 210.0, 4500.0),
            row("Gentoo", 48.0, 15.0, 215.0, 4600.0),
        ];
        let ds = Dataset::new(rows);
        let cleaned = crate::stats::DropAnyNaStrategy.clean(&ds);
        let mut clf = KnnSpeciesClassifier::new(2);
        clf.fit(&cleaned).expect("fit");
        let p = clf.predict(&[39.2, 18.0, 181.0, 3750.0]).expect("predict");
        assert_eq!(p, "Adelie");
    }

    #[test]
    fn label_encoder_roundtrip() {
        let enc = LabelEncoder::fit(
            ["Gentoo", "Adelie", "Chinstrap"]
                .iter()
                .map(|s| s.to_string()),
        );
        let id = enc.encode("Adelie").unwrap();
        assert_eq!(enc.decode(id), Some("Adelie"));
    }
}
