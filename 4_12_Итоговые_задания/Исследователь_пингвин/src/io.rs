//! Модуль ввода-вывода: чтение CSV и подсчёт пропусков.
//!
//! **SOLID — SRP + DIP**: за загрузку датасета отвечает трейт `DatasetLoader`
//! (абстракция), а `CsvDatasetLoader` — конкретная реализация, внедряемая
//! в `main`. Это позволяет в будущем подставить `ParquetDatasetLoader` и т.п.,
//! не трогая статистику и графики.
//!
//! **ООП**: `Dataset` инкапсулирует необработанные записи и столбцы.

use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;

use serde::Deserialize;

/// Одна строка CSV. Пропущенные значения представлены как `Option<f64>`.
/// Поля называются ровно как в `penguins.csv`.
#[derive(Debug, Clone, Deserialize)]
pub struct PenguinRow {
    pub species: String,
    pub island: String,
    pub bill_length_mm: Option<f64>,
    pub bill_depth_mm: Option<f64>,
    pub flipper_length_mm: Option<f64>,
    pub body_mass_g: Option<f64>,
    pub sex: Option<String>,
    pub year: i32,
}

impl PenguinRow {
    /// Достать числовое значение столбца по имени.
    /// Возвращает `None`, если столбец неизвестен или значение пропущено.
    pub fn field(&self, column: &str) -> Option<f64> {
        match column {
            "bill_length_mm" => self.bill_length_mm,
            "bill_depth_mm" => self.bill_depth_mm,
            "flipper_length_mm" => self.flipper_length_mm,
            "body_mass_g" => self.body_mass_g,
            _ => None,
        }
    }

    /// Признаки, на которых обучается ML-модель (см. [`crate::ml`]).
    /// Берём только числовые измерения пингвина.
    pub fn numeric_features(&self) -> Option<[f64; 4]> {
        Some([
            self.bill_length_mm?,
            self.bill_depth_mm?,
            self.flipper_length_mm?,
            self.body_mass_g? as f64,
        ])
    }

    /// Полный набор полей, которые должны быть непустыми, чтобы строка
    /// survived `df.dropna()` без параметров (Python): числовые измерения
    /// И категориальный пол. Совпадает с поведением pandas.
    pub fn has_no_na(&self) -> bool {
        self.bill_length_mm.is_some()
            && self.bill_depth_mm.is_some()
            && self.flipper_length_mm.is_some()
            && self.body_mass_g.is_some()
            && self.sex.is_some()
    }

    /// Категориальная метка вида (целевая переменная ML).
    pub fn label(&self) -> &str {
        &self.species
    }
}

/// Необработанный датасет — коллекция строк и список доступных числовых
/// столбцов. Инкапсулирует внутреннее представление от внешних модулей.
#[derive(Debug, Clone)]
pub struct Dataset {
    rows: Vec<PenguinRow>,
    numeric_columns: Vec<&'static str>,
}

impl Dataset {
    /// Создать датасет из строк. Числовые столбцы захардкожены так же,
    /// как они присутствуют в `penguins.csv`.
    pub fn new(rows: Vec<PenguinRow>) -> Self {
        Self {
            rows,
            numeric_columns: vec![
                "bill_length_mm",
                "bill_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
            ],
        }
    }

    /// Общее число пингвинов (строк) в датасете.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Итератор по строкам.
    pub fn iter(&self) -> impl Iterator<Item = &PenguinRow> {
        self.rows.iter()
    }

    /// Получить все значения числового столбца (включая пропуски как `None`).
    pub fn column(&self, name: &str) -> Vec<Option<f64>> {
        self.rows.iter().map(|r| r.field(name)).collect()
    }

    /// Число пропущенных значений в столбце.
    pub fn missing_count(&self, column: &str) -> usize {
        self.rows
            .iter()
            .filter(|r| r.field(column).is_none())
            .count()
    }

    /// Уникальные названия видов (в стабильном порядке).
    pub fn species(&self) -> Vec<String> {
        let set: BTreeSet<String> = self.rows.iter().map(|r| r.species.clone()).collect();
        set.into_iter().collect()
    }

    /// Доступные числовые столбцы.
    pub fn numeric_columns(&self) -> &[&'static str] {
        &self.numeric_columns
    }

    /// Проверить, что столбец существует, иначе вернуть осмысленную ошибку.
    pub fn require_column(&self, column: &str) -> anyhow::Result<()> {
        if self.numeric_columns.contains(&column) {
            Ok(())
        } else {
            anyhow::bail!(
                "unknown column `{}`; expected one of {:?}",
                column,
                self.numeric_columns
            )
        }
    }
}

/// Абстракция загрузчика датасета (DIP: зависим от абстракции, не от реализации).
pub trait DatasetLoader {
    fn load(&self) -> anyhow::Result<Dataset>;
}

/// Конкретная реализация: читает CSV через крейт `csv` + `serde`.
pub struct CsvDatasetLoader<'a> {
    path: &'a Path,
}

impl<'a> CsvDatasetLoader<'a> {
    pub fn new(path: &'a Path) -> Self {
        Self { path }
    }
}

impl<'a> DatasetLoader for CsvDatasetLoader<'a> {
    fn load(&self) -> anyhow::Result<Dataset> {
        let file = File::open(self.path)
            .map_err(|e| anyhow::anyhow!("cannot open {}: {e}", self.path.display()))?;
        let mut rdr = csv::ReaderBuilder::new().flexible(false).from_reader(file);

        let mut rows = Vec::new();
        for (i, record) in rdr.deserialize::<PenguinRow>().enumerate() {
            let row = record.map_err(|e| anyhow::anyhow!("CSV parse error at line {}: {e}", i + 2))?;
            rows.push(row);
        }
        if rows.is_empty() {
            anyhow::bail!("dataset `{}` is empty", self.path.display());
        }
        Ok(Dataset::new(rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_count_is_zero_for_full_column() {
        let rows = vec![
            PenguinRow {
                species: "Adelie".into(),
                island: "T".into(),
                bill_length_mm: Some(39.1),
                bill_depth_mm: Some(18.7),
                flipper_length_mm: Some(181.0),
                body_mass_g: Some(3750.0),
                sex: Some("male".into()),
                year: 2007,
            },
        ];
        let ds = Dataset::new(rows);
        assert_eq!(ds.missing_count("bill_length_mm"), 0);
    }

    #[test]
    fn unknown_column_errors() {
        let ds = Dataset::new(vec![]);
        assert!(ds.require_column("nope").is_err());
    }
}
