//! Многопоточный расчёт статистик по видам через `rayon`.
//!
//! **Многопоточность**: `rayon` параллельно обходит коллекцию видов и для
//! каждого считает среднее. На маленьком датасете выигрыш невелик, но
//! архитектура готова к масштабированию.
//!
//! **ООП**: `ParallelStats` — объект-значение с агрегированными результатами.

use rayon::prelude::*;

use crate::stats::CleanedData;

/// Результат расчёта для одного вида.
#[derive(Debug, Clone)]
pub struct SpeciesStat {
    pub species: String,
    pub mean: f64,
}

/// Агрегированные статистики по всем видам.
#[derive(Debug, Clone, Default)]
pub struct ParallelStats {
    pub per_species: Vec<SpeciesStat>,
}

impl ParallelStats {
    /// Параллельно посчитать средние значения `column` по каждому виду.
    /// `rayon::par_iter` автоматически распределяет работу по пулу потоков.
    pub fn means(data: &CleanedData, column: &str) -> Self {
        let groups = data.group_by_species();
        let per_species: Vec<SpeciesStat> = groups
            .par_iter()
            .filter_map(|(species, rows)| {
                let values: Vec<f64> = rows
                    .iter()
                    .filter_map(|r| r.field(column))
                    .collect();
                if values.is_empty() {
                    return None;
                }
                let sum: f64 = values.iter().sum();
                Some(SpeciesStat {
                    species: species.clone(),
                    mean: sum / values.len() as f64,
                })
            })
            .collect();
        Self { per_species }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{Dataset, PenguinRow};
    use crate::stats::{CleanedData, DropAnyNaStrategy, DropStrategy};

    fn row(species: &str, b1: f64) -> PenguinRow {
        PenguinRow {
            species: species.into(),
            island: "T".into(),
            bill_length_mm: Some(b1),
            bill_depth_mm: Some(18.0),
            flipper_length_mm: Some(181.0),
            body_mass_g: Some(3750.0),
            sex: Some("male".into()),
            year: 2007,
        }
    }

    #[test]
    fn parallel_means_match_sequential() {
        let ds = Dataset::new(vec![
            row("Adelie", 39.0),
            row("Adelie", 41.0),
            row("Gentoo", 46.0),
        ]);
        let cleaned: CleanedData = DropAnyNaStrategy.clean(&ds);
        let stats = ParallelStats::means(&cleaned, "bill_length_mm");
        let by_name: std::collections::HashMap<&str, f64> = stats
            .per_species
            .iter()
            .map(|s| (s.species.as_str(), s.mean))
            .collect();
        assert_eq!(by_name["Adelie"], 40.0);
        assert_eq!(by_name["Gentoo"], 46.0);
    }
}
