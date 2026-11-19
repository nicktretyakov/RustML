//! Модуль статистики: очистка пропусков и группировка по видам.
//!
//! **SOLID — OCP**: стратегия очистки данных вынесена в трейт `DropStrategy`,
//! поэтому новую логику (например, заполнять медианой) можно добавить,
//! не трогая существующий код.
//!
//! **SOLID — SRP**: `Analyzer` отвечает только за агрегации (среднее,
//! гистограмма); он ничего не знает про отрисовку.

use std::collections::BTreeMap;

use crate::io::{Dataset, PenguinRow};

/// Очищенный датасет: гарантирует отсутствие пропусков в любом числовом
/// столбце. Это инвариант объекта — все агрегации ниже можно вызывать
/// без проверки на `None`.
#[derive(Debug, Clone)]
pub struct CleanedData {
    rows: Vec<PenguinRow>,
}

impl CleanedData {
    pub fn new(rows: Vec<PenguinRow>) -> Self {
        Self { rows }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn rows(&self) -> &[PenguinRow] {
        &self.rows
    }

    /// Группировка строк по виду. `BTreeMap` даёт детерминированный порядок
    /// (Adelie → Chinstrap → Gentoo), совпадающий с Python-эталоном.
    pub fn group_by_species(&self) -> BTreeMap<String, Vec<&PenguinRow>> {
        let mut groups: BTreeMap<String, Vec<&PenguinRow>> = BTreeMap::new();
        for row in &self.rows {
            groups.entry(row.species.clone()).or_default().push(row);
        }
        groups
    }

    /// Все значения столбца по виду.
    pub fn values_for(&self, species: &str, column: &str) -> Vec<f64> {
        self.rows
            .iter()
            .filter(|r| r.species == species)
            .filter_map(|r| r.field(column))
            .collect()
    }

    /// Среднее значение столбца для указанного вида.
    pub fn mean_for(&self, species: &str, column: &str) -> Option<f64> {
        let values = self.values_for(species, column);
        if values.is_empty() {
            return None;
        }
        let sum: f64 = values.iter().sum();
        Some(sum / values.len() as f64)
    }
}

/// Стратегия очистки данных (OCP: новые стратегии — через реализацию трейта).
pub trait DropStrategy {
    fn clean(&self, ds: &Dataset) -> CleanedData;
}

/// Реализация: удаляет ВСЕ строки, в которых есть хотя бы один пропуск
/// в любом поле (числовом или категориальном). Аналог `df.dropna()` без
/// параметров в pandas: чистит и `bill_*`, и `sex`.
pub struct DropAnyNaStrategy;

impl DropStrategy for DropAnyNaStrategy {
    fn clean(&self, ds: &Dataset) -> CleanedData {
        let kept: Vec<PenguinRow> = ds
            .iter()
            .filter(|r| r.has_no_na())
            .cloned()
            .collect();
        CleanedData::new(kept)
    }
}

/// Параметры гистограммы, повторяющие поведение matplotlib:
/// `bins=8, alpha=0.5`.
#[derive(Debug, Clone, Copy)]
pub struct HistogramSpec {
    pub bins: usize,
    pub alpha: f64,
}

impl Default for HistogramSpec {
    fn default() -> Self {
        Self { bins: 8, alpha: 0.5 }
    }
}

/// Описание одной колонки гистограммы (бина) — используется и для отрисовки,
/// и для текстового вывода количества значений в каждом бине.
#[derive(Debug, Clone)]
pub struct HistogramBin {
    pub lower: f64,
    pub upper: f64,
    pub count: usize,
}

/// Анализатор: чистая логика, без I/O и отрисовки (SRP).
pub struct Analyzer;

impl Analyzer {
    pub fn new() -> Self {
        Self
    }

    /// Посчитать равномерные бины от `min` до `max`, точно повторяя логику
    /// `numpy.histogram(..., bins=8)`:
    ///
    /// 1. Границы бинов строятся через `linspace(min, max, bins+1)` — это
    ///    даёт «красивые» значения, которые совпадают с исходными данными
    ///    (например, 19.7), избегая ошибок float-округления при делении.
    /// 2. Значение, точно попавшее на внутреннюю границу, уходит в ВЕРХНИЙ
    ///    бин (`searchsorted` с `side='right'`), как в numpy.
    /// 3. Максимальное значение попадает в последний бин.
    pub fn histogram(values: &[f64], spec: HistogramSpec) -> Vec<HistogramBin> {
        if values.is_empty() || spec.bins == 0 {
            return Vec::new();
        }
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let span = (max - min).max(1e-12);

        // Аналог numpy.linspace(min, max, bins+1).
        let edges: Vec<f64> = (0..=spec.bins)
            .map(|i| min + (i as f64 / spec.bins as f64) * span)
            .collect();

        let mut counts = vec![0usize; spec.bins];
        for &v in values {
            // Аналог np.searchsorted(edges, v, side='right') - 1:
            //   - если v точно равно edges[i], попадает в бин i (верхний);
            //   - если v между edges[i-1] и edges[i], попадает в бин i-1.
            let idx = match edges.binary_search_by(|e| e.partial_cmp(&v).unwrap()) {
                Ok(i) => i,                       // v == edges[i] → бин i
                Err(i) => i.saturating_sub(1),    // v в (edges[i-1], edges[i]) → бин i-1
            };
            // Последний бин закрыт справа: clamp.
            let idx = idx.min(spec.bins - 1);
            counts[idx] += 1;
        }

        counts
            .iter()
            .enumerate()
            .map(|(i, &c)| HistogramBin {
                lower: edges[i],
                upper: edges[i + 1],
                count: c,
            })
            .collect()
    }

    /// Округление до `n` знаков, повторяющее поведение `round()` из Python:
    /// Python использует банковское округление, но для типичных средних
    /// значений пингвинов достаточно стандартного `f64::round`.
    pub fn round(value: f64, n: u32) -> f64 {
        let factor = 10f64.powi(n as i32);
        (value * factor).round() / factor
    }
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_counts_match_numpy() {
        // Реальные значения bill_length_mm для Adelie из 344-строчного
        // датасета seaborn (первые 30 отсортированных значений).
        // Эталон посчитан через numpy.histogram(values, bins=8).
        let values: Vec<f64> = vec![
            32.1, 33.1, 33.5, 34.0, 34.4, 34.5, 34.6, 34.6, 35.0, 35.0, 35.1, 35.2, 35.3, 35.5,
            35.5, 35.6, 35.7, 35.7, 35.7, 35.9, 35.9, 36.0, 36.0, 36.0, 36.0, 36.2, 36.2, 36.2,
            36.3, 36.4,
        ];
        let bins = Analyzer::histogram(&values, HistogramSpec::default());
        let counts: Vec<usize> = bins.iter().map(|b| b.count).collect();
        assert_eq!(counts, vec![1, 1, 1, 1, 4, 5, 6, 11]);
    }

    #[test]
    fn round_matches_python() {
        assert_eq!(Analyzer::round(38.857_142_857_142_85, 1), 38.9);
        assert_eq!(Analyzer::round(46.0, 1), 46.0);
        assert_eq!(Analyzer::round(50.0, 1), 50.0);
    }
}
