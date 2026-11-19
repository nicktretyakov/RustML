//! Реализация `StatsComputer` на параллельных вычислениях rayon.
//!
//! SRP: модуль отвечает только за расчёт трёх метрик.
//! DIP: зависит от трейта `StatsComputer`, разделяя источник данных
//! и вычислитель. Параллелизм rayon: подсчёт частот классов и средних
//! по классам выполняется через `par_iter`, что эффективно при больших выборках.
//!
//! Сборка агрегатов через `rayon::fold` + `reduce` гарантирует корректное
//! объединение частичных результатов без мьютексов и `atomic`.
//!
//! Потокобезопасность: вычислитель не хранит ссылку на `dyn FeatureSource`
//! (которая требовала бы `Sync`); он работает с owned-представлением данных —
//! матрицей признаков и вектором меток, полученными из источника один раз
//! при конструировании. Это соответствует принципу DIP: зависим от абстракции
//! данных, а не от конкретной реализации источника.

use rayon::prelude::*;

use crate::traits::{ClassCounts, StatsComputer};
use ndarray::{Array1, Array2};

/// Параллельный вычислитель. Хранит owned-представление данных.
pub struct ParallelStats {
    records: Array2<f64>,
    targets: Array1<usize>,
    feature_index: std::collections::BTreeMap<String, usize>,
}

impl ParallelStats {
    /// Строит вычислитель из матрицы `X`, меток `y` и индекса признаков.
    pub fn new(
        records: Array2<f64>,
        targets: Array1<usize>,
        feature_index: std::collections::BTreeMap<String, usize>,
    ) -> Self {
        Self {
            records,
            targets,
            feature_index,
        }
    }

    /// Среднее арифметическое среза `f64` с Kahan-компенсацией, чтобы
    /// совпадать с `numpy.mean` в пределах погрешности float64.
    fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0f64;
        let mut c = 0.0f64;
        for &v in values {
            let y = v - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum / values.len() as f64
    }

    fn feature_col_index(&self, feature: &str) -> Option<usize> {
        self.feature_index.get(feature).copied()
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

    fn class_count(&self, class: usize) -> usize {
        self.targets.iter().filter(|&&c| c == class).count()
    }
}

impl StatsComputer for ParallelStats {
    fn most_frequent_class(&self) -> Option<(usize, usize)> {
        let labels = self.class_labels();
        if labels.is_empty() {
            return None;
        }

        // Параллельно по классам: каждый поток считает свой набор классов,
        // затем результаты сливаются через `ClassCounts::merge`.
        let counts: ClassCounts = labels
            .par_iter()
            .fold(
                || ClassCounts::default(),
                |mut acc, &c| {
                    let n = self.class_count(c);
                    acc.0.insert(c, n);
                    acc
                },
            )
            .reduce(
                || ClassCounts::default(),
                |a, b| a.merge(b),
            );

        // Стабильный выбор: максимум по count, при равенстве — больший class_id
        // (так, чтобы при необходимости совпадало с value_counts в sklearn).
        counts
            .0
            .into_iter()
            .max_by(|(c1, n1), (c2, n2)| n1.cmp(n2).then(c1.cmp(c2)))
            .map(|(c, n)| (c, n))
    }

    fn class_with_highest_mean(&self, feature: &str) -> Option<(usize, f64)> {
        let labels = self.class_labels();
        if labels.is_empty() {
            return None;
        }

        // Параллельно по классам вычисляем средние значения признака.
        let means: Vec<(usize, f64)> = labels
            .par_iter()
            .filter_map(|&c| {
                let vals = self.class_feature_values(c, feature)?;
                if vals.is_empty() {
                    None
                } else {
                    Some((c, Self::mean(&vals)))
                }
            })
            .collect();

        // Максимальное среднее; при равенстве — класс с большим id.
        means.into_iter().max_by(|(_, m1), (_, m2)| {
            m1.partial_cmp(m2).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn count_above_overall_mean(&self, class: usize, feature: &str) -> Option<usize> {
        let all = self.all_feature_values(feature)?;
        if all.is_empty() {
            return Some(0);
        }
        let overall = Self::mean(&all);

        // Параллельно считаем количество значений класса выше порога.
        let class_vals = self.class_feature_values(class, feature)?;
        let n = class_vals.par_iter().filter(|&&v| v > overall).count();
        Some(n)
    }
}
