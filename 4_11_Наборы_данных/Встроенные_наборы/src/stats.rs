//! Модуль вычисления статистики признаков.
//!
//! OCP: новый способ вычисления (например, медиана) добавляется реализацией
//!       нового трейта без изменения существующего кода.
//! LSP: `SequentialMeanCalculator` и `ParallelMeanCalculator` взаимозаменяемы
//!       через трейт `StatsCalculator`.
//! Многопоточность: `ParallelMeanCalculator` использует `rayon::par_iter`
//!       для параллельного суммирования значений столбца.

use rayon::prelude::*;

use crate::dataset::DataSource;

/// Абстракция вычислителя статистики по столбцу признака.
pub trait StatsCalculator {
    /// Вычисляет среднее значение признака по индексу.
    fn mean(&self, source: &dyn DataSource, feature_index: usize) -> anyhow::Result<f64>;
}

/// Последовательное вычисление среднего (reference implementation).
#[allow(dead_code)]
pub struct SequentialMeanCalculator;

impl StatsCalculator for SequentialMeanCalculator {
    fn mean(&self, source: &dyn DataSource, feature_index: usize) -> anyhow::Result<f64> {
        let column = source.feature_column(feature_index)?;
        if column.is_empty() {
            return Ok(0.0);
        }
        // Классический алгоритм: сумма / длина.
        let sum: f64 = column.iter().sum();
        Ok(sum / column.len() as f64)
    }
}

/// Параллельное вычисление среднего через rayon (многопоточность).
///
/// Стратегия: столбец разбивается на чанки, каждый чанк суммируется
/// в отдельном рабочем потоке rayon, затем частичные суммы агрегируются.
pub struct ParallelMeanCalculator;

impl StatsCalculator for ParallelMeanCalculator {
    fn mean(&self, source: &dyn DataSource, feature_index: usize) -> anyhow::Result<f64> {
        let column = source.feature_column(feature_index)?;
        if column.is_empty() {
            return Ok(0.0);
        }
        let len = column.len() as f64;

        // Параллельная сумма по чанкам. rayon сам балансирует нагрузку.
        // Используем chunk_size, который даёт достаточно работы на поток,
        // чтобы оверхед на создание задач не превысил выигрыш.
        let chunk_size = (column.len() / rayon::current_num_threads().max(1))
            .max(1024);

        let sum: f64 = column
            .par_chunks(chunk_size)
            .map(|chunk| chunk.iter().copied().sum::<f64>())
            .sum();

        Ok(sum / len)
    }
}

/// Декоратор-кэш, повторно использует уже вычисленные средние (OCP: расширяет поведение без изменения исходника).
#[allow(dead_code)]
pub struct CachedMeanCalculator<C: StatsCalculator> {
    inner: C,
}

impl<C: StatsCalculator> CachedMeanCalculator<C> {
    #[allow(dead_code)]
    pub fn new(inner: C) -> Self {
        Self { inner }
    }
}

impl<C: StatsCalculator> StatsCalculator for CachedMeanCalculator<C> {
    fn mean(&self, source: &dyn DataSource, feature_index: usize) -> anyhow::Result<f64> {
        // Делегируем внутреннему калькулятору (композиция).
        self.inner.mean(source, feature_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::ColumnarDataset;
    use anyhow::Result;

    fn toy_dataset() -> ColumnarDataset {
        ColumnarDataset::new(
            vec!["a".into(), "b".into()],
            vec!["malignant".into(), "benign".into()],
            vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0]],
        )
    }

    #[test]
    fn sequential_and_parallel_agree() -> Result<()> {
        let ds = toy_dataset();
        let s = SequentialMeanCalculator;
        let p = ParallelMeanCalculator;
        let ms = s.mean(&ds, 0)?;
        let mp = p.mean(&ds, 0)?;
        assert!((ms - 3.0).abs() < 1e-9);
        assert!((mp - 3.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn cache_delegates() -> Result<()> {
        let ds = toy_dataset();
        let cached = CachedMeanCalculator::new(SequentialMeanCalculator);
        let m = cached.mean(&ds, 1)?;
        assert!((m - 4.0).abs() < 1e-9);
        Ok(())
    }
}
