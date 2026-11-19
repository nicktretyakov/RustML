//! Оркестратор приложения.
//!
//! DIP: `App` зависит от абстракций `DataSource`, `StatsCalculator`, `Reporter`,
//!       а не от конкретных реализаций. Это позволяет подставлять любой источник,
//!       любой алгоритм и любой репортёр без изменения кода App (OCP).
//! SRP: App лишь координирует этапы; логика загрузки, вычисления и вывода вынесена.

use anyhow::{Context, Result};

use crate::dataset::{ColumnarDataset, CsvDataset, DataSource};
use crate::reporter::{BufferReporter, Reporter, linfa_integration_demo};
use crate::stats::StatsCalculator;

/// Декларативный макрос: проверка индекса в диапазоне 0..len.
/// Используется для валидации пользовательского ввода n.
#[macro_export]
macro_rules! assert_valid_index {
    ($idx:expr, $len:expr) => {
        if $idx >= $len {
            return Err(anyhow::anyhow!(
                "Индекс {} вне диапазона (0..{}). Введите число от 0 до {}.",
                $idx,
                $len,
                $len - 1
            ));
        }
    };
}

/// Обобщённое приложение, скомпонованное из абстракций (Dependency Injection).
pub struct App<D, C, R>
where
    D: DataSource,
    C: StatsCalculator,
    R: Reporter,
{
    dataset: D,
    calculator: C,
    reporter: R,
}

impl<D, C, R> App<D, C, R>
where
    D: DataSource,
    C: StatsCalculator,
    R: Reporter,
{
    /// Конструктор с инъекцией зависимостей.
    pub fn new(dataset: D, calculator: C, reporter: R) -> Self {
        Self {
            dataset,
            calculator,
            reporter,
        }
    }

    /// Основной сценарий: вывод метаинформации о датасете и статистики признака n.
    pub fn run(&self, n: usize) -> Result<()> {
        // 1. Вывод общего количества образцов и признаков.
        let n_samples = self.dataset.n_samples();
        let n_features = self.dataset.feature_names().len();
        self.reporter.report_samples(n_samples);
        self.reporter.report_features(n_features);

        // 2. Вывод названий классов.
        self.reporter.report_classes(self.dataset.class_names());

        // 3. Проверка индекса n.
        assert_valid_index!(n, n_features);

        // 4. Демонстрация использования ML-крейта linfa: построение linfa::Dataset
        //    и параллельная (rayon) статистика через calculator.
        let _linfa_status = linfa_integration_demo(&self.dataset)
            .context("Не удалось построить linfa::Dataset")?;

        // 5. Вычисление среднего значения признака n.
        let feature_name = &self.dataset.feature_names()[n];
        let mean = self.calculator
            .mean(&self.dataset, n)
            .with_context(|| format!("Не удалось вычислить среднее для признака '{feature_name}'"))?;

        // 6. Вывод названия признака и его среднего (округлённого до 2 знаков).
        self.reporter.report_feature_stat(feature_name, mean);

        Ok(())
    }
}

/// Специализированный конструктор для стандартной конфигурации из CSV.
/// Загружает row-major CSV, транспонирует в column-major для эффективного slice-доступа,
/// и использует параллельный калькулятор среднего.
#[allow(dead_code)]
pub fn build_default_app<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<App<ColumnarDataset, crate::stats::ParallelMeanCalculator, crate::reporter::ConsoleReporter>>
{
    let row = CsvDataset::from_path(path)?;
    let col = ColumnarDataset::from_row_major(&row);
    Ok(App::new(
        col,
        crate::stats::ParallelMeanCalculator,
        crate::reporter::ConsoleReporter::new(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::ColumnarDataset;
    use crate::reporter::BufferReporter;
    use crate::stats::SequentialMeanCalculator;
    use anyhow::Result;

    fn toy_dataset() -> ColumnarDataset {
        ColumnarDataset::new(
            vec!["mean radius".into(), "mean area".into()],
            vec!["malignant".into(), "benign".into()],
            vec![vec![10.0, 20.0, 30.0], vec![100.0, 200.0, 300.0]],
        )
    }

    #[test]
    fn app_runs_full_pipeline() -> Result<()> {
        let ds = toy_dataset();
        let app = App::new(ds, SequentialMeanCalculator, BufferReporter::new());
        app.run(1)?;
        // Доступ к приватному полю reporter через helper в родительском модуле.
        let lines = reporter_lines(&app);
        assert_eq!(lines[0], "Samples: 3");
        assert_eq!(lines[1], "Features: 2");
        assert_eq!(lines[2], "Classes: ['malignant' 'benign']");
        assert_eq!(lines[3], "mean area: 200");
        Ok(())
    }

    #[test]
    fn app_rejects_invalid_index() {
        let ds = toy_dataset();
        let app = App::new(ds, SequentialMeanCalculator, BufferReporter::new());
        let res = app.run(5);
        assert!(res.is_err());
    }
}

/// Помощник для тестов: возвращает строки, собранные BufferReporter.
/// Определён в родительском модуле, поэтому имеет доступ к приватному полю `reporter`.
#[allow(dead_code)]
fn reporter_lines<D: DataSource, C: StatsCalculator>(
    app: &App<D, C, BufferReporter>,
) -> Vec<String> {
    app.reporter.lines()
}
