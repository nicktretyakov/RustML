use anyhow::Result;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array2;
use plotters::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;
use std::sync::Arc;

// ============================================================================
// 1. МАКРОСЫ (Declarative Macros)
// ============================================================================

/// Макрос для удобного и стандартизированного вывода метрик в консоль
macro_rules! report_metric {
    ($label:expr, $val:expr) => {
        println!("{}: {}", $label, $val);
    };
    ($label:expr, $val:expr, round) => {
        println!("{}: {:.2}", $label, $val);
    };
}

/// Макрос для быстрого создания детекторов аномалий
macro_rules! create_detector {
    ($detector_type:ty $(, $arg:expr)*) => {
        Box::new(<$detector_type>::new($($arg),*))
    };
}

// ============================================================================
// 2. ООП И ПРИНЦИПЫ SOLID
// ============================================================================

/// Модель данных для парсинга CSV (Single Responsibility)
#[derive(Debug, Deserialize)]
struct HumidityRecord {
    humidity: f64,
}

/// Интерфейс источника данных (Dependency Inversion Principle)
trait DataReader {
    fn read_readings(&self) -> Result<Vec<f64>>;
}

/// Реализация чтения из CSV
struct CsvDataReader {
    file_path: String,
}

impl CsvDataReader {
    fn new(path: &str) -> Self {
        Self {
            file_path: path.to_string(),
        }
    }
}

impl DataReader for CsvDataReader {
    fn read_readings(&self) -> Result<Vec<f64>> {
        let mut rdr = csv::Reader::from_path(&self.file_path)?;
        let mut readings = Vec::new();
        for result in rdr.deserialize() {
            let record: HumidityRecord = result?;
            readings.push(record.humidity);
        }
        Ok(readings)
    }
}

/// Интерфейс стратегии поиска сбоев датчика (Open/Closed & Interface Segregation)
trait AnomalyDetector: Send + Sync {
    fn name(&self) -> &str;
    fn count_glitches(&self, data: &[f64]) -> usize;
}

/// Вспомогательная структура статистики с многопоточными вычислениями (Rayon)
struct StatsEngine;

impl StatsEngine {
    /// Линейная интерполяция процентиля (полный аналог numpy.percentile)
    fn percentile(sorted_data: &[f64], p: f64) -> f64 {
        let idx = (p / 100.0) * (sorted_data.len() - 1) as f64;
        let lower = idx.floor() as usize;
        let upper = idx.ceil() as usize;
        let weight = idx - lower as f64;
        sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
    }

    /// Параллельный расчет среднего значения
    fn mean(data: &[f64]) -> f64 {
        let sum: f64 = data.par_iter().sum();
        sum / data.len() as f64
    }

    /// Параллельный расчет выборочного стандартного отклонения (ddof=1, как в pandas.std())
    fn std_dev(data: &[f64], mean: f64) -> f64 {
        let variance: f64 = data
            .par_iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (data.len() - 1) as f64;
        variance.sqrt()
    }
}

/// Стратегия 1: Детектор на основе межквартильного размаха (IQR)
struct IqrDetector {
    low: f64,
    high: f64,
}

impl IqrDetector {
    fn new(data: &[f64]) -> Self {
        let mut sorted = data.to_vec();
        sorted.par_sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q1 = StatsEngine::percentile(&sorted, 25.0);
        let q3 = StatsEngine::percentile(&sorted, 75.0);
        let iqr = q3 - q1;
        Self {
            low: q1 - 1.5 * iqr,
            high: q3 + 1.5 * iqr,
        }
    }

    /// Ограничение значений границами IQR (аналог np.clip) в многопоточном режиме
    fn clip(&self, data: &[f64]) -> Vec<f64> {
        data.par_iter()
            .map(|&x| x.clamp(self.low, self.high))
            .collect()
    }
}

impl AnomalyDetector for IqrDetector {
    fn name(&self) -> &str {
        "IQR glitches"
    }

    fn count_glitches(&self, data: &[f64]) -> usize {
        data.par_iter()
            .filter(|&&x| x < self.low || x > self.high)
            .count()
    }
}

/// Стратегия 2: Детектор на основе Z-оценки
struct ZScoreDetector {
    threshold: f64,
}

impl ZScoreDetector {
    fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl AnomalyDetector for ZScoreDetector {
    fn name(&self) -> &str {
        "Z-score glitches"
    }

    fn count_glitches(&self, data: &[f64]) -> usize {
        let mean = StatsEngine::mean(data);
        let std = StatsEngine::std_dev(data, mean);
        data.par_iter()
            .filter(|&&x| ((x - mean) / std).abs() > self.threshold)
            .count()
    }
}

/// Стратегия 3: Машинное обучение (Кластеризация K-Means через крейт linfa)
struct MlKMeansDetector;

impl MlKMeansDetector {
    fn new() -> Self {
        Self
    }
}

impl AnomalyDetector for MlKMeansDetector {
    fn name(&self) -> &str {
        "ML K-Means anomalies"
    }

    fn count_glitches(&self, data: &[f64]) -> usize {
        // Преобразуем 1D вектор в 2D матрицу для linfa
        let records = Array2::from_shape_vec((data.len(), 1), data.to_vec()).unwrap();
        let dataset = DatasetBase::from(records);

        // Обучаем модель K-Means на 2 кластера (нормальные показания и выбросы)
        let model = KMeans::params(2)
            .max_n_iterations(100)
            .fit(&dataset)
            .expect("Ошибка обучения модели K-Means");

        let predictions = model.predict(&dataset);
        let count_0 = predictions.iter().filter(|&&c| c == 0).count();
        let count_1 = predictions.len() - count_0;

        // Меньший кластер считаем аномалиями
        count_0.min(count_1)
    }
}

// ============================================================================
// 3. ВИЗУАЛИЗАЦИЯ (Аналог Matplotlib Boxplot)
// ============================================================================

trait ChartPlotter {
    fn plot_comparison(&self, before: &[f64], after: &[f64], output_file: &str) -> Result<()>;
}

struct PlottersBoxPlot;

impl PlottersBoxPlot {
    /// Вспомогательный метод для расчета 5-числовой сводки (Quartiles принимает f32)
    fn five_number_summary(data: &[f64]) -> Quartiles {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted[0] as f32;
        let q1 = StatsEngine::percentile(&sorted, 25.0) as f32;
        let median = StatsEngine::percentile(&sorted, 50.0) as f32;
        let q3 = StatsEngine::percentile(&sorted, 75.0) as f32;
        let max = *sorted.last().unwrap() as f32;

        Quartiles::new(&[min, q1, median, q3, max])
    }
}

impl ChartPlotter for PlottersBoxPlot {
    fn plot_comparison(&self, before: &[f64], after: &[f64], output_file: &str) -> Result<()> {
        let root = BitMapBackend::new(output_file, (1000, 500)).into_drawing_area();
        root.fill(&WHITE)?;

        // Делим холст на 2 графика рядом друг с другом (аналог subplots(1, 2))
        let areas = root.split_evenly((1, 2));
        let datasets = [("Before", before), ("After", after)];

        // Приводим границы оси Y к f32, так как Boxplot в plotters требует f32
        let global_min = (before.iter().cloned().fold(f64::INFINITY, f64::min) - 5.0) as f32;
        let global_max = (before.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 5.0) as f32;

        for (idx, (title, data)) in datasets.iter().enumerate() {
            let mut chart = ChartBuilder::on(&areas[idx])
                .caption(*title, ("sans-serif", 25))
                .margin(20)
                .x_label_area_size(30)
                .y_label_area_size(50)
                .build_cartesian_2d(0..2, global_min..global_max)?;

            chart
                .configure_mesh()
                .y_desc("Humidity (%)")
                .disable_x_mesh()
                .x_labels(0)
                .draw()?;

            let quartiles = Self::five_number_summary(data);
            chart.draw_series(std::iter::once(Boxplot::new_vertical(1, &quartiles)))?;
        }

        root.present()?;
        Ok(())
    }
}

// ============================================================================
// 4. КООРДИНАТОР ПРИЛОЖЕНИЯ (Orchestrator)
// ============================================================================

struct HumidityAnalyzer {
    reader: Box<dyn DataReader>,
    plotter: Box<dyn ChartPlotter>,
}

impl HumidityAnalyzer {
    fn new(reader: Box<dyn DataReader>, plotter: Box<dyn ChartPlotter>) -> Self {
        Self { reader, plotter }
    }

    fn run(&self) -> Result<()> {
        // 1. Чтение данных
        let readings = Arc::new(self.reader.read_readings()?);

        // 2. Инициализация детекторов через макросы (Полиморфизм)
        let iqr_detector = IqrDetector::new(&readings);
        let detectors: Vec<Box<dyn AnomalyDetector>> = vec![
            create_detector!(IqrDetector, &readings),
            create_detector!(ZScoreDetector, 3.0),
            create_detector!(MlKMeansDetector),
        ];

        // 3. Многопоточный запуск всех детекторов параллельно
        let results: Vec<(&str, usize)> = detectors
            .par_iter()
            .map(|detector| (detector.name(), detector.count_glitches(&readings)))
            .collect();

        for (name, count) in results {
            report_metric!(name, count);
        }

        // 4. Ограничение значений (Clip) и расчет средних
        let capped = iqr_detector.clip(&readings);
        let mean_before = StatsEngine::mean(&readings);
        let mean_after = StatsEngine::mean(&capped);

        report_metric!("Mean before", mean_before, round);
        report_metric!("Mean after", mean_after, round);

        // 5. Отрисовка диаграмм размаха (Boxplot)
        self.plotter
            .plot_comparison(&readings, &capped, "humidity_boxplot.png")?;

        println!("\nГрафик успешно сохранен в файл: humidity_boxplot.png");
Ok(())
}
}
fn main() -> Result<()> {
// Внедрение зависимостей (Dependency Injection)
let reader = Box::new(CsvDataReader::new("humidity.csv"));
let plotter = Box::new(PlottersBoxPlot);
let app = HumidityAnalyzer::new(reader, plotter);
app.run()?;
Ok(())
}
