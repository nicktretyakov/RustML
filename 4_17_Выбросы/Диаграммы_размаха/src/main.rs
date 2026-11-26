use anyhow::{anyhow, Result};
use csv::ReaderBuilder;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::Array2;
use plotters::data::Quartiles;
use plotters::prelude::*;
use rayon::prelude::*;
use std::io::{self, Write};
use std::num::ParseFloatError;
use std::sync::Arc;
use thiserror::Error;

// ------------------------------------------------------------------
// 1. Ошибки (крейт макросов thiserror)
// ------------------------------------------------------------------
#[derive(Error, Debug)]
pub enum ReadingError {
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Parse error: {0}")]
    ParseFloat(#[from] ParseFloatError),
    #[error("Column '{0}' not found")]
    ColumnNotFound(String),
    #[error("Plot error: {0}")]
    Plot(String),
    #[error("ML error: {0}")]
    Ml(String),
}

// ------------------------------------------------------------------
// 2. Репозиторий (чтение CSV + параллельный парсинг через rayon)
// ------------------------------------------------------------------
trait ReadingRepository: Send + Sync {
    fn get_minutes(&self, name: &str) -> std::result::Result<Vec<f64>, ReadingError>;
}

struct CsvReadingRepository {
    path: String,
}

impl CsvReadingRepository {
    fn new(path: &str) -> Self {
        Self { path: path.to_string() }
    }
}

impl ReadingRepository for CsvReadingRepository {
    fn get_minutes(&self, name: &str) -> std::result::Result<Vec<f64>, ReadingError> {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_path(&self.path)?;

        let headers = rdr.headers()?.clone();
        let col_idx = headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| ReadingError::ColumnNotFound(name.to_string()))?;

        let strings: std::result::Result<Vec<String>, csv::Error> = rdr
            .records()
            .map(|r| r.map(|rec| rec.get(col_idx).unwrap_or("").to_string()))
            .collect();
        let strings = strings?;

        // Многопоточный парсинг через rayon
        let minutes: std::result::Result<Vec<f64>, ParseFloatError> =
            strings.par_iter().map(|s| s.parse::<f64>()).collect();
        Ok(minutes?)
    }
}

// ------------------------------------------------------------------
// 3. Статистика (Quartiles из plotters работает с f32)
// ------------------------------------------------------------------
#[derive(Debug, Clone)]
struct ReadingStatistics {
    median: f64,
    q1: f64,
    q3: f64,
}

trait StatisticsCalculator: Send + Sync {
    fn calculate(&self, data: &[f64]) -> ReadingStatistics;
}

struct PlottersStatisticsCalculator;

impl StatisticsCalculator for PlottersStatisticsCalculator {
    fn calculate(&self, data: &[f64]) -> ReadingStatistics {
        let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        let quartiles = Quartiles::new(&data_f32);
        let vals = quartiles.values();
        ReadingStatistics {
            median: vals[2] as f64,
            q1: vals[1] as f64,
            q3: vals[3] as f64,
        }
    }
}

// ------------------------------------------------------------------
// 4. Отрисовка коробчатой диаграммы (вручную, без Boxplot)
// ------------------------------------------------------------------
trait BoxPlotRenderer: Send + Sync {
    fn render(&self, data: &[f64], title: &str, output: &str)
        -> std::result::Result<(), ReadingError>;
}

struct PlottersBoxPlotRenderer;

impl BoxPlotRenderer for PlottersBoxPlotRenderer {
    fn render(&self, data: &[f64], title: &str, output: &str)
        -> std::result::Result<(), ReadingError>
    {
        let root = BitMapBackend::new(output, (800, 600)).into_drawing_area();
        root.fill(&WHITE).map_err(|e| ReadingError::Plot(e.to_string()))?;

        let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        let quartiles = Quartiles::new(&data_f32);
        let v = quartiles.values(); // &[f32; 5]
        let lower_fence = v[0];
        let q1          = v[1];
        let median      = v[2];
        let q3          = v[3];
        let upper_fence = v[4];

        let y_min = data_f32.iter().cloned().fold(f32::INFINITY, f32::min);
        let y_max = data_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let y_pad = ((y_max - y_min) * 0.1).max(1.0);
        let y_lo = (lower_fence - y_pad).max(0.0f32);
        let y_hi = upper_fence + y_pad;

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(20)
            .y_label_area_size(60)
            .build_cartesian_2d(0.0f32..1.0f32, y_lo..y_hi)
            .map_err(|e| ReadingError::Plot(e.to_string()))?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_x_axis()
            .draw()
            .map_err(|e| ReadingError::Plot(e.to_string()))?;

        // Геометрия коробчатой диаграммы
        let x_lo  = 0.30f32;
        let x_hi  = 0.70f32;
        let x_mid = 0.50f32;
        let cap   = 0.10f32;

        // Усы
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(x_mid, lower_fence), (x_mid, q1)],
                BLACK.stroke_width(1),
            )))
            .map_err(|e| ReadingError::Plot(e.to_string()))?;
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(x_mid, q3), (x_mid, upper_fence)],
                BLACK.stroke_width(1),
            )))
            .map_err(|e| ReadingError::Plot(e.to_string()))?;

        // Крышки усов
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(x_mid - cap, lower_fence), (x_mid + cap, lower_fence)],
                BLACK.stroke_width(2),
            )))
            .map_err(|e| ReadingError::Plot(e.to_string()))?;
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(x_mid - cap, upper_fence), (x_mid + cap, upper_fence)],
                BLACK.stroke_width(2),
            )))
            .map_err(|e| ReadingError::Plot(e.to_string()))?;

        // Коробка (Q1..Q3)
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(x_lo, q1), (x_hi, q3)],
                BLACK.mix(0.25).filled().stroke_width(2),
            )))
            .map_err(|e| ReadingError::Plot(e.to_string()))?;

        // Медиана
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(x_lo, median), (x_hi, median)],
                RED.stroke_width(3),
            )))
            .map_err(|e| ReadingError::Plot(e.to_string()))?;

        root.present().map_err(|e| ReadingError::Plot(e.to_string()))?;
        Ok(())
    }
}

// ------------------------------------------------------------------
// 5. ML: линейная регрессия (linfa)
// ------------------------------------------------------------------
struct ReadingPredictor;

impl ReadingPredictor {
    fn predict_next_day(&self, data: &[f64]) -> std::result::Result<f64, ReadingError> {
        let n = data.len();
        if n < 2 {
            return Err(ReadingError::Ml("Not enough data".into()));
        }
        let x: Array2<f64> = Array2::from_shape_fn((n, 1), |(i, _)| (i + 1) as f64);
        let y: ndarray::Array1<f64> = ndarray::Array1::from(data.to_vec());

        let dataset = Dataset::new(x, y);
        let model = LinearRegression::default()
            .fit(&dataset)
            .map_err(|e| ReadingError::Ml(e.to_string()))?;

        let next_x: Array2<f64> =
            Array2::from_shape_vec((1, 1), vec![(n + 1) as f64])
                .map_err(|e| ReadingError::Ml(e.to_string()))?;
        let pred = model.predict(&next_x);
        Ok(pred[0])
    }
}

// ------------------------------------------------------------------
// 6. Анализатор (SOLID: SRP/DIP, зависимости — абстракции)
// ------------------------------------------------------------------
struct ReadingAnalyzer<R, S, P> {
    repo: Arc<R>,
    stats: Arc<S>,
    renderer: Arc<P>,
}

impl<R, S, P> ReadingAnalyzer<R, S, P>
where
    R: ReadingRepository + 'static,
    S: StatisticsCalculator + 'static,
    P: BoxPlotRenderer + 'static,
{
    fn new(repo: Arc<R>, stats: Arc<S>, renderer: Arc<P>) -> Self {
        Self { repo, stats, renderer }
    }

    fn analyze(&self, name: &str) -> std::result::Result<(), ReadingError> {
        let minutes = self.repo.get_minutes(name)?;

        // Параллельно: статистика и ML-прогноз
        let (stats, pred) = rayon::join(
            || self.stats.calculate(&minutes),
            || ReadingPredictor.predict_next_day(&minutes),
        );

        println!("Typical day: {:.2} minutes", stats.median);
        println!("Middle half: {:.2} to {:.2} minutes", stats.q1, stats.q3);

        let title = format!("Reading: {}", name);
        self.renderer.render(&minutes, &title, "boxplot.png")?;
        println!("Box plot saved to boxplot.png");

        match pred {
            Ok(p) => println!("Predicted next day: {:.2} minutes", p),
            Err(e) => eprintln!("ML prediction failed: {}", e),
        }
        Ok(())
    }
}

// ------------------------------------------------------------------
// 7. main
// ------------------------------------------------------------------
fn main() -> Result<()> {
    print!("Enter name: ");
    io::stdout().flush()?;
    let mut name = String::new();
    io::stdin().read_line(&mut name)?;
    let name = name.trim().to_string();

    let analyzer = ReadingAnalyzer::new(
        Arc::new(CsvReadingRepository::new("reading.csv")),
        Arc::new(PlottersStatisticsCalculator),
        Arc::new(PlottersBoxPlotRenderer),
    );

    analyzer.analyze(&name).map_err(|e| anyhow!(e))?;
    Ok(())
}
