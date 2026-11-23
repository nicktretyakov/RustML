use ndarray::Array2;
use linfa::traits::Fit;
use linfa_clustering::KMeans;
use plotters::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::error::Error;

// ============================================================================
// Трейты (Interface Segregation + Dependency Inversion)
// ============================================================================

/// Абстрактный источник весов — легко добавить город, село, страну и т.д.
pub trait WeightSource: Send + Sync {
    fn label(&self) -> &str;
    fn sample(&self) -> Vec<f64>;
}

// ============================================================================
// Реализация (Single Responsibility)
// ============================================================================

pub struct GaussianSource {
    label: String,
    mean: f64,
    sigma: f64,
    n: usize,
    seed: u64,
}

impl GaussianSource {
    pub fn new(label: impl Into<String>, mean: f64, sigma: f64, n: usize, seed: u64) -> Self {
        Self { label: label.into(), mean, sigma, n, seed }
    }
}

impl WeightSource for GaussianSource {
    fn label(&self) -> &str { &self.label }
    fn sample(&self) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let dist = Normal::new(self.mean, self.sigma).expect("valid params");
        (0..self.n)
            .map(|_| (dist.sample(&mut rng) * 100.0).round() / 100.0)
            .collect()
    }
}

// ============================================================================
// Данные для отрисовки
// ============================================================================

pub struct HistogramSeries {
    pub min: f64,
    pub width: f64,
    pub bins: usize,
    pub densities: Vec<f64>,
    pub label: String,
    pub color: RGBColor,
    pub alpha: f64,
}

pub struct DensityHistogramPlot {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub series: Vec<HistogramSeries>,
}

// ============================================================================
// Макрос: гистограмма плотности (DRY + параллельный подсчёт через rayon)
// ============================================================================

/// Возвращает (min, width, bins, densities), где
/// densities[i] = count_i / (N * width)
macro_rules! density_histogram {
    ($values:expr, $bins:expr) => {{
        let values: &[f64] = $values;
        let bins: usize = $bins;
        assert!(!values.is_empty(), "empty data");
        let n = values.len() as f64;

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut width = (max - min) / bins as f64;
        if width <= 0.0 { width = 1.0; }

        // Параллельный подсчёт по бинам
        let counts: Vec<usize> = (0..bins)
            .into_par_iter()
            .map(|i| {
                let lo = min + width * i as f64;
                let hi = lo + width;
                let is_last = i == bins - 1;
                values
                    .par_iter()
                    .filter(|&&v| v >= lo && (v < hi || (is_last && v <= hi)))
                    .count()
            })
            .collect();

        let densities: Vec<f64> =
            counts.iter().map(|&c| c as f64 / (n * width)).collect();

        (min, width, bins, densities)
    }};
}

// ============================================================================
// Machine Learning: MLE-оценка параметров нормального распределения
// ============================================================================

fn fit_gaussian_mle(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

// ============================================================================
// Рендеринг (единственная ответственность — отрисовка)
// ============================================================================

fn render_plot(plot: &DensityHistogramPlot, path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_min = plot.series.iter().map(|s| s.min).fold(f64::INFINITY, f64::min);
    let x_max = plot.series.iter()
        .map(|s| s.min + s.width * s.bins as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let y_max = plot.series.iter()
        .flat_map(|s| s.densities.iter().cloned())
        .fold(0.0_f64, f64::max) * 1.15;

    let mut chart = ChartBuilder::on(&root)
        .caption(&plot.title, ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min..x_max, 0.0f64..y_max)?;

    chart.configure_mesh()
        .x_desc(&plot.x_label)
        .y_desc(&plot.y_label)
        .draw()?;

    for s in &plot.series {
        let color = s.color;
        let alpha = s.alpha;
        let label = s.label.clone();

        chart
            .draw_series(s.densities.iter().enumerate().map(|(i, &d)| {
                let x0 = s.min + s.width * i as f64;
                let x1 = x0 + s.width;
                Rectangle::new([(x0, 0.0), (x1, d)], color.mix(alpha).filled())
            }))?
            .label(label)
            .legend(move |(x, y)| {
                Rectangle::new([(x, y - 5), (x + 15, y + 5)], color.mix(alpha).filled())
            });
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

// ============================================================================
// main
// ============================================================================

fn main() -> Result<(), Box<dyn Error>> {
    // ---------- Генерация данных (Open/Closed: новые источники — без правки main) ----------
    let city = GaussianSource::new("city", 3.4, 0.5, 400, 23);
    let rural = GaussianSource::new("rural", 3.2, 0.45, 60, 24);

    let city_data = city.sample();
    let rural_data = rural.sample();

    // ---------- ML #1: MLE-оценка параметров нормального распределения ----------
    let (cm, cs) = fit_gaussian_mle(&city_data);
    let (rm, rs) = fit_gaussian_mle(&rural_data);
    println!("[ML] city  ≈ N({:.3}, {:.3})", cm, cs);
    println!("[ML] rural ≈ N({:.3}, {:.3})", rm, rs);

    // ---------- ML #2: KMeans (unsupervised) на объединённых данных ----------
    {
        use linfa::DatasetBase;
        let all: Vec<f64> = city_data.iter().chain(rural_data.iter()).cloned().collect();
        let arr = Array2::from_shape_vec((all.len(), 1), all)?;
        let ds = DatasetBase::from(arr);
        let model = KMeans::params(2).max_n_iterations(100).fit(&ds)?;
        println!("[ML] KMeans центроиды: {:?}", model.centroids());
    }

    // ---------- Гистограммы плотности (макрос + rayon) ----------
    let (c_min, c_w, c_bins, c_dens) = density_histogram!(&city_data, 10);
    let (r_min, r_w, r_bins, r_dens) = density_histogram!(&rural_data, 10);

    let plot = DensityHistogramPlot {
        title: "Birth Weights".into(),
        x_label: "Weight (kg)".into(),
        y_label: "Density".into(),
        series: vec![
            HistogramSeries {
                min: c_min, width: c_w, bins: c_bins, densities: c_dens,
                label: "city".into(), color: BLUE, alpha: 0.5,
            },
            HistogramSeries {
                min: r_min, width: r_w, bins: r_bins, densities: r_dens,
                label: "rural".into(), color: RED, alpha: 0.5,
            },
        ],
    };

    render_plot(&plot, "birth_weights.png")?;
    println!("Сохранено: birth_weights.png");
    Ok(())
}
