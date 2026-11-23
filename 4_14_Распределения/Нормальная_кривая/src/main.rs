use plotters::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg64;
use rayon::prelude::*;
use std::error::Error;

// ---------- Макрос для печати «обученной» модели ----------
macro_rules! print_fitted {
    ($mean:expr, $std:expr) => {
        println!("Fitted curve: mean {:.2} mm, std {:.2} mm", $mean, $std);
    };
}

// ---------- Абстракции (SOLID: I — Interface Segregation) ----------
trait DataSource: Send + Sync {
    fn generate(&self) -> Vec<f64>;
}

trait DescriptiveStats: Send + Sync {
    fn mean(&self) -> f64;
    fn std(&self) -> f64; // population std, как np.std (ddof=0)
}

trait ContinuousPdf: Send + Sync {
    fn pdf(&self, x: f64) -> f64;
}

trait Plotter {
    fn plot(&self, batch: &BoltBatch, model: &NormalDistribution, path: &str)
        -> Result<(), Box<dyn Error>>;
}

// ---------- Данные (модель предметной области) ----------
pub struct BoltBatch {
    pub lengths: Vec<f64>,
}

impl BoltBatch {
    fn min(&self) -> f64 {
        self.lengths.par_iter().cloned().reduce(|| f64::INFINITY, f64::min)
    }
    fn max(&self) -> f64 {
        self.lengths.par_iter().cloned().reduce(|| f64::NEG_INFINITY, f64::max)
    }
}

// ---------- S: единственная ответственность — статистика ----------
impl DescriptiveStats for BoltBatch {
    fn mean(&self) -> f64 {
        let sum: f64 = self.lengths.par_iter().sum();
        sum / self.lengths.len() as f64
    }
    fn std(&self) -> f64 {
        let m = self.mean();
        let var: f64 = self
            .lengths
            .par_iter()
            .map(|x| (x - m).powi(2))
            .sum::<f64>()
            / self.lengths.len() as f64;
        var.sqrt()
    }
}

// ---------- Генератор данных (замена rng.normal(50, 0.4, 240)) ----------
pub struct SeededNormalSource {
    pub seed: u64,
    pub n: usize,
    pub mean: f64,
    pub std: f64,
}

impl DataSource for SeededNormalSource {
    fn generate(&self) -> Vec<f64> {
        let mut rng = Pcg64::seed_from_u64(self.seed);
        let dist = Normal::new(self.mean, self.std).unwrap();
        (0..self.n)
            .map(|_| (dist.sample(&mut rng) * 100.0).round() / 100.0)
            .collect()
    }
}

// ---------- ML: MLE-фитинг нормального распределения ----------
pub struct NormalDistribution {
    pub mean: f64,
    pub std: f64,
}

impl ContinuousPdf for NormalDistribution {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std;
        (-0.5 * z * z).exp() / (self.std * (2.0 * std::f64::consts::PI).sqrt())
    }
}

/// Оценка максимального правдоподобия для N(μ, σ²):
/// μ̂ = mean, σ̂ = population std.
pub struct MleNormalFitter;

impl MleNormalFitter {
    pub fn fit(batch: &BoltBatch) -> NormalDistribution {
        NormalDistribution { mean: batch.mean(), std: batch.std() }
    }
}

// ---------- Рендерер (SOLID: O — открыт к расширению, закрыт к изменению) ----------
pub struct HistogramPlotter {
    pub bins: usize,
    pub curve_points: usize,
}

impl Plotter for HistogramPlotter {
    fn plot(&self, batch: &BoltBatch, model: &NormalDistribution, path: &str)
        -> Result<(), Box<dyn Error>>
    {
        let lo = batch.min();
        let hi = batch.max();
        let bin_w = (hi - lo) / self.bins as f64;

        // Гистограмма плотности
        let mut counts = vec![0usize; self.bins];
        for &x in &batch.lengths {
            let idx = (((x - lo) / bin_w) as usize).min(self.bins - 1);
            counts[idx] += 1;
        }
        let n = batch.lengths.len() as f64;
        let densities: Vec<f64> = counts.iter()
            .map(|&c| c as f64 / (n * bin_w))
            .collect();

        // Кривая: 100 точек от min до max
        let curve: Vec<(f64, f64)> = (0..self.curve_points)
            .map(|i| {
                let t = i as f64 / (self.curve_points - 1) as f64;
                let x = lo + (hi - lo) * t;
                (x, model.pdf(x))
            })
            .collect();

        let y_max = densities.iter().cloned().fold(0.0_f64, f64::max)
            .max(curve.iter().map(|(_, y)| *y).fold(0.0_f64, f64::max))
            * 1.1;

        let root = BitMapBackend::new(path, (900, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Bolt Lengths", ("sans-serif", 28))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(lo..hi, 0.0..y_max)?;

        chart.configure_mesh()
            .x_desc("Length (mm)")
            .y_desc("Density")
            .draw()?;

        chart.draw_series((0..self.bins).map(|i| {
            let x0 = lo + i as f64 * bin_w;
            let x1 = x0 + bin_w;
            Rectangle::new([(x0, 0.0), (x1, densities[i])], BLUE.mix(0.4).filled())
        }))?;

        // ← исправлено: без `&` перед RED.stroke_width(2)
        chart.draw_series(LineSeries::new(curve, RED.stroke_width(2)))?;

        root.present()?;
        Ok(())
    }
}

// ---------- Точка входа ----------
fn main() -> Result<(), Box<dyn Error>> {
    // Аналог rng.normal(50, 0.4, size=240) с seed=31
    let batch = BoltBatch {
        lengths: SeededNormalSource { seed: 31, n: 240, mean: 50.0, std: 0.4 }.generate(),
    };

    // «Обучение» модели (MLE)
    let model = MleNormalFitter::fit(&batch);

    // Печать через макрос
    print_fitted!(model.mean, model.std);

    // График
    let plotter = HistogramPlotter { bins: 15, curve_points: 100 };
    plotter.plot(&batch, &model, "bolts.png")?;

    println!("Saved: bolts.png");
    Ok(())
}
