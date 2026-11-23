use plotters::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution as RandDistribution, Normal as RandNormal};
use rand_pcg::Pcg64;
use rayon::prelude::*;
use statrs::distribution::{Continuous, Normal as StatNormal};
use std::error::Error;

macro_rules! report {
    ($($arg:tt)*) => { println!($($arg)*); };
}

// ---------- ТРЕЙТЫ ----------

trait Dataset {
    fn values(&self) -> &[f64];
    fn len(&self) -> usize { self.values().len() }
}

trait Statistics {
    fn mean(&self) -> f64;
    fn std_pop(&self) -> f64;
    fn min(&self) -> f64;
    fn max(&self) -> f64;
    fn fraction_in_range(&self, lo: f64, hi: f64) -> f64;
}

trait Fit {
    type Model;
    fn fit(&self, data: &[f64]) -> Self::Model;
}

trait Density {
    fn pdf(&self, x: f64) -> f64;
}

// ---------- РЕАЛИЗАЦИИ ----------

struct BatteryDataset { hours: Vec<f64> }

impl Dataset for BatteryDataset {
    fn values(&self) -> &[f64] { &self.hours }
}

impl Statistics for BatteryDataset {
    fn mean(&self) -> f64 {
        self.hours.par_iter().sum::<f64>() / self.len() as f64
    }
    fn std_pop(&self) -> f64 {
        let m = self.mean();
        (self.hours.par_iter()
            .map(|x| (x - m).powi(2))
            .sum::<f64>() / self.len() as f64).sqrt()
    }
    // FIX 1: reduce(identity, op) — identity это Fn() -> Item
    fn min(&self) -> f64 {
        self.hours.par_iter().cloned().reduce(|| f64::INFINITY, f64::min)
    }
    fn max(&self) -> f64 {
        self.hours.par_iter().cloned().reduce(|| f64::NEG_INFINITY, f64::max)
    }
    fn fraction_in_range(&self, lo: f64, hi: f64) -> f64 {
        let c = self.hours.par_iter().filter(|&&x| x > lo && x < hi).count();
        c as f64 / self.len() as f64
    }
}

struct Gaussian { mu: f64, sigma: f64 }

struct GaussianFit;
impl Fit for GaussianFit {
    type Model = Gaussian;
    fn fit(&self, data: &[f64]) -> Gaussian {
        let n = data.len() as f64;
        let mu = data.par_iter().sum::<f64>() / n;
        let sigma = (data.par_iter()
            .map(|x| (x - mu).powi(2))
            .sum::<f64>() / n).sqrt();
        Gaussian { mu, sigma }
    }
}

impl Density for Gaussian {
    fn pdf(&self, x: f64) -> f64 {
        StatNormal::new(self.mu, self.sigma).unwrap().pdf(x)
    }
}

// ---------- ПЛОТТЕР ----------

struct HistogramPlot<'a, D: Dataset> {
    data: &'a D,
    bins: usize,
}

impl<'a, D: Dataset> HistogramPlot<'a, D> {
    fn new(data: &'a D, bins: usize) -> Self { Self { data, bins } }

    // FIX 3: `M: Density + Sync`, чтобы &M был Send+Sync для rayon
    fn render<M: Density + Sync>(&self, model: &M, worst: f64, path: &str)
        -> Result<(), Box<dyn Error>>
    {
        let lo = self.data.values().iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = self.data.values().iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let width = (hi - lo) / self.bins as f64;

        let mut counts = vec![0usize; self.bins];
        for &x in self.data.values() {
            let idx = (((x - lo) / width) as usize).min(self.bins - 1);
            counts[idx] += 1;
        }
        let n = self.data.len() as f64;
        let density: Vec<f64> = counts.iter()
            .map(|&c| c as f64 / (n * width))
            .collect();

        let xs: Vec<f64> = (0..100)
            .map(|i| lo + (hi - lo) * i as f64 / 99.0)
            .collect();
        let ys: Vec<f64> = xs.par_iter().map(|&x| model.pdf(x)).collect();

        let max_y = (density.iter().cloned()
            .fold(f64::NEG_INFINITY, f64::max))
            .max(ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max)) * 1.15;

        let root = BitMapBackend::new(path, (1000, 700)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Battery Life", ("sans-serif", 30))
            .margin(15)
            .x_label_area_size(45)
            .y_label_area_size(45)
            .build_cartesian_2d((lo - 0.5)..(hi + 0.5), 0.0..max_y)?;

        chart.configure_mesh()
            .x_desc("Hours")
            .y_desc("Density")
            .draw()?;

        chart.draw_series((0..self.bins).map(|i| {
            let x0 = lo + i as f64 * width;
            Rectangle::new(
                [(x0, 0.0), (x0 + width, density[i])],
                BLUE.mix(0.45).filled(),
            )
        }))?;

        chart.draw_series(LineSeries::new(
            xs.iter().cloned().zip(ys.iter().cloned()),
            RED.stroke_width(2),
        ))?.label("Fitted Gaussian")
           .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));

        chart.draw_series(std::iter::once(
            Circle::new((worst, 0.0), 6, RED.filled())
        ))?.label("Worst")
           .legend(|(x, y)| Circle::new((x + 10, y), 6, RED.filled()));

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }
}

// ---------- MAIN ----------

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = Pcg64::seed_from_u64(44);
    let normal = RandNormal::new(14.0f64, 1.5f64)?;

    // FIX 2: явный тип f64 для sample(), иначе `.round()` неоднозначен
    let hours: Vec<f64> = (0..300)
        .map(|_| {
            let v: f64 = normal.sample(&mut rng);
            (v * 10.0).round() / 10.0
        })
        .collect();

    let data = BatteryDataset { hours };

    let model = GaussianFit.fit(data.values());

    let mean   = data.mean();
    let std    = data.std_pop();
    let worst  = data.min();
    let within = data.fraction_in_range(mean - 2.0 * std, mean + 2.0 * std);

    report!("Worst battery: {:.1} hours, {:.2} std below the mean",
            worst, (mean - worst) / std);
    report!("Within 2 std: {:.1}% (the rule says about 95%)",
            within * 100.0);

    HistogramPlot::new(&data, 20)
        .render(&model, worst, "battery_life.png")?;

    report!("Plot saved to battery_life.png");
    Ok(())
}
