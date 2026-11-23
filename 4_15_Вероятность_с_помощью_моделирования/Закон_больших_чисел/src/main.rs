use anyhow::{anyhow, Result};
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};
use plotters::prelude::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use rayon::prelude::*;
use std::io::{self, Write};
use std::sync::Arc;

// ============ МАКРОС ============
macro_rules! log_stat {
    ($n:expr, $v:expr) => {
        println!("After {} boxes: {:.3}", $n, $v);
    };
}

// ============ ТРЕЙТЫ (SOLID: ISP + DIP) ============
pub trait ToySimulator: Send + Sync {
    fn simulate(&self, boxes: usize) -> Vec<bool>;
}

pub trait ShareCalculator: Send + Sync {
    fn cumulative_share(&self, toys: &[bool]) -> Vec<f64>;
}

pub trait Plotter: Send + Sync {
    fn plot(&self, shares: &[f64], threshold: f64, path: &str) -> Result<()>;
}

pub trait ConvergencePredictor: Send + Sync {
    fn predict(&self, shares: &[f64]) -> Result<f64>;
}

// ============ РЕАЛИЗАЦИИ (SOLID: SRP + OCP + LSP) ============
pub struct SeededSimulator {
    pub seed: u64,
    pub toy_prob: f64,
}

impl ToySimulator for SeededSimulator {
    fn simulate(&self, boxes: usize) -> Vec<bool> {
        let mut rng = Pcg64::seed_from_u64(self.seed);
        (0..boxes).map(|_| rng.gen::<f64>() < self.toy_prob).collect()
    }
}

pub struct CumulativeShareCalculator;

impl ShareCalculator for CumulativeShareCalculator {
    fn cumulative_share(&self, toys: &[bool]) -> Vec<f64> {
        let mut count = 0usize;
        toys.iter()
            .enumerate()
            .map(|(i, &t)| {
                if t {
                    count += 1;
                }
                count as f64 / (i + 1) as f64
            })
            .collect()
    }
}

pub struct PngPlotter;

impl Plotter for PngPlotter {
    fn plot(&self, shares: &[f64], threshold: f64, path: &str) -> Result<()> {
        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;
        let n = shares.len().max(1);
        let max_y = (shares.iter().cloned().fold(0.0f64, f64::max)).max(0.5) * 1.2;

        let mut chart = ChartBuilder::on(&root)
            .caption("Toys in Cereal Boxes", ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(1.0f64..n as f64, 0.0f64..max_y)?;

        chart
            .configure_mesh()
            .x_desc("Boxes opened")
            .y_desc("Share with a toy")
            .draw()?;

        chart
            .draw_series(LineSeries::new(
                (1..=n).map(|i| (i as f64, shares[i - 1])),
                &BLUE,
            ))?
            .label("share")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        // Чёрная пунктирная линия на уровне 0.2 (ручной dash-паттерн)
        let dash = 8.0;
        let seg: Vec<(f64, f64)> = (0..)
            .map(|k| 1.0 + k as f64 * (dash * 2.0))
            .take_while(|&x| x < n as f64)
            .map(|x| (x, (x + dash).min(n as f64)))
            .collect();
        chart
            .draw_series(seg.into_iter().map(|(x1, x2)| {
                PathElement::new(
                    vec![(x1, threshold), (x2, threshold)],
                    BLACK.stroke_width(2),
                )
            }))?
            .label("0.2")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.stroke_width(2)));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.85))
            .border_style(BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }
}

pub struct LinearConvergencePredictor {
    pub window_fractions: Vec<f64>,
}

impl ConvergencePredictor for LinearConvergencePredictor {
    fn predict(&self, shares: &[f64]) -> Result<f64> {
        let n = shares.len();
        if n < 20 {
            return Ok(0.2);
        }
        // Многопоточно: усредняем асимптотику по нескольким окнам
        let preds: Result<Vec<f64>, anyhow::Error> = self
            .window_fractions
            .par_iter()
            .map(|&frac| {
                let w = ((n as f64) * frac).max(20.0) as usize;
                fit_asymptote(&shares[..w.min(n)])
            })
            .collect();
        let preds = preds?;
        Ok(preds.iter().sum::<f64>() / preds.len() as f64)
    }
}

/// Обучаем линейную регрессию share ~ 1/n на «хвосте» и берём intercept (n → ∞).
fn fit_asymptote(shares: &[f64]) -> Result<f64> {
    let n = shares.len();
    let start = n / 2;
    let xs: Vec<f64> = (start..n).map(|i| 1.0 / (i + 1) as f64).collect();
    let ys: Vec<f64> = shares[start..].to_vec();

    let x = Array2::from_shape_vec((xs.len(), 1), xs)
        .map_err(|e| anyhow!("ndarray: {e}"))?;
    let y = Array1::from_vec(ys);
    let ds = Dataset::new(x, y);

    let model = LinearRegression::default()
        .fit(&ds)
        .map_err(|e| anyhow!("linfa fit: {e:?}"))?;

    let query = Array2::from_shape_vec((1, 1), vec![0.0f64])
        .map_err(|e| anyhow!("ndarray: {e}"))?;
    let pred = model.predict(&query);
    Ok(pred[0])
}

// ============ main (SOLID: DIP — всё через Arc<dyn Trait>) ============
fn main() -> Result<()> {
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let boxes: usize = input.trim().parse()?;

    let simulator: Arc<dyn ToySimulator> =
        Arc::new(SeededSimulator { seed: 25, toy_prob: 0.2 });
    let calc: Arc<dyn ShareCalculator> = Arc::new(CumulativeShareCalculator);
    let plotter: Arc<dyn Plotter> = Arc::new(PngPlotter);
    let predictor: Arc<dyn ConvergencePredictor> = Arc::new(LinearConvergencePredictor {
        window_fractions: vec![0.25, 0.5, 0.75, 1.0],
    });

    let toys = simulator.simulate(boxes);
    let shares = calc.cumulative_share(&toys);

    if boxes >= 10 {
        log_stat!(10, shares[9]);
    }
    if boxes > 0 {
        log_stat!(boxes, shares[boxes - 1]);
    }

    plotter.plot(&shares, 0.2, "plot.png")?;
    println!("Graph saved to plot.png");

    let predicted = predictor.predict(&shares)?;
    println!("ML-predicted asymptotic share: {:.3}", predicted);

    io::stdout().flush()?;
    Ok(())
}
