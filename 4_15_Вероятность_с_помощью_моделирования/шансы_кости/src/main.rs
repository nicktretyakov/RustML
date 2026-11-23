use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::io::{self, Write};

use smartcore::linalg::basic::arrays::Array;          // ← было Array2
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;

const VISITS: usize = 2400;
const SEED: u64 = 18;
const TOKENS: [f64; 11] = [12.0, 0.0, 0.0, 3.0, 0.0, 5.0, 0.0, 3.0, 0.0, 0.0, 12.0];

macro_rules! report_metric {
    ($label:expr, $value:expr, $prec:expr) => {{
        let d: i32 = $prec;
        let f = 10_f64.powi(d);
        let v = ($value * f).round() / f;
        println!("{} {:.prec$}", $label, v, prec = d as usize);
    }};
}

#[derive(Debug, Clone)]
struct Analysis {
    goal: u8,
    chance_doubles: f64,
    chance_high: f64,
    chance_high_after_doubles: f64,
    totals: Vec<u8>,
}

trait Simulation: Send + Sync {
    fn run(&self) -> (Vec<u8>, Vec<u8>);
}

trait Analyzer: Send + Sync {
    fn analyze(&self, first: &[u8], second: &[u8]) -> Analysis;
}

trait ExpectedTokens: Send + Sync {
    fn expected(&self, totals: &[u8]) -> f64;
}

trait Reporter {
    fn report(&self, a: &Analysis, expected: f64);
}

struct DiceSimulation {
    seed: u64,
    visits: usize,
}

impl Simulation for DiceSimulation {
    fn run(&self) -> (Vec<u8>, Vec<u8>) {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut first = Vec::with_capacity(self.visits);
        let mut second = Vec::with_capacity(self.visits);
        for _ in 0..self.visits {
            first.push(rng.gen_range(1u8..=6));
            second.push(rng.gen_range(1u8..=6));
        }
        (first, second)
    }
}

struct DiceAnalyzer {
    goal: u8,
}

impl Analyzer for DiceAnalyzer {
    fn analyze(&self, first: &[u8], second: &[u8]) -> Analysis {
        let (n, doubles, high, high_doubles) = first
            .par_iter()
            .zip(second.par_iter())
            .fold(
                || (0usize, 0usize, 0usize, 0usize),
                |(n, d, h, hd), (&a, &b)| {
                    let is_double = a == b;
                    let total = a + b;
                    let is_high = total >= self.goal;
                    (
                        n + 1,
                        d + is_double as usize,
                        h + is_high as usize,
                        hd + (is_high && is_double) as usize,
                    )
                },
            )
            .reduce(|| (0, 0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3));

        let totals: Vec<u8> = first.iter().zip(second).map(|(a, b)| a + b).collect();

        Analysis {
            goal: self.goal,
            chance_doubles: doubles as f64 / n as f64,
            chance_high: high as f64 / n as f64,
            chance_high_after_doubles: if doubles == 0 {
                0.0
            } else {
                high_doubles as f64 / doubles as f64
            },
            totals,
        }
    }
}

struct TokenExpector;

impl ExpectedTokens for TokenExpector {
    fn expected(&self, totals: &[u8]) -> f64 {
        let n = totals.len() as f64;
        (2u8..=12)
            .into_par_iter()
            .map(|s| {
                let count = totals.iter().filter(|&&t| t == s).count() as f64;
                TOKENS[(s - 2) as usize] * (count / n)
            })
            .sum()
    }
}

struct ConsoleReporter;

impl Reporter for ConsoleReporter {
    fn report(&self, a: &Analysis, expected: f64) {
        report_metric!("Chance of doubles:", a.chance_doubles, 3);
        report_metric!(&format!("Chance of {} or more:", a.goal), a.chance_high, 3);
        report_metric!(
            &format!("Chance of {} or more after doubles:", a.goal),
            a.chance_high_after_doubles,
            3
        );
        report_metric!("Expected tokens per visit:", expected, 2);
    }
}

fn ml_cross_check(first: &[u8], second: &[u8]) -> Option<f64> {
    let rows = first.len();
    let mut data: Vec<f64> = Vec::with_capacity(rows * 2);
    let mut labels: Vec<i32> = Vec::with_capacity(rows);
    for i in 0..rows {
        data.push(first[i] as f64);
        data.push(second[i] as f64);
        labels.push((first[i] == second[i]) as i32);
    }

    let x = DenseMatrix::new(rows, 2, data, false).unwrap();
    let model = LogisticRegression::fit(&x, &labels, Default::default()).ok()?;

    let coef = model.coefficients();
    let intercept = model.intercept();

    let mut sum_probs = 0.0;
    for i in 0..rows {
        let mut z = *intercept.get((0, 0));
        for j in 0..2 {
            z += *x.get((i, j)) * *coef.get((0, j));
        }
        let p = 1.0 / (1.0 + (-z).exp());
        sum_probs += p;
    }
    Some(sum_probs / rows as f64)
}

fn main() {
    print!("Enter goal sum: ");
    io::stdout().flush().unwrap();
    let mut buf = String::new();
    io::stdin().read_line(&mut buf).unwrap();
    let goal: u8 = buf.trim().parse().expect("integer expected");

    let sim: Box<dyn Simulation> = Box::new(DiceSimulation { seed: SEED, visits: VISITS });
    let (first, second) = sim.run();

    let analyzer: Box<dyn Analyzer> = Box::new(DiceAnalyzer { goal });
    let analysis = analyzer.analyze(&first, &second);

    let expector: Box<dyn ExpectedTokens> = Box::new(TokenExpector);
    let expected = expector.expected(&analysis.totals);

    let reporter: Box<dyn Reporter> = Box::new(ConsoleReporter);
    reporter.report(&analysis, expected);

    if let Some(ml_p) = ml_cross_check(&first, &second) {
        eprintln!("[ML cross-check] LogisticRegression P(double) = {:.3}", ml_p);
    }
}
