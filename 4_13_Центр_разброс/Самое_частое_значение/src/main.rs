//! Rentals analysis in Rust.
//!
//! Reproduces the pandas reference output
//!     item: <mode> (<count> of <total>)
//!     boot_size: ...
//!     level: ...
//!     days: ...
//!     Days rented, mean vs mode: <mean1dp> vs <mode>
//! while demonstrating: declarative macros, external crates (csv/serde/rayon/
//! anyhow), multithreading (rayon), OOP through traits & structs, SOLID
//! design, and machine learning (Ordinary-Least-Squares linear regression and
//! a categorical Naive Bayes classifier).

use std::collections::HashMap;
use std::hash::Hash;
use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;
use serde::Deserialize;

// =========================================================================
// 1. MACROS
// =========================================================================

/// Print one column's mode line in the same format as the Python reference.
macro_rules! stat_line {
    ($name:expr, $mode:expr, $count:expr, $total:expr) => {
        println!("{}: {} ({} of {})", $name, $mode, $count, $total);
    };
}

/// Register a list of analyzers as boxed trait objects.
macro_rules! analyzers {
    ( $( $a:expr ),* $(,)? ) => {
        vec![ $( Box::new($a) as Box<dyn ColumnAnalyzer> ),* ]
    };
}

/// Print a labeled metric line (used by the ML report).
macro_rules! labeled {
    ($label:expr, $value:expr) => {
        println!("  {}: {}", $label, $value);
    };
}

// =========================================================================
// 2. DATA MODEL  (SRP: a Rental is just data + deserialization)
// =========================================================================

#[derive(Debug, Clone, Deserialize)]
struct Rental {
    item: String,
    boot_size: u32,
    level: String,
    days: u32,
}

fn load_rentals<P: AsRef<Path>>(path: P) -> Result<Vec<Rental>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut out = Vec::new();
    for record in rdr.deserialize() {
        out.push(record?);
    }
    Ok(out)
}

// =========================================================================
// 3. RESULT TYPE
// =========================================================================

#[derive(Debug)]
struct ColumnReport {
    name: String,
    mode: String,
    count: usize,
}

// =========================================================================
// 4. OOP / SOLID — TRAITS
//   * ISP : small, focused traits (ColumnAnalyzer / SupervisedModel /
//           Classifier) instead of one fat interface.
//   * OCP : new analyzers or models can be added without modifying the
//           engine — just implement the trait.
//   * LSP : any implementor is substitutable inside the engine.
//   * DIP : the engine depends on `Box<dyn ColumnAnalyzer>`, an abstraction,
//           not on concrete analyzer types.
//   * SRP : each concrete analyzer / model has exactly one responsibility.
// =========================================================================

/// Analyzes one column of the rental table.
trait ColumnAnalyzer: Send + Sync {
    fn name(&self) -> &'static str;
    fn analyze(&self, rentals: &[Rental]) -> ColumnReport;
}

/// Generic mode + frequency. The smallest most-frequent value is returned,
/// matching `pandas.Series.mode()[0]` (which sorts modes ascending and keeps
/// the first).
fn mode_count<T>(vals: &[T]) -> Option<(T, usize)>
where
    T: Eq + Hash + Ord + Clone,
{
    let mut counts: HashMap<T, usize> = HashMap::new();
    for v in vals {
        *counts.entry(v.clone()).or_insert(0) += 1;
    }
    let max_n = *counts.values().max()?;
    let mode = counts
        .into_iter()
        .filter(|(_, n)| *n == max_n)
        .map(|(v, _)| v)
        .min()?;
    Some((mode, max_n))
}

// --- concrete analyzers (one responsibility each) ------------------------

struct ItemAnalyzer;
struct BootSizeAnalyzer;
struct LevelAnalyzer;
struct DaysAnalyzer;

impl ColumnAnalyzer for ItemAnalyzer {
    fn name(&self) -> &'static str {
        "item"
    }
    fn analyze(&self, rentals: &[Rental]) -> ColumnReport {
        let vals: Vec<&str> = rentals.iter().map(|r| r.item.as_str()).collect();
        let (m, c) = mode_count(&vals).expect("non-empty rentals");
        ColumnReport {
            name: self.name().into(),
            mode: m.into(),
            count: c,
        }
    }
}

impl ColumnAnalyzer for BootSizeAnalyzer {
    fn name(&self) -> &'static str {
        "boot_size"
    }
    fn analyze(&self, rentals: &[Rental]) -> ColumnReport {
        let vals: Vec<u32> = rentals.iter().map(|r| r.boot_size).collect();
        let (m, c) = mode_count(&vals).expect("non-empty rentals");
        ColumnReport {
            name: self.name().into(),
            mode: m.to_string(),
            count: c,
        }
    }
}

impl ColumnAnalyzer for LevelAnalyzer {
    fn name(&self) -> &'static str {
        "level"
    }
    fn analyze(&self, rentals: &[Rental]) -> ColumnReport {
        let vals: Vec<&str> = rentals.iter().map(|r| r.level.as_str()).collect();
        let (m, c) = mode_count(&vals).expect("non-empty rentals");
        ColumnReport {
            name: self.name().into(),
            mode: m.into(),
            count: c,
        }
    }
}

impl ColumnAnalyzer for DaysAnalyzer {
    fn name(&self) -> &'static str {
        "days"
    }
    fn analyze(&self, rentals: &[Rental]) -> ColumnReport {
        let vals: Vec<u32> = rentals.iter().map(|r| r.days).collect();
        let (m, c) = mode_count(&vals).expect("non-empty rentals");
        ColumnReport {
            name: self.name().into(),
            mode: m.to_string(),
            count: c,
        }
    }
}

// =========================================================================
// 5. ENGINE  (DIP + multithreading)
// =========================================================================

/// Holds the analyzers as trait objects and runs them in parallel.
struct AnalyzerEngine {
    analyzers: Vec<Box<dyn ColumnAnalyzer>>,
}

impl AnalyzerEngine {
    fn new() -> Self {
        Self {
            analyzers: analyzers![
                ItemAnalyzer,
                BootSizeAnalyzer,
                LevelAnalyzer,
                DaysAnalyzer
            ],
        }
    }

    /// Run every analyzer in parallel across rayon's thread pool.
    fn run(&self, rentals: &[Rental]) -> Vec<ColumnReport> {
        self.analyzers
            .par_iter() // <-- multithreading
            .map(|a| a.analyze(rentals))
            .collect()
    }
}

// =========================================================================
// 6. MACHINE LEARNING
//   * SupervisedModel  -> regression abstraction
//   * Classifier       -> classification abstraction
//   Both are `Send + Sync` so inference can run on the rayon pool.
// =========================================================================

trait SupervisedModel: Send + Sync {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]);
    fn predict(&self, x: &[f64]) -> f64;
    fn name(&self) -> &'static str;
}

trait Classifier: Send + Sync {
    fn fit(&mut self, x: &[Vec<String>], y: &[String]);
    fn predict(&self, x: &[String]) -> String;
    fn name(&self) -> &'static str;
}

// --- Ordinary Least Squares linear regression (single feature) -----------
// Closed form:  slope = cov(x,y)/var(x);  intercept = mean(y) - slope*mean(x)
struct LinearRegression {
    slope: f64,
    intercept: f64,
}

impl LinearRegression {
    fn new() -> Self {
        Self {
            slope: 0.0,
            intercept: 0.0,
        }
    }
}

impl SupervisedModel for LinearRegression {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let n = x.len() as f64;
        if n == 0.0 {
            return;
        }
        let xs: Vec<f64> = x.iter().map(|row| row[0]).collect();
        let mx = xs.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let (num, den) = xs
            .iter()
            .zip(y.iter())
            .fold((0.0_f64, 0.0_f64), |(num, den), (&xi, &yi)| {
                (num + (xi - mx) * (yi - my), den + (xi - mx).powi(2))
            });
        self.slope = if den == 0.0 { 0.0 } else { num / den };
        self.intercept = my - self.slope * mx;
    }

    fn predict(&self, x: &[f64]) -> f64 {
        self.intercept + self.slope * x[0]
    }

    fn name(&self) -> &'static str {
        "LinearRegression (OLS)"
    }
}

// --- Categorical Naive Bayes with Laplace smoothing ----------------------
struct CategoricalNaiveBayes {
    classes: Vec<String>,
    class_prior: HashMap<String, f64>,
    class_total: HashMap<String, usize>,
    cond_counts: HashMap<(usize, String, String), usize>, // (feat, value, class)
    feat_cardinality: Vec<usize>,
    n_features: usize,
}

impl CategoricalNaiveBayes {
    fn new() -> Self {
        Self {
            classes: Vec::new(),
            class_prior: HashMap::new(),
            class_total: HashMap::new(),
            cond_counts: HashMap::new(),
            feat_cardinality: Vec::new(),
            n_features: 0,
        }
    }
}

impl Classifier for CategoricalNaiveBayes {
    fn fit(&mut self, x: &[Vec<String>], y: &[String]) {
        self.n_features = x.first().map(|r| r.len()).unwrap_or(0);
        let n = y.len();
        if n == 0 {
            return;
        }
        let mut cc: HashMap<String, usize> = HashMap::new();
        for c in y {
            *cc.entry(c.clone()).or_insert(0) += 1;
        }
        self.classes = cc.keys().cloned().collect();
        self.class_total = cc.clone();
        self.class_prior = cc
            .iter()
            .map(|(c, k)| (c.clone(), *k as f64 / n as f64))
            .collect();

        let mut feat_values: Vec<HashMap<String, ()>> =
            vec![HashMap::new(); self.n_features];
        for (row, c) in x.iter().zip(y.iter()) {
            for (i, v) in row.iter().enumerate() {
                *self
                    .cond_counts
                    .entry((i, v.clone(), c.clone()))
                    .or_insert(0) += 1;
                feat_values[i].insert(v.clone(), ());
            }
        }
        self.feat_cardinality = feat_values.iter().map(|m| m.len()).collect();
    }

    fn predict(&self, x: &[String]) -> String {
        let mut best: Option<(String, f64)> = None;
        for class in &self.classes {
            let mut logp = self.class_prior.get(class).copied().unwrap_or(0.0).ln();
            let cn = *self.class_total.get(class).unwrap_or(&0) as f64;
            for (i, v) in x.iter().enumerate() {
                let cnt = *self
                    .cond_counts
                    .get(&(i, v.clone(), class.clone()))
                    .unwrap_or(&0) as f64;
                let d = self.feat_cardinality.get(i).copied().unwrap_or(0) as f64;
                let p = (cnt + 1.0) / (cn + d); // Laplace smoothing
                logp += p.ln();
            }
            match &best {
                None => best = Some((class.clone(), logp)),
                Some((_, bp)) if logp > *bp => best = Some((class.clone(), logp)),
                _ => {}
            }
        }
        best.map(|(c, _)| c).unwrap_or_default()
    }

    fn name(&self) -> &'static str {
        "CategoricalNaiveBayes"
    }
}

// =========================================================================
// 7. MAIN
// =========================================================================

fn main() -> Result<()> {
    let rentals = load_rentals("rentals.csv")?;
    let total = rentals.len();

    // --- core analysis: modes per column, computed in parallel ------------
    let reports = AnalyzerEngine::new().run(&rentals);
    for r in &reports {
        stat_line!(r.name, r.mode, r.count, total);
    }

    // --- days mean vs mode ------------------------------------------------
    let days: Vec<f64> = rentals.iter().map(|r| r.days as f64).collect();
    let mean = days.iter().sum::<f64>() / days.len() as f64;
    let day_vals: Vec<u32> = rentals.iter().map(|r| r.days).collect();
    let (day_mode, _) = mode_count(&day_vals).expect("non-empty rentals");
    // {:.1} rounds half-to-even; for 95/30 = 3.1666... this is 3.2,
    // identical to Python's round(mean, 1).
    println!("Days rented, mean vs mode: {:.1} vs {}", mean, day_mode);

    // --- machine learning section -----------------------------------------
    println!();
    println!("--- machine learning ---");

    // Regression: predict `days` from `boot_size`.
    let x_reg: Vec<Vec<f64>> = rentals.iter().map(|r| vec![r.boot_size as f64]).collect();
    let y_reg: Vec<f64> = days.clone();
    let mut reg = LinearRegression::new();
    reg.fit(&x_reg, &y_reg);
    let pred42 = reg.predict(&[42.0]);
    let model: &dyn SupervisedModel = &reg; // DIP: use the abstraction
    println!("{} (predict days from boot_size)", model.name());
    labeled!("model", format!("days ~ {:.3}*boot_size + {:.3}", reg.slope, reg.intercept));
    labeled!("predicted days @ boot_size 42", format!("{:.2}", pred42));
    let mse: f64 = x_reg
        .iter()
        .zip(&y_reg)
        .map(|(xi, yi)| {
            let p = reg.predict(xi);
            (p - yi).powi(2)
        })
        .sum::<f64>()
        / y_reg.len() as f64;
    labeled!("train MSE", format!("{:.3}", mse));

    // Classification: predict `level` from (item, boot_size bucket).
    let x_cls: Vec<Vec<String>> = rentals
        .iter()
        .map(|r| {
            vec![
                r.item.clone(),
                ((r.boot_size.saturating_sub(38)) / 2).to_string(),
            ]
        })
        .collect();
    let y_cls: Vec<String> = rentals.iter().map(|r| r.level.clone()).collect();
    let mut nb = CategoricalNaiveBayes::new();
    nb.fit(&x_cls, &y_cls);
    let clf: &dyn Classifier = &nb; // DIP: use the abstraction
    println!("{} (predict level from item + boot bucket)", clf.name());

    // Parallel inference + training accuracy (rayon).
    let n = x_cls.len();
    let correct: usize = (0..n)
        .into_par_iter() // <-- multithreading in ML inference
        .filter(|&i| nb.predict(&x_cls[i]) == y_cls[i])
        .count();
    labeled!("train accuracy", format!("{}/{} = {:.1}%", correct, n, 100.0 * correct as f64 / n as f64));

    // A prediction for a brand-new rental.
    let sample = vec!["ski".to_string(), "2".to_string()]; // ski, boot bucket 2 (= 42)
    labeled!("predict(ski, boot~42)", nb.predict(&sample));

    Ok(())
}
