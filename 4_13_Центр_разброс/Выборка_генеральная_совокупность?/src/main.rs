//! bat-wings-std — Rust port of the NumPy/Pandas standard-deviation task.
//!
//! A biologist computed two different standard deviations for the wing-span
//! sample of several bats. The discrepancy is the classic sample-vs-population
//! one: NumPy's `np.std` divides by `n` (ddof = 0) by default, while Pandas'
//! `Series.std` divides by `n - 1` (ddof = 1, the *sample* / Bessel-corrected
//! estimator). Because these bats are a *sample*, we must force NumPy to use
//! `ddof = 1` so both tools agree, then report how much the population
//! estimator underestimated the sample one.
//!
//! Beyond reproducing the Python logic, this binary demonstrates:
//!   * **macros**            — `macro_rules!` helpers plus the built-in
//!                              `println!` / `vec!` macros;
//!   * **external crates**   — `ndarray` (NumPy analogue) and `rayon`
//!                              (data-parallel multithreading);
//!   * **OOP**               — traits as interfaces, structs with `impl`,
//!                              encapsulated fields and trait-object
//!                              polymorphism;
//!   * **SOLID**             — see the per-item comments below;
//!   * **multithreading**    — `rayon` parallel reduction *and* explicit
//!                              `std::thread` workers joined at the end;
//!   * **machine learning**  — a `StandardScaler` (z-score standardization),
//!                              the canonical ML preprocessing transformer,
//!                              fitted on the wing spans and exposing a
//!                              scikit-learn-style `Transformer` API.
//!
//! Input format (identical to the Python starter):
//!   line 1:  n            (number of measurements)
//!   next n:  one integer per line, wing span in cm
//!
//! Example input  ->  expected stdout
//!   5            Sample std: 3.19 cm, tools agree: True
//!   33           Dividing by n underestimated it by 0.34 cm
//!   27
//!   30
//!   35
//!   29

use ndarray::Array1;
use rayon::prelude::*;
use std::io::{self, BufRead, Write};
use std::sync::Arc;

// ===========================================================================
// MACROS
// ===========================================================================
// Declarative `macro_rules!` helpers keep the call sites terse and declarative,
// and they showcase Rust's hygienic macro system alongside the built-in
// `println!`, `vec!` and `format!` macros used throughout `main`.

/// Builds a `Vec<f64>` from an arbitrary list of numeric expressions.
/// (`as f64` lets callers pass integer literals such as `33, 27, 30`.)
/// Exercised by the unit tests below; the allow keeps the release build quiet.
#[allow(unused_macros)]
macro_rules! f64_vec {
    ($($x:expr),+ $(,)?) => {{ vec![ $( $x as f64 ),+ ] }};
}

/// Emits a labelled, unit-suffixed statistic line to **stderr** (so the
/// supplementary diagnostics never disturb the byte-exact stdout contract).
macro_rules! print_stat {
    ($label:expr, $value:expr, $unit:expr) => {
        eprintln!("{}: {:.2} {}", $label, $value, $unit);
    };
}

/// Renders a bool the Python way (`True` / `False`) so the agreement line
/// matches the reference output character-for-character.
macro_rules! py_bool {
    ($b:expr) => {{
        if $b { "True" } else { "False" }
    }};
}

// ===========================================================================
// TRAITS  —  Interface Segregation (SOLID-I) + Dependency Inversion (SOLID-D)
// ===========================================================================
// Each trait carries a single, narrowly-scoped responsibility, and every
// consumer (including `main`) depends on the *trait*, never on a concrete
// struct. That is the heart of the Dependency-Inversion Principle: high-level
// policy here is independent of the low-level engine chosen at run time.

/// Single Responsibility: fetch raw observations only.
/// (Open/Closed: a `MockSource` or `FileSource` can be added without touching
/// any existing code.)
trait DataSource {
    fn fetch(&self) -> Vec<f64>;
}

/// Single Responsibility: pure statistical reduction over a `&[f64]`.
/// `ddof` ("delta degrees of freedom") mirrors NumPy/Pandas:
///   ddof = 0  ->  divide by n        (population)
///   ddof = 1  ->  divide by n - 1    (sample / Bessel correction)
trait Statistics {
    fn mean(&self, data: &[f64]) -> f64;
    /// Sum of squared deviations from the mean (the shared inner kernel).
    fn sum_squared_dev(&self, data: &[f64]) -> f64;
    fn variance(&self, data: &[f64], ddof: usize) -> f64;
    fn std_dev(&self, data: &[f64], ddof: usize) -> f64;
}

/// Single Responsibility: a *fitted* transformer in the scikit-learn sense
/// (anything that maps raw features to processed features). The estimator
/// surface (`fit` + `transform`) is intentionally split: `fit` lives on the
/// concrete builder, `transform` on this trait — Interface Segregation keeps
/// callers that only need to *apply* a model from seeing its training API.
trait Transformer {
    fn name(&self) -> &'static str;
    fn transform(&self, data: &[f64]) -> Vec<f64>;
}

// ===========================================================================
// STRUCTS  —  concrete implementations (Liskov: any of them substitutes for
// its trait)
// ===========================================================================

/// Reads exactly `n` integer measurements from stdin, mirroring the Python
/// `for i in range(n): values.append(int(input()))` loop.
struct StdinSource {
    n: usize,
}

impl DataSource for StdinSource {
    fn fetch(&self) -> Vec<f64> {
        let stdin = io::stdin();
        let mut out = Vec::with_capacity(self.n);
        for line in stdin.lock().lines().take(self.n) {
            let line = line.expect("expected a UTF-8 line");
            let v: f64 = line.trim().parse().expect("expected an integer span");
            out.push(v);
        }
        out
    }
}

/// Sequential statistics engine — single-threaded reference implementation.
struct SequentialStats;

impl Statistics for SequentialStats {
    fn mean(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return f64::NAN;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }
    fn sum_squared_dev(&self, data: &[f64]) -> f64 {
        let m = self.mean(data);
        data.iter().map(|x| {
            let d = x - m;
            d * d
        }).sum()
    }
    fn variance(&self, data: &[f64], ddof: usize) -> f64 {
        let n = data.len();
        if n <= ddof {
            return f64::NAN;
        }
        self.sum_squared_dev(data) / (n - ddof) as f64
    }
    fn std_dev(&self, data: &[f64], ddof: usize) -> f64 {
        self.variance(data, ddof).sqrt()
    }
}

/// Multithreaded statistics engine built on `rayon`. The sum and the
/// sum-of-squared-deviations are reduced in parallel across the available
/// cores; reductions on `f64` are associative-friendly and the data is
/// read-only, so the parallel version is race-free and deterministic.
struct ParallelStats;

impl Statistics for ParallelStats {
    fn mean(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return f64::NAN;
        }
        // `par_iter().sum()` is a parallel reduction (work-stealing, lock-free).
        let sum: f64 = data.par_iter().sum();
        sum / data.len() as f64
    }
    fn sum_squared_dev(&self, data: &[f64]) -> f64 {
        let m = self.mean(data);
        data.par_iter()
            .map(|x| {
                let d = x - m;
                d * d
            })
            .sum() // parallel map-reduce
    }
    fn variance(&self, data: &[f64], ddof: usize) -> f64 {
        let n = data.len();
        if n <= ddof {
            return f64::NAN;
        }
        self.sum_squared_dev(data) / (n - ddof) as f64
    }
    fn std_dev(&self, data: &[f64], ddof: usize) -> f64 {
        self.variance(data, ddof).sqrt()
    }
}

/// ML preprocessor: z-score standardization (the `StandardScaler` from
/// scikit-learn). It stores the mean and sample standard deviation learned
/// during `fit`, then maps every feature `x` to `(x - mean) / std`. This is
/// the most common feature-scaling step feeding distance/gradient-based ML
/// models (k-NN, SVM, logistic regression, neural nets).
struct StandardScaler {
    mean: f64,
    std: f64, // sample std (ddof = 1) — the ML default
}

impl StandardScaler {
    /// Constructs an unfitted scaler (mean / std = NaN).
    fn new() -> Self {
        Self { mean: f64::NAN, std: f64::NAN }
    }

    /// Fits the scaler on `data`. The estimator is *passed a `&dyn
    /// Statistics`* rather than a concrete engine — Dependency Inversion:
    /// `StandardScaler` does not know or care whether stats came from the
    /// sequential or the parallel engine.
    fn fit(mut self, data: &[f64], engine: &dyn Statistics) -> Self {
        self.mean = engine.mean(data);
        self.std = engine.std_dev(data, 1); // sample standard deviation
        self
    }
}

impl Transformer for StandardScaler {
    fn name(&self) -> &'static str { "StandardScaler(z-score)" }

    fn transform(&self, data: &[f64]) -> Vec<f64> {
        // Guard against zero/NaN std to avoid division blow-ups on degenerate
        // samples (a constant feature). Falls back to no-op scaling.
        let s = if self.std == 0.0 || self.std.is_nan() {
            1.0
        } else {
            self.std
        };
        data.iter().map(|x| (x - self.mean) / s).collect()
    }
}

// ===========================================================================
// ndarray-backed view  —  the "Pandas Series / NumPy array" counterpart
// ===========================================================================

/// Wraps an `ndarray::Array1<f64>` and computes standard deviation for a given
/// `ddof`. This is the Rust analogue of constructing both `np.array(values)`
/// and `pd.Series(values)` and comparing their `.std()` outputs.
struct NdArrayColumn(Array1<f64>);

impl NdArrayColumn {
    fn from_vec(v: Vec<f64>) -> Self {
        Self(Array1::from_vec(v))
    }

    /// `ddof = 1` reproduces `pandas.Series.std()`; `ddof = 0` reproduces the
    /// `numpy.std()` *default* that the original bug relied on.
    fn std_dev(&self, ddof: usize) -> f64 {
        let view = self.0.view();
        let n = view.len();
        if n <= ddof {
            return f64::NAN;
        }
        let mean = view.sum() / n as f64;
        let ssd = view.mapv(|x| {
            let d = x - mean;
            d * d
        }).sum();
        (ssd / (n - ddof) as f64).sqrt()
    }
}

// ===========================================================================
// Explicit multithreading with std::thread
// ===========================================================================

/// Spawns two OS threads that independently compute the population (`ddof=0`)
/// and sample (`ddof=1`) standard deviations, then joins them. This sits
/// alongside the `rayon` data-parallelism above to show both flavours of
/// concurrency: fine-grained parallel reduction *and* coarse task parallelism.
///
/// `Arc<Vec<f64>>` gives both threads cheap, shared, read-only access to the
/// data with no copying.
fn concurrent_std_pair(data: Arc<Vec<f64>>) -> (f64, f64) {
    let d0 = Arc::clone(&data);
    let d1 = Arc::clone(&data);

    let pop_handle = std::thread::spawn(move || {
        ParallelStats.std_dev(&d0, 0)
    });
    let sample_handle = std::thread::spawn(move || {
        ParallelStats.std_dev(&d1, 1)
    });

    let pop = pop_handle.join().expect("population-std thread panicked");
    let sample = sample_handle.join().expect("sample-std thread panicked");
    (pop, sample)
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Round to two decimals, matching `numpy.round(x, 2)`.
fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

// ===========================================================================
// main — orchestration depending only on traits
// ===========================================================================

fn main() {
    // --- read n, then n measurements from stdin (Python parity) ----------
    let mut first = String::new();
    io::stdin()
        .read_line(&mut first)
        .expect("could not read the first line (n)");
    let n: usize = first
        .trim()
        .parse()
        .expect("expected n as a non-negative integer");

    let source = StdinSource { n };
    let values: Vec<f64> = source.fetch();

    // ndarray view == the pandas.Series column the Python code operates on.
    let column = NdArrayColumn::from_vec(values.clone());

    // Dependency Inversion: pick the engine through the trait object, so the
    // logic below is identical whether we use SequentialStats or ParallelStats.
    let engine: &dyn Statistics = &ParallelStats;

    // ---- THE FIX ---------------------------------------------------------
    // Original bug: `np.std(spans)` defaults to ddof=0 (population). For a
    // *sample* we must use ddof=1 so NumPy matches Pandas' sample default.
    let numpy_std = round2(engine.std_dev(&values, 1)); // ddof = 1  (fixed)
    let pandas_std = round2(column.std_dev(1));         // pandas default = 1
    let agree = numpy_std == pandas_std;

    println!(
        "Sample std: {} cm, tools agree: {}",
        numpy_std,
        py_bool!(agree)
    );

    // ---- the "underestimated by" gap line --------------------------------
    // Python:  gap = column.std() - np.std(spans)
    //          == sample std (ddof=1)  -  population std (ddof=0, NumPy default)
    let shared = Arc::new(values.clone());
    let (pop_std, _sample_std) = concurrent_std_pair(Arc::clone(&shared));
    let gap = column.std_dev(1) - pop_std; // sample minus population
    println!("Dividing by n underestimated it by {} cm", round2(gap));

    // ---- Liskov runtime check --------------------------------------------
    // SequentialStats and ParallelStats are interchangeable substitutes for the
    // `Statistics` trait, so both must agree on the (rounded) sample std.
    let seq_sample = round2(SequentialStats.std_dev(&values, 1));
    let par_sample = round2(ParallelStats.std_dev(&values, 1));
    let _ = writeln!(
        io::stderr(),
        "[LSP] sequential sample std = {} cm, parallel sample std = {} cm (agree: {})",
        seq_sample,
        par_sample,
        py_bool!(seq_sample == par_sample)
    );

    // ---- ML demonstration: fit a StandardScaler on the wing spans --------
    // Standardization (z-scoring) is the bread-and-butter ML preprocessing
    // step that *consumes* the sample standard deviation we just fixed.
    let scaler = StandardScaler::new().fit(&values, engine);
    let scaled = scaler.transform(&values);
    let z_rounded: Vec<f64> = scaled.iter().map(|z| round2(*z)).collect();

    // Supplementary ML output goes to stderr so stdout stays byte-for-byte
    // identical to the Python reference. The `print_stat!` macro formats it.
    let _ = writeln!(io::stderr(),
        "[ML] {} fitted on {} samples", scaler.name(), values.len());
    print_stat!("[ML] mean", scaler.mean, "cm");
    print_stat!("[ML] sample_std (ddof=1)", scaler.std, "cm");
    let _ = writeln!(io::stderr(),
        "[ML] standardized spans (z-scores): {:?}", z_rounded);
}

// ===========================================================================
// Tests — assert the math and the SOLID contracts hold (run with `cargo test`)
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<f64> {
        f64_vec![33, 27, 30, 35, 29]
    }

    #[test]
    fn sample_std_matches_reference() {
        let data = sample();
        // Known reference: sample std = 3.19 (rounded to 2 dp).
        assert_eq!(round2(ParallelStats.std_dev(&data, 1)), 3.19);
    }

    #[test]
    fn numpy_and_pandas_agree_when_ddof_is_one() {
        let data = sample();
        let column = NdArrayColumn::from_vec(data.clone());
        // Liskov: both engines must agree, so we exercise the trait object.
        let engines: Vec<Box<dyn Statistics>> =
            vec![Box::new(SequentialStats), Box::new(ParallelStats)];
        for eng in &engines {
            assert_eq!(round2(eng.std_dev(&data, 1)), round2(column.std_dev(1)));
        }
    }

    #[test]
    fn population_underestimates_sample_by_0_34() {
        let data = sample();
        let column = NdArrayColumn::from_vec(data.clone());
        let gap = column.std_dev(1) - ParallelStats.std_dev(&data, 0);
        assert_eq!(round2(gap), 0.34);
    }

    #[test]
    fn standard_scaler_produces_zero_mean_unit_std() {
        // The defining property of z-score standardization: transformed data
        // has mean ~0 and sample std ~1.
        let data = sample();
        let scaler = StandardScaler::new().fit(&data, &ParallelStats);
        let z = scaler.transform(&data);
        let mean = ParallelStats.mean(&z);
        let std = ParallelStats.std_dev(&z, 1);
        assert!(mean.abs() < 1e-9);
        assert!((std - 1.0).abs() < 1e-9);
    }

    #[test]
    fn transformer_is_open_for_extension() {
        // OCP smoke test: a brand-new transformer plugs into the trait without
        // touching StandardScaler or main.
        struct Identity;
        impl Transformer for Identity {
            fn name(&self) -> &'static str { "Identity" }
            fn transform(&self, data: &[f64]) -> Vec<f64> { data.to_vec() }
        }
        let data = sample();
        let t: Box<dyn Transformer> = Box::new(Identity);
        assert_eq!(t.transform(&data), data);
    }
}
