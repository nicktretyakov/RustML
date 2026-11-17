//! Declarative macros (`macro_rules!`) used across the crate.
//!
//! These complement the procedural derive macros pulled in by `serde`
//! (`#[derive(Serialize, Deserialize)]` on [`crate::model::RiverRecord`]).

/// Print a labelled metric on stdout in a uniform column layout.
///
/// ```
/// river_analysis::report_metric!("R-squared", 0.9876);
/// river_analysis::report_metric!("slope (cm/mm)", 36.2);
/// ```
#[macro_export]
macro_rules! report_metric {
    ($label:expr, $value:expr) => {{
        println!("{:<22} {:>12.4}", $label, $value as f64);
    }};
}

/// Compile-time-ish runtime assertion that two numbers agree within a tolerance.
/// Used as a sanity check on the fitted model (e.g. the slope must be positive
/// because heavier rainfall raises the river level).
#[macro_export]
macro_rules! assert_close {
    ($a:expr, $b:expr, $tol:expr) => {{
        let (a, b) = ($a as f64, $b as f64);
        assert!(
            (a - b).abs() <= $tol as f64,
            "assert_close failed: {} vs {} (tolerance {})",
            a,
            b,
            $tol as f64
        );
    }};
}

/// Build a small `Vec<(f64, f64)>` of points inline. Handy for ad-hoc plotting
/// in tests and demonstrates a variadic `macro_rules!` with repetition.
#[macro_export]
macro_rules! points {
    ( $( $x:expr => $y:expr ),* $(,)? ) => {{
        vec![ $( ($x as f64, $y as f64) ),* ]
    }};
}
