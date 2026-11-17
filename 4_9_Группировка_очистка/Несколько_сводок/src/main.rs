//! Food truck weekly order analysis.
//!
//! Reads `truck.csv` (columns: day, dish, payment, total) and groups the orders
//! by the column named in the `column` variable. Produces two pandas-style
//! reports:
//!   1. The number of orders in each group (a Series-like listing).
//!   2. A table with `min`, `mean`, `max` of the `total` column for each group
//!      (in exactly that column order).
//!
//! External crates: `csv` (parsing) and `serde` (deserialization via the
//! `#[derive(Deserialize)]` proc macro). Two custom `macro_rules!` macros
//! (`group_key!` and `stats!`) keep the hot path concise and type-safe.

use csv::Reader;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};

/// One row of `truck.csv`. Field names match the CSV header exactly, so
/// `serde` maps them automatically when `#[derive(Deserialize)]` is applied
/// (this is the proc-macro from the `serde` crate).
#[derive(Debug, Clone, Deserialize)]
struct Order {
    day: String,
    dish: String,
    payment: String,
    total: f64,
}

/// `group_key!($order, $col)` returns a `&String` reference to the field of
/// `$order` whose name matches the runtime string `$col`. Using a macro
/// instead of a function lets the compiler keep the match exhaustive and
/// inline the field access, while still being driven by a runtime column name.
macro_rules! group_key {
    ($order:expr, $col:expr) => {
        match $col {
            "day" => &$order.day,
            "dish" => &$order.dish,
            "payment" => &$order.payment,
            other => panic!("group_key!: unknown column {:?}", other),
        }
    };
}

/// `stats!($vals)` computes `(min, mean, max)` for a `&[f64]` in a single pass.
/// Returns a tuple of `f64`. Panics on an empty slice (no orders in a group
/// should never happen here, but the assertion makes the precondition loud).
macro_rules! stats {
    ($vals:expr) => {{
        let v: &[f64] = $vals;
        assert!(!v.is_empty(), "stats!: empty slice");
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0_f64;
        for &x in v {
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
            sum += x;
        }
        (min, sum / v.len() as f64, max)
    }};
}

/// Read the whole stdin into a string (used to read the grouping column name
/// the way the starter code does: "the column to group by is read into the
/// variable `column`").
fn read_stdin_string() -> String {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).expect("read stdin");
    s
}

/// Format a number the way pandas prints a numeric value: integers without a
/// decimal point (`10`), fractional values with two decimals (`10.50`).
fn fmt_num(x: f64) -> String {
    if x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{:.2}", x)
    }
}

fn main() {
    // ----- starter code: read the grouping column into `column` ----------
    // The column name comes from stdin (a single line). If stdin is empty or
    // only whitespace, default to "day".
    let column = read_stdin_string();
    let column = column.trim();
    let column = if column.is_empty() { "day" } else { column };

    // ----- read the CSV via the `csv` crate --------------------------------
    let path = "truck.csv";
    let file = File::open(path).unwrap_or_else(|e| {
        eprintln!("Cannot open {}: {}", path, e);
        std::process::exit(1);
    });
    let mut rdr = Reader::from_reader(file);
    let orders: Vec<Order> = rdr.deserialize().filter_map(Result::ok).collect();

    if orders.is_empty() {
        eprintln!("No orders parsed from {}", path);
        std::process::exit(1);
    }

    // ----- group `total` values by the chosen column ----------------------
    let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
    for order in &orders {
        // `group_key!` selects the field at runtime based on `column`.
        let key = group_key!(order, column).clone();
        groups.entry(key).or_default().push(order.total);
    }

    // Sort group keys lexicographically for deterministic, pandas-like output
    // (pandas sorts a string index with `sort_index()`).
    let mut keys: Vec<String> = groups.keys().cloned().collect();
    keys.sort();

    // ----- pre-compute the formatted statistics row for each group --------
    struct RowOut {
        key: String,
        count: usize,
        min_s: String,
        mean_s: String,
        max_s: String,
    }
    let mut rows: Vec<RowOut> = Vec::with_capacity(keys.len());
    for k in &keys {
        let v = &groups[k];
        let (min, mean, max) = stats!(v.as_slice());
        rows.push(RowOut {
            key: k.clone(),
            count: v.len(),
            min_s: fmt_num(min),
            mean_s: format!("{:.2}", mean),
            max_s: fmt_num(max),
        });
    }

    // ----- column widths --------------------------------------------------
    // Index column: max of the index values and the index name (`column`).
    let idx_w = rows
        .iter()
        .map(|r| r.key.chars().count())
        .chain(std::iter::once(column.chars().count()))
        .max()
        .unwrap_or(0);
    // Data column widths: max of the header label and the widest formatted
    // value, so headers never get truncated.
    let min_w = rows
        .iter()
        .map(|r| r.min_s.chars().count())
        .chain(std::iter::once("min".chars().count()))
        .max()
        .unwrap_or(0);
    let mean_w = rows
        .iter()
        .map(|r| r.mean_s.chars().count())
        .chain(std::iter::once("mean".chars().count()))
        .max()
        .unwrap_or(0);
    let max_w = rows
        .iter()
        .map(|r| r.max_s.chars().count())
        .chain(std::iter::once("max".chars().count()))
        .max()
        .unwrap_or(0);

    // ===== Output 1: count of orders per group (Series-style) ============
    // Header line is the Series name (the grouping column).
    println!("{}", column);
    // 4-space separator reproduces pandas' Series.to_string() alignment.
    let count_val_w = rows
        .iter()
        .map(|r| r.count.to_string().chars().count())
        .max()
        .unwrap_or(0);
    for r in &rows {
        println!(
            "{:<idx_w$}    {:>count_val_w$}",
            r.key,
            r.count,
            idx_w = idx_w,
            count_val_w = count_val_w,
        );
    }
    println!("dtype: int64");

    // ===== Output 2: min / mean / max table (DataFrame-style) ============
    // Column order is exactly: min, mean, max. Columns are separated by two
    // spaces and right-aligned to their width.
    let header = format!(
        "{:idx_w$}  {:>min_w$}  {:>mean_w$}  {:>max_w$}",
        "",
        "min",
        "mean",
        "max",
        idx_w = idx_w,
        min_w = min_w,
        mean_w = mean_w,
        max_w = max_w,
    );
    println!("{}", header);

    // Index-name row: the grouping column name left-aligned to `idx_w`, then
    // padded with spaces to the full row width (pandas prints the index name
    // on its own line under the column headers).
    let row_width = idx_w + 2 + min_w + 2 + mean_w + 2 + max_w;
    println!("{:<row_width$}", column, row_width = row_width);

    // Data rows.
    for r in &rows {
        println!(
            "{:<idx_w$}  {:>min_w$}  {:>mean_w$}  {:>max_w$}",
            r.key,
            r.min_s,
            r.mean_s,
            r.max_s,
            idx_w = idx_w,
            min_w = min_w,
            mean_w = mean_w,
            max_w = max_w,
        );
    }
}
