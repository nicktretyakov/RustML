use polars::prelude::*;
use std::io::{self, Write};

/// Печатает число без лишнего `.0`, если оно целое.
fn fmt_num(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Аналог pd.read_csv("phones.csv")
    let file = std::fs::File::open("phones.csv")?;
    let df = CsvReader::new(file).finish()?;

    // Аналог input()
    print!("column: ");
    io::stdout().flush()?;
    let mut column = String::new();
    io::stdin().read_line(&mut column)?;
    let column = column.trim();

    // Аналог phones[["model", column]].head(3)
    let head = df.select(["model", column])?.head(Some(3));
    println!("{head}");

    // В polars 0.44 Column нужно материализовать в Series
    let s: &Series = df.column(column)?.as_materialized_series();

    // min/max — обобщённые, тип нужно указать явно (f64 реализует NumCast)
    // mean — возвращает Option<f64>, а не Result
    let min = s.min::<f64>()?.unwrap_or(f64::NAN);
    let max = s.max::<f64>()?.unwrap_or(f64::NAN);
    let mean = s.mean().unwrap_or(f64::NAN);

    // Аналог f"{column}: from {values.min()} to {values.max()}, average {round(values.mean(), 1)}"
    println!(
        "{column}: from {} to {}, average {mean:.1}",
        fmt_num(min),
        fmt_num(max)
    );

    Ok(())
}
