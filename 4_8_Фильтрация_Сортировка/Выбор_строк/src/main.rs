use polars::prelude::*;
use std::io::{self, Write};

fn main() -> PolarsResult<()> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("race.csv".into()))?
        .finish()?;

    print!("Введите место: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let place: usize = input.trim().parse().expect("Введите целое число");

    // Доступ по имени колонки — надёжнее, чем по индексу
    let runner_col = df.column("runner")?.str()?;  // &StringChunked
    let runner = runner_col
        .get(place - 1)
        .ok_or_else(|| PolarsError::ComputeError("место выходит за пределы".into()))?;

    // minutes может быть i64 или f64 — приводим к f64 на всякий случай
    let minutes_col = df.column("minutes")?.cast(&DataType::Float64)?;
    let minutes = minutes_col
        .f64()?
        .get(place - 1)
        .ok_or_else(|| PolarsError::ComputeError("место выходит за пределы".into()))?;

    println!("Place {}: {}", place, runner);
    println!("Time: {} minutes", minutes);

    // Последняя строка
    let last_idx = runner_col.len() - 1;
    let last_runner = runner_col
        .get(last_idx)
        .ok_or_else(|| PolarsError::ComputeError("колонка пуста".into()))?;
    println!("Last: {}", last_runner);

    Ok(())
}
