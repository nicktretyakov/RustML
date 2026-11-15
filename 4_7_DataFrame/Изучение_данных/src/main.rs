use polars::prelude::*;
use std::fs::File;
use std::io::{self, Write};

fn main() -> PolarsResult<()> {
    // 1. Чтение CSV — метод называется without `try_`
    let file = File::open("taxi_trips.csv")?;
    let trips = CsvReadOptions::default()
        .into_reader_with_file_handle(file)   // <-- исправлено
        .finish()?;

    // 2. stdin
    print!("n = ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let n: usize = input.trim().parse().unwrap_or(0);

    // 3. Количество строк
    println!("Trips: {}", trips.height());

    // 4. «numbers / text» — в polars describe() включает ВСЕ столбцы,
    //    поэтому pandas-проверка "name in summary.columns" тут не работает.
    //    Правильный эквивалент — проверка типа столбца.
    for name in trips.get_column_names() {
        let dtype = trips.column(name)?.dtype();
        if dtype.is_numeric() {
            println!("{name}: numbers");
        } else {
            println!("{name}: text");
        }
    }

    // 5. head(n) через Lazy API с DSL-функциями/макросами polars
    let head_df = trips
        .lazy()
        .select([col("*")])      // col() — DSL-выражение (аналог макроса из prelude)
        .limit(n as IdxSize)     // limit принимает IdxSize, НЕ Expr
        .collect()?;
    println!("{head_df}");

    Ok(())
}
