use polars::prelude::*;

fn main() -> PolarsResult<()> {
    // 1. Чтение CSV
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("movies.csv".into()))?
        .finish()?;

    // 2. Переходим в lazy-режим и считаем агрегаты
    let stats = df
        .lazy()
        .select([
            col("rating").count().alias("count"),
            col("rating").min().alias("min"),
            col("rating").max().alias("max"),
            col("rating").mean().alias("mean"),
            col("votes").median().alias("median_votes"),
        ])
        .collect()?;  // ← материализуем результат

    // 3. Извлекаем значения
    let count = stats.column("count")?.get(0)?.try_extract::<u32>()?;
    let min = stats.column("min")?.get(0)?.try_extract::<f64>()?;
    let max = stats.column("max")?.get(0)?.try_extract::<f64>()?;
    let mean = stats.column("mean")?.get(0)?.try_extract::<f64>()?;
    let median_votes = stats.column("median_votes")?.get(0)?.try_extract::<f64>()? as i64;

    // 4. Вывод
    println!("Movies: {}", count);
    println!("Ratings run from {} to {}", min, max);
    println!("Average rating: {:.2}", mean);
    println!("Half the movies have over {} votes", median_votes);

    Ok(())
}
