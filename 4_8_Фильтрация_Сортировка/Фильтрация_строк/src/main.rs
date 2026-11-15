use anyhow::{Context, Result};
use polars::prelude::*; // с features = ["lazy", "csv"] подтягивает col, lit, LazyFrame
use std::io;

fn main() -> Result<()> {
    let df = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some("trails.csv".into()))
        .context("trails.csv")?
        .finish()
        .context("csv")?;

    let max_km: f64 = read_line()?;
    let min_climb: i64 = read_line()?;

    // сохраняем исходный индекс ДО фильтра
    let picked = df
        .lazy()
        .with_row_index("__idx", None)
        .with_column(col("km").cast(DataType::Float64))
        .with_column(col("climb_m").cast(DataType::Int64))
        .filter(
            col("km").lt_eq(lit(max_km))
                .and(col("climb_m").gt_eq(lit(min_climb))),
        )
        .collect()
        .context("filter")?;

    let n = picked.height();

    // вытаскиваем колонки в обычные Vec
    // ВАЖНО: колонка называется "name", а не "trail"
    let idx:   Vec<u32>  = picked.column("__idx")?.u32()?.into_no_null_iter().collect();
    let name:  Vec<&str> = picked.column("name")?.str()?.into_no_null_iter().collect();
    let km:    Vec<f64>  = picked.column("km")?.f64()?.into_no_null_iter().collect();
    let climb: Vec<i64>  = picked.column("climb_m")?.i64()?.into_no_null_iter().collect();

    // ширины колонок как у pandas
    let name_w = "name".len()
        .max(name.iter().map(|s| s.len()).max().unwrap_or(0));
    let km_w = "km".len()
        .max(km.iter().map(|v| format!("{:.1}", v).len()).max().unwrap_or(0));
    let climb_w = "climb_m".len()
        .max(climb.iter().map(|v| v.to_string().len()).max().unwrap_or(0));
    let idx_w = idx.iter().map(|i| i.to_string().len()).max().unwrap_or(1);

    // заголовок (pandas выравнивает заголовки вправо)
    println!(
        "{:>iw$}  {:>nw$}  {:>kw$}  {:>cw$}",
        "", "name", "km", "climb_m",
        iw = idx_w, nw = name_w, kw = km_w, cw = climb_w,
    );

    // строки: индекс вправо, name вправо, числа вправо
    for i in 0..n {
        println!(
            "{:>iw$}  {:>nw$}  {:>kw$.1}  {:>cw$}",
            idx[i], name[i], km[i], climb[i],
            iw = idx_w, nw = name_w, kw = km_w, cw = climb_w,
        );
    }

    println!("Trails that fit: {}", n);
    Ok(())
}

fn read_line<T>() -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let mut s = String::new();
    io::stdin().read_line(&mut s)?;
    Ok(s.trim().parse()?)
}
