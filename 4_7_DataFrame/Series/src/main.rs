use polars::prelude::*;
use std::io::{self, BufRead};

fn read_line() -> String {
    let mut s = String::new();
    io::stdin().lock().read_line(&mut s).expect("read error");
    s.trim().to_string()
}

fn main() -> PolarsResult<()> {
    // ---- ввод ----
    let early:  i64 = read_line().parse().expect("early");
    let midday: i64 = read_line().parse().expect("midday");
    let late:   i64 = read_line().parse().expect("late");
    let trip        = read_line();

    // pd.Series([early, midday, late], index=["early","midday","late"])
    // В polars аналог — DataFrame с колонками "trip" и "passengers".
    // df! — макрос из polars::prelude, ближайший аналог конструктора pandas.
    let df = df![
        "trip"       => &["early", "midday", "late"],
        "passengers" => &[early, midday, late],
    ]?;

    // print(passengers) — Display у DataFrame рисует таблицу как pandas
    println!("{df}");

    // round(passengers.mean(), 1)
    // через lazy-запрос с агрегатом mean()
    let mean = df
        .clone()
        .lazy()
        .select([col("passengers").mean()])
        .collect()?
        .column("passengers")?
        .f64()?
        .get(0)
        .unwrap_or(0.0);
    println!("Average: {:.1} passengers", mean);

    // passengers[trip] — фильтр по колонке "trip" через lazy
    let value = df
        .lazy()
        .filter(col("trip").eq(lit(trip.as_str())))
        .select([col("passengers")])
        .collect()?
        .column("passengers")?
        .i64()?
        .get(0)
        .unwrap_or(0);
    println!("{trip}: {value} passengers");

    Ok(())
}
