//! Беговой клуб: чтение `runs.csv`, расчёт среднего значения выбранного столбца
//! для каждого бегуна и определение бегуна с наибольшим средним.
//!
//! Запуск:
//!     cargo run -- <column> [path-to-csv]
//!
//! Параметры:
//!     column  — имя числового столбца: "distance_km" | "minutes" | "climb_m"
//!     path    — путь к CSV (по умолчанию "runs.csv" рядом с бинарником)
//!
//! Пример вывода для столбца `distance_km` и двух бегунов Ana, Bo:
//!     runner
//!     Ana    6.1
//!     Bo     8.4
//!     Name: distance_km, dtype: float64
//!     Top runner: Bo with 8.4

mod macros;
mod series;

use series::Series;

use csv::ReaderBuilder;
use serde::Deserialize;
use std::env;
use std::error::Error;
use std::process::ExitCode;

/// Одна запись из `runs.csv`. Поля — ровно столбцы файла.
/// `distance_km` и `minutes` — вещественные; `climb_m` — целое число метров
/// набора высоты (но для расчёта среднего приводим всё к `f64`).
#[derive(Debug, Deserialize)]
pub struct Run {
    pub runner: String,
    #[serde(deserialize_with = "de_f64")]
    pub distance_km: f64,
    #[serde(deserialize_with = "de_f64")]
    pub minutes: f64,
    #[serde(deserialize_with = "de_i64")]
    pub climb_m: i64,
}

/// Гибкий десериализатор `f64`: принимает как "8.4", так и "8" (целое без точки),
/// пропускает окружающие пробелы и пустые значения (пусто = 0.0).
fn de_f64<'de, D>(d: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;
    let s: &str = Deserialize::deserialize(d)?;
    let s = s.trim();
    if s.is_empty() {
        return Ok(0.0);
    }
    s.parse::<f64>().map_err(|e| D::Error::custom(format!("bad float {s:?}: {e}")))
}

/// Гибкий десериализатор `i64` для `climb_m`: пусто = 0.
fn de_i64<'de, D>(d: D) -> Result<i64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;
    let s: &str = Deserialize::deserialize(d)?;
    let s = s.trim();
    if s.is_empty() {
        return Ok(0);
    }
    s.parse::<i64>().map_err(|e| D::Error::custom(format!("bad int {s:?}: {e}")))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let column = match args.get(1) {
        Some(c) => c.as_str(),
        None => {
            eprintln!("Использование: cargo run -- <column> [path-to-csv]");
            eprintln!("  column: distance_km | minutes | climb_m");
            return ExitCode::from(2);
        }
    };
    let path = args.get(2).map(String::as_str).unwrap_or("runs.csv");

    if let Err(e) = run(column, path) {
        eprintln!("Ошибка: {e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn run(column: &str, path: &str) -> Result<(), Box<dyn Error>> {
    let runs = read_csv(path)?;

    // Выбор столбца по имени. Из-за отсутствия в Rust reflection этот
    // статический `match` — идиоматичный способ диспетчеризации по строке.
    // `match` обращается к макросу `build_series!`, который сам по имени
    // столбца вызовет нужную ветку `dispatch_column!`.
    let series: Series = match column {
        "distance_km" => build_series!("distance_km", &runs),
        "minutes"     => build_series!("minutes",     &runs),
        "climb_m"     => build_series!("climb_m",     &runs),
        other => return Err(format!("Неизвестный столбец: {other}. Допустимо: distance_km, minutes, climb_m").into()),
    };

    // 1) Среднее значение столбца для каждого бегуна — печать как pandas Series.
    series.print();

    // 2) Бегун с наибольшим средним.
    let pairs = series.as_pairs();
    if let Some((name, value)) = top_runner!(&pairs) {
        println!("Top runner: {} with {:.1}", name, value);
    }
    Ok(())
}

/// Чтение всего CSV в `Vec<Run>`. Заголовок берётся из первой строки файла.
fn read_csv(path: &str) -> Result<Vec<Run>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .trim(csv::Trim::All)
        .from_path(path)?;
    let mut out = Vec::new();
    for result in rdr.deserialize() {
        let run: Run = result?;
        out.push(run);
    }
    Ok(out)
}
