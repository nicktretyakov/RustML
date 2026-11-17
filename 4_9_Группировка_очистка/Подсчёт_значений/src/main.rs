//! Профилирование столбцов журнала поездок `bikes.csv`.
//!
//! Программа читает CSV через крейт `csv`. Для каждого столбца из списка `columns`
//! (в исходном порядке) печатается одна строка вида:
//!
//! ```text
//! color: 4 different values, most often red (12)
//! ```
//!
//! Подсчёт частот значений и печать строки профиля вынесены в макрос `profile!`.
//! В конце добавлена типизированная сводка через крейт `serde` — она десериализует
//! каждую запись в структуру `Bike` и считает среднюю длительность поездки, чтобы
//! продемонстрировать совместную работу обоих крейтов.

use csv::ReaderBuilder;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::process;

/// Одна поездка. Поля идут в том же порядке, что и столбцы в `bikes.csv`,
/// поэтому `csv::StringRecord::deserialize` корректно разбирает запись позиционно.
/// Поля нужны serde при десериализации; rustc этого не видит — разрешаем dead_code.
#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct Bike {
    trip_id: u32,
    bike_id: u32,
    color: String,
    size: String,
    start_station: String,
    duration_min: u32,
}

/// Макрос профилирования одного столбца.
///
/// Аргументы:
/// - `$records` — срез `&[csv::StringRecord]` со всеми строками;
/// - `$headers` — `&csv::StringRecord` с заголовками;
/// - `$name`    — имя столбца (`&str`).
///
/// Макрос разворачивается внутри цикла и при отсутствии столбца печатает
/// предупреждение и переходит к следующей итерации (`continue`).
macro_rules! profile {
    ($records:expr, $headers:expr, $name:expr) => {{
        let headers: &csv::StringRecord = $headers;
        let idx = match headers.iter().position(|h| h == $name) {
            Some(i) => i,
            None => {
                eprintln!("предупреждение: столбец '{}' не найден — пропускаю", $name);
                continue;
            }
        };

        // Подсчёт частот: значение -> сколько раз встретилось в столбце.
        let mut counts: HashMap<String, usize> = HashMap::new();
        for rec in $records {
            if let Some(val) = rec.get(idx) {
                *counts.entry(val.to_string()).or_insert(0) += 1;
            }
        }

        let distinct = counts.len();

        // Самое частое значение. При равенстве частот берём лексикографически
        // наименьшее, чтобы вывод был детерминированным от запуска к запуску.
        let (top_val, top_count) = counts
            .iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(b.0)))
            .map(|(v, c)| (v.as_str(), *c))
            .unwrap_or(("", 0usize));

        println!(
            "{}: {} different values, most often {} ({})",
            $name, distinct, top_val, top_count
        );
    }};
}

fn run() -> Result<(), Box<dyn Error>> {
    // Аргументы командной строки: [путь_к_csv] [столбец ...].
    // Если столбцы не переданы — используется набор по умолчанию.
    let mut args = env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "bikes.csv".to_string());
    let columns_from_args: Vec<String> = args.collect();
    let columns: Vec<&str> = if columns_from_args.is_empty() {
        // Начальный код считывает названия профилируемых столбцов в `columns`.
        // Порядок в этом векторе задаёт порядок вывода строк.
        vec!["color", "size", "start_station"]
    } else {
        columns_from_args.iter().map(|s| s.as_str()).collect()
    };

    // Чтение CSV с заголовком.
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(&path)?;
    let headers = rdr.headers()?.clone();
    let records: Vec<csv::StringRecord> = rdr.records().collect::<Result<_, _>>()?;

    println!(
        "Профиль {} столбцов из «{}» ({} поездок):\n",
        columns.len(),
        path,
        records.len()
    );

    // Профилируем каждый столбец в исходном порядке.
    for name in &columns {
        profile!(&records, &headers, *name);
    }

    // Типизированная сводка через serde: средняя длительность поездок.
    // Демонстрирует совместное использование крейтов csv и serde на тех же данных.
    let mut total_duration = 0u64;
    let mut typed = 0usize;
    for rec in &records {
        if let Ok(bike) = rec.deserialize::<Bike>(None) {
            total_duration += bike.duration_min as u64;
            typed += 1;
        }
    }
    if typed > 0 {
        let avg = total_duration as f64 / typed as f64;
        println!();
        println!(
            "Сводка (serde): десериализовано {} записей, средняя длительность {:.1} мин",
            typed, avg
        );
    }

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ошибка: {}", err);
        process::exit(1);
    }
}
