//! Набор декларативных макросов (`macro_rules!`), которые делают код в `main.rs`
//! лаконичным и легко расширяемым на новые числовые столбцы.
//!
//! Макросы используются здесь, чтобы:
//! * по строковому имени столбца выбрать нужное поле структуры `Run`
//!   (это нельзя сделать обычной функцией без reflection, которого в Rust нет);
//! * сформировать "Series" и напечатать его в pandas-стиле;
//! * найти бегуна с максимальным средним значением.

/// `dispatch_column!(run, "distance_km")` — извлекает из записи `Run` поле,
/// соответствующее имени столбца, и возвращает его как `f64`.
///
/// Здесь `run` — выражение типа `&Run`, а второй аргумент — строковый литерал
/// с именем столбца. На этапе компиляции известны только поддерживаемые
/// имена столбцов; любой другой литерал приведёт к ошибке раскрытия макроса,
/// что защищает от опечаток в имени столбца.
#[macro_export]
macro_rules! dispatch_column {
    ($run:expr, "distance_km") => { $run.distance_km as f64 };
    ($run:expr, "minutes")     => { $run.minutes     as f64 };
    ($run:expr, "climb_m")     => { $run.climb_m     as f64 };
}

/// `mean!(values)` — считает среднее арифметическое среза `&[f64]`.
///
/// Возвращает `0.0` для пустого среза, чтобы избежать деления на ноль.
/// Хоть это и простая операция, оформим её макросом ради единообразия:
/// позже сюда легко добавить, например, медиану или взвешенное среднее,
/// не меняя код в `main.rs`.
#[macro_export]
macro_rules! mean {
    ($values:expr) => {{
        let v: &[f64] = $values;
        if v.is_empty() {
            0.0
        } else {
            let sum: f64 = v.iter().sum();
            sum / v.len() as f64
        }
    }};
}

/// `top_runner!(averages)` — находит бегуна с наибольшим средним значением.
///
/// Принимает срез `&[(&str, f64)]` (имя, среднее). Возвращает `Option<(&str, f64)>` —
/// `None` для пустого среза, иначе пару "имя, значение" для максимума.
/// Макрос инкапсулирует логику выбора максимума, чтобы `main.rs` остался
/// декларативным и читался как постановка задачи.
#[macro_export]
macro_rules! top_runner {
    ($averages:expr) => {{
        let a: &[(&str, f64)] = $averages;
        a.iter().max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)).copied()
    }};
}

/// `build_series!(column_name, rows)` — собирает `Series` из записей `Run`,
/// группируя значения по имени бегуна и заменяя каждое значение его средним.
///
/// `column_name` — строковый литерал имени столбца; `rows` — срез `&[Run]`.
/// Возвращает `Series` с одной строкой на бегуна и порядком первого появления,
/// как в pandas `groupby("runner")[column].mean()`.
#[macro_export]
macro_rules! build_series {
    ($column_name:tt, $rows:expr) => {{
        let rows: &[$crate::Run] = $rows;
        // Сохраняем порядок первого появления имени бегуна.
        let mut order: Vec<String> = Vec::new();
        // Накапливаем значения по каждому бегуну.
        let mut grouped: std::collections::HashMap<String, Vec<f64>> =
            std::collections::HashMap::new();

        for run in rows {
            let value = $crate::dispatch_column!(run, $column_name);
            grouped.entry(run.runner.clone()).or_default().push(value);
            if !order.contains(&run.runner) {
                order.push(run.runner.clone());
            }
        }

        let entries: Vec<(String, f64)> = order
            .iter()
            .map(|name| {
                let vals = grouped.get(name).unwrap();
                (name.clone(), $crate::mean!(vals))
            })
            .collect();

        Series::new($column_name, entries)
    }};
}
