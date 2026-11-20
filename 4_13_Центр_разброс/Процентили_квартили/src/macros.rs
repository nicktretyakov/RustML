//! Модуль `macros` — собственные декларативные макросы и парсинг ввода.
//!
//! Макросы снижают дублирование кода и упрощают расширение: новые числовые
//! правила форматирования вводятся одной строкой `format_metric!`.

use std::error::Error;
use std::io::{self, BufRead, Write};

/// Декларативный макрос для форматирования вещественной метрики с одним
/// знаком после запятой. Используется в отчёте (DRY).
macro_rules! format_metric {
    ($label:expr, $value:expr) => {
        format!("{}: {:.1} liters", $label, $value)
    };
}

/// Декларативный макрос: формирует строку квартиля клиента.
macro_rules! quarter_line {
    ($quarter:expr) => {
        format!("This home is in quarter {}", $quarter)
    };
}

pub(crate) use format_metric;
pub(crate) use quarter_line;

/// Читает значение расхода воды клиента из stdin.
pub fn read_client_liters() -> Result<f64, Box<dyn Error>> {
    // Печатаем приглашение в stdout (flush), затем читаем строку из stdin.
    let stdout = io::stdout();
    let mut out = stdout.lock();
    out.write_all(b"Enter this home's water usage in liters: ")?;
    out.flush()?;

    let stdin = io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line)?;
    let trimmed = line.trim();
    let value: f64 = trimmed.parse()?;
    Ok(value)
}
