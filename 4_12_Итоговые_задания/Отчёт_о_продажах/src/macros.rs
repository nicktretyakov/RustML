//! Кастомные макросы и помощники форматирования.
//!
//! `macro_rules!` — мощный механизм метапрограммирования Rust. Здесь
//! определены макросы для логирования с форматированием и для безопасного
//! извлечения первого элемента из отсортированного вектора агрегаций.

/// Макрос логирования: выводит сообщение с префиксом уровня.
/// Пример: `log!(info, "Orders in {}: {}", region, count);`
#[macro_export]
macro_rules! log {
    ($level:ident, $($arg:tt)*) => {{
        let prefix = match stringify!($level) {
            "info" => "[INFO]",
            "warn" => "[WARN]",
            "err" => "[ERR ]",
            _ => "[LOG ]",
        };
        println!("{} {}", prefix, format!($($arg)*));
    }};
}

/// Макрос форматирования итоговой строки региона.
/// Принимает уже отформатированные строки прибыли и доли (чтобы обойти
/// гигиенические правила macro_rules, которые ищут имена функций в месте
/// вызова макроса, а не в месте его определения).
/// `print_summary!(region, product, profit_str, share_str)`
/// => "Most profit in East: Desk (414), 23.4% of the region"
#[macro_export]
macro_rules! print_summary {
    ($region:expr, $product:expr, $profit_str:expr, $share_str:expr) => {
        println!(
            "Most profit in {}: {} ({}), {}% of the region",
            $region, $product, $profit_str, $share_str
        )
    };
}

/// Макрос-помощник для безопасного получения топ-1 товара по прибыли.
/// Возвращает Option<&(String, f64)>.
#[macro_export]
macro_rules! top1 {
    ($slice:expr) => {
        $slice.first()
    };
}

/// Вспомогательная функция форматирования числа: 414.0 -> "414", 337.5 -> "337.5".
/// Реализует поведение как в примере вывода (без лишних .0).
pub fn format_profit(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        // Округлим до разумной точности, как в ожидаемом выводе.
        let r = (v * 10.0).round() / 10.0;
        format!("{}", r)
    }
}

/// Округление процентов до 1 знака после запятой (без trailing .0).
pub fn format_share(v: f64) -> String {
    let r = (v * 10.0).round() / 10.0;
    format!("{}", r)
}
