//! Собственные декларативные макросы (crate-level).
//!
//! Макросы в Rust — это кодогенерация на уровне синтаксиса. Здесь определены
//! три макроса, которые уменьшают бойлерплейт в `main` и тестах.

/// Вывести «имя: значение» с округлением до 1 знака — повторяет Python-эталон
/// вида `Adelie: 38.9`.
#[macro_export]
macro_rules! dbg_num {
    ($name:expr, $value:expr) => {
        println!("{}: {:.1}", $name, $value);
    };
    ($name:expr, $value:expr, $digits:expr) => {
        println!("{}: {:.*}", $name, $digits, $value);
    };
}

/// Утверждение, что фактическое число строк совпадает с ожидаемым.
/// Используется в интеграционных проверках.
#[macro_export]
macro_rules! assert_count {
    ($label:expr, $actual:expr, $expected:expr) => {{
        let actual = $actual;
        let expected = $expected;
        assert!(
            actual == expected,
            "{}: expected {}, got {}",
            $label,
            expected,
            actual
        );
    }};
}

/// Сформировать человекочитаемую сводку из нескольких полей.
/// Возвращает `String`, удобную для логирования.
#[macro_export]
macro_rules! summary {
    ($($key:expr => $value:expr),+ $(,)?) => {{
        let mut s = String::new();
        $(
            s.push_str(&format!("{}: {}\n", $key, $value));
        )+
        s.trim_end().to_string()
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    fn summary_concatenates_fields() {
        let s = summary!("A" => 1, "B" => 2);
        assert_eq!(s, "A: 1\nB: 2");
    }
}
