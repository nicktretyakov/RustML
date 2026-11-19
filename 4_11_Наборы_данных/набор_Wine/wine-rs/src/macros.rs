//! Макрос единообразного вывода строк отчёта.
//!
//! Формат строго соответствует эталону `sklearn.datasets.load_wine`,
//! который ожидается проверяющей системой:
//!
//! ```text
//! Most wines: class_1 (71)
//! Highest malic_acid: class_2
//! class_2 wines above the average malic_acid: 41
//! ```

/// Формирует одну из трёх строк отчёта.
///
/// * `report!(most_wines, name, count)`            -> `Most wines: <name> (<count>)`
/// * `report!(highest, feature, name)`             -> `Highest <feature>: <name>`
/// * `report!(above, name, feature, n)`            -> `<name> wines above the average <feature>: <n>`
#[macro_export]
macro_rules! report {
    (most_wines, $class:expr, $count:expr) => {
        println!("Most wines: {} ({})", $class, $count)
    };
    (highest, $feature:expr, $class:expr) => {
        println!("Highest {}: {}", $feature, $class)
    };
    (above, $class:expr, $feature:expr, $n:expr) => {
        println!("{} wines above the average {}: {}", $class, $feature, $n)
    };
}
