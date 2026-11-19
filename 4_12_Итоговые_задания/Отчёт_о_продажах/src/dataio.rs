//! Модуль ввода-вывода: чтение CSV и парсинг в структуры Order.
//!
//! SOLID (Single Responsibility): этот модуль отвечает только за загрузку
//! и десериализацию данных, ничего не зная о бизнес-логике.

use std::error::Error;
use std::path::Path;

use crate::models::Order;

/// Загружает все заказы из CSV-файла.
pub fn load_orders<P: AsRef<Path>>(path: P) -> Result<Vec<Order>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut out = Vec::new();
    for result in rdr.deserialize() {
        let record: Order = result?;
        out.push(record);
    }
    Ok(out)
}

/// Фильтрует заказы через произвольный фильтр (Dependency Inversion).
pub fn filter_orders<'a, F>(orders: &'a [Order], filter: &F) -> Vec<&'a Order>
where
    F: crate::models::OrderFilter,
{
    orders.iter().filter(|o| filter.matches(o)).collect()
}
