//! Модуль моделей предметной области.
//!
//! Реализует принципы ООП (инкапсуляция данных в структурах, полиморфизм
//! через трейты) и SOLID (Single Responsibility — каждая структура отвечает
//! за одну сущность; Dependency Inversion — расчётные трейты абстрагируют
//! зависимости).

use serde::{Deserialize, Serialize};

/// Сырой заказ, прочитанный из CSV. Отвечает только за хранение строки данных.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Order {
    pub region: String,
    pub product: String,
    pub price: f64,
    pub cost: f64,
    pub units: f64,
}

impl Order {
    /// Прибыль одной строки заказа: (price - cost) * units.
    /// Метод инкапсулирован в саму сущность (ООП).
    #[inline]
    pub fn profit(&self) -> f64 {
        (self.price - self.cost) * self.units
    }
}

/// Заказ с предвычисленным столбцом profit — результат enrichment-фазы.
#[derive(Debug, Clone, PartialEq)]
pub struct EnrichedOrder {
    pub region: String,
    pub product: String,
    pub profit: f64,
}

impl From<&Order> for EnrichedOrder {
    fn from(o: &Order) -> Self {
        Self {
            region: o.region.clone(),
            product: o.product.clone(),
            profit: o.profit(),
        }
    }
}

/// Трейт-фильтр (Dependency Inversion: абстракция, а не конкретный тип).
/// Любой фильтр заказов реализует этот интерфейс.
pub trait OrderFilter {
    fn matches(&self, order: &Order) -> bool;
}

/// Конкретный фильтр по региону.
pub struct RegionFilter {
    pub region: String,
}

impl OrderFilter for RegionFilter {
    fn matches(&self, order: &Order) -> bool {
        order.region.eq_ignore_ascii_case(&self.region)
    }
}

/// Трейт-агрегатор прибыли (Single Responsibility + Dependency Inversion).
/// Реализация может быть последовательной или параллельной.
pub trait ProfitAggregator {
    /// Возвращает вектор (product, sum_profit), отсортированный по убыванию прибыли.
    fn aggregate(&self, orders: &[EnrichedOrder]) -> Vec<(String, f64)>;
}

/// Подсчёт агрегированной прибыли.
pub fn aggregate_profit(orders: &[EnrichedOrder]) -> Vec<(String, f64)> {
    use std::collections::HashMap;
    let mut map: HashMap<String, f64> = HashMap::new();
    for o in orders {
        *map.entry(o.product.clone()).or_insert(0.0) += o.profit;
    }
    let mut v: Vec<(String, f64)> = map.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}
