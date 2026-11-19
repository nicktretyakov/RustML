//! Модуль расчёта прибыли.
//!
//! SOLID:
//! - Single Responsibility: агрегирует прибыль по продуктам.
//! - Open/Closed: параллельный агрегатор — отдельная структура; можно добавить
//!   новые реализации, не трогая существующие.
//! - Dependency Inversion: реализует трейт `ProfitAggregator`.
//!
//! Многопоточность: используется крейт `rayon` для параллельного расчёта
//! прибыли по строкам заказов (большой объём данных обрабатывается в потоках).

use rayon::prelude::*;
use std::collections::HashMap;

use crate::models::{EnrichedOrder, Order, ProfitAggregator};

/// Преобразует заказы в enriched-форму параллельно (многопоточность rayon).
pub fn enrich_parallel(orders: &[Order]) -> Vec<EnrichedOrder> {
    orders
        .par_iter()
        .map(EnrichedOrder::from)
        .collect()
}

/// Параллельный агрегатор прибыли: каждая строка обрабатывается в потоках,
/// затем с помощью reduce-фазы (через HashMap) объединяется.
pub struct ParallelProfitAggregator;

impl ProfitAggregator for ParallelProfitAggregator {
    fn aggregate(&self, orders: &[EnrichedOrder]) -> Vec<(String, f64)> {
        // Многопоточный fold: каждый поток собирает локальный HashMap.
        let fold = || HashMap::<String, f64>::new();
        let combine = |mut a: HashMap<String, f64>, b: HashMap<String, f64>| {
            for (k, v) in b {
                *a.entry(k).or_insert(0.0) += v;
            }
            a
        };

        let map = orders
            .par_iter()
            .fold(fold, |mut acc, o| {
                *acc.entry(o.product.clone()).or_insert(0.0) += o.profit;
                acc
            })
            .reduce(fold, combine);

        let mut v: Vec<(String, f64)> = map.into_iter().collect();
        // Сортировка по убыванию прибыли (как в pandas sort_values(ascending=False)).
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }
}

/// Вычисляет общую прибыль по всем товарам.
pub fn total_profit(totals: &[(String, f64)]) -> f64 {
    totals.iter().map(|(_, p)| p).sum()
}

/// Вычисляет долю в процентах, округлённую до 1 знака после запятой.
pub fn share_percent(part: f64, total: f64) -> f64 {
    if total.abs() < f64::EPSILON {
        0.0
    } else {
        ((part / total) * 100.0 * 10.0).round() / 10.0
    }
}
