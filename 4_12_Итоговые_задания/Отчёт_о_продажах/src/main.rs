//! Главная точка входа.
//!
//! Демонстрирует: ООП (структуры + трейты), SOLID (разделение ответственности,
//! инверсия зависимостей через трейты), многопоточность (rayon), внешние крейты
//! (csv, serde, plotters, rayon), кастомные макросы и машинное обучение
//! (линейная регрессия градиентным спуском).

mod chart;
mod dataio;
mod macros;
mod ml;
mod models;
mod profit;

use std::io::{self, BufRead, Write};

use chart::{draw_bar_chart, Axes};
use ml::{standardize, LinearRegression, Regressor};
use models::{ProfitAggregator, RegionFilter};
use profit::{enrich_parallel, ParallelProfitAggregator, share_percent, total_profit};

// Кастомные макросы log!, print_summary!, top1! экспортированы через
// #[macro_export] из модуля macros и доступны в корневом пространстве имён.

fn main() {
    let stdout = io::stdout();
    let mut out = stdout.lock();

    // 1. Считывание региона из stdin.
    write!(out, "Region: ").unwrap();
    out.flush().unwrap();
    let stdin = io::stdin();
    let region = stdin
        .lock()
        .lines()
        .next()
        .unwrap_or(Ok(String::new()))
        .unwrap_or_default();
    let region = region.trim().to_string();
    if region.is_empty() {
        eprintln!("Region is empty");
        return;
    }

    // 2. Загрузка заказов (SOLID: ответственность за I/O — в модуле dataio).
    let all = match dataio::load_orders("sales.csv") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to read sales.csv: {e}");
            return;
        }
    };

    // 3. Фильтрация по региону (Dependency Inversion через трейт OrderFilter).
    let filter = RegionFilter {
        region: region.clone(),
    };
    let filtered: Vec<&models::Order> = dataio::filter_orders(&all, &filter);

    log!(
        info,
        "Filtered {} orders for region '{}'",
        filtered.len(),
        region
    );

    // 4. Расчёт прибыли по каждой строке (многопоточность rayon).
    let owned: Vec<models::Order> = filtered.into_iter().cloned().collect();
    let enriched = enrich_parallel(&owned);

    // 5. Агрегация прибыли по товарам (через трейт ProfitAggregator).
    let aggregator = ParallelProfitAggregator;
    let totals = aggregator.aggregate(&enriched);

    if totals.is_empty() {
        println!("No orders in {region}");
        return;
    }

    // 6. Итоговые метрики.
    let n_orders = owned.len();
    let total = total_profit(&totals);
    let (top_product, top_profit) = totals[0].clone();
    let share = share_percent(top_profit, total);

    // Две строки вывода (как в примере задачи).
    println!("Orders in {}: {}", region, n_orders);
    let profit_str = crate::macros::format_profit(top_profit);
    let share_str = crate::macros::format_share(share);
    print_summary!(region, top_product, profit_str, share_str);

    // 7. ML-модуль: обучим линейную регрессию (units -> profit) на строках региона.
    //    Демонстрация того, что ML-компонент реально работает.
    let x: Vec<f64> = owned.iter().map(|o| o.units).collect();
    let y: Vec<f64> = owned.iter().map(|o| o.profit()).collect();
    let (xs, mean_x, std_x) = standardize(&x);
    let mut model = LinearRegression::new(0.1, 200);
    model.fit(&xs, &y);
    let (w, b) = model.weights();
    log!(
        info,
        "ML linear regression (profit ~ units): w(std)={:.3} b={:.2} (mean_units={:.2}, std_units={:.2})",
        w,
        b,
        mean_x,
        std_x
    );

    // 8. Построение столбчатого графика (аналог fig, ax.bar).
    let categories: Vec<String> = totals.iter().map(|(p, _)| p.clone()).collect();
    let heights: Vec<f64> = totals.iter().map(|(_, v)| *v).collect();
    let axes = Axes::new(format!("Profit in {}", region))
        .xlabel("Product")
        .ylabel("Profit")
        .bars(categories.clone(), heights.clone());

    let chart_path = format!("profit_{}.png", region.to_lowercase());
    let bars_drawn = match draw_bar_chart(&chart_path, &axes) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Chart error: {e}");
            0
        }
    };

    // 9. Описание графика (как в примере вывода Coddy).
    println!("Charts drawn: 1");
    println!("Title: Profit in {}", region);
    println!("X label: Product");
    println!("Y label: Profit");
    println!("Bars: {}", bars_drawn);
    let cats_str = categories.join(", ");
    println!("Categories: {}", cats_str);
    // Heights: целые/десятичные как в примере (без trailing .0 для целых).
    let heights_str: Vec<String> = heights.iter().map(|h| crate::macros::format_profit(*h)).collect();
    println!("Heights: [{}]", heights_str.join(", "));
}
