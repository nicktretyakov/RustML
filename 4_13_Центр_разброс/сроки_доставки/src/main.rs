//! Отчёт по доставкам одного ресторана для приложения доставки.
//!
//! Порт Python-скрипта (pandas + numpy) на Rust с использованием:
//! - макросов: `serde::Deserialize` (derive-макрос) для модели данных;
//! - многопоточности: `rayon` — параллельная сортировка, параллельные суммы,
//!   параллельный расчёт по всем ресторанам;
//! - ООП + SOLID: поведение разнесено по типажам (`DeliveryRepository`,
//!   `DeliveryFilter`, `StatisticsEngine`, `Reporter`, `MlAnalyzer`), а
//!   оркестратор `ReportService` зависит от абстракций (типажей-объектов);
//! - машинного обучения: `smartcore` KMeans — сегментация доставок ресторана
//!   на «быстрые» и «медленные» кластеры.
//!
//! Принципы SOLID в архитектуре:
//! - SRP — у каждой сущности одна обязанность (модель, загрузка, фильтр,
//!   статистика, вывод, ML, оркестрация).
//! - OCP — новые фильтры/статистики/репортёры добавляются реализацией типажа,
//!   без правки `ReportService`.
//! - LSP — любая реализация `DeliveryRepository` взаимозаменяема.
//! - ISP — типажи минимальны (`DeliveryFilter` содержит только `matches`).
//! - DIP — `ReportService` зависит от типажей-объектов, а не от конкретных типов.

use std::collections::HashMap;
use std::error::Error;
use std::io::{self, BufRead};
use std::path::Path;

use rayon::prelude::*;
use serde::Deserialize;
use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;

// =========================================================================
//  Модель данных (макрос serde::Deserialize)
// =========================================================================

/// Одна строка deliveries.csv: restaurant, driver, minutes.
/// Поле `minutes` — `f64`, чтобы корректно читать дробные значения.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Delivery {
    pub restaurant: String,
    pub driver: String,
    pub minutes: f64,
}

// =========================================================================
//  Типаж 1: Repository  (SRP — только загрузка; DIP — абстракция для оркестратора)
// =========================================================================

pub trait DeliveryRepository {
    fn load_all(&self) -> Result<Vec<Delivery>, Box<dyn Error>>;
}

/// Конкретная реализация: чтение CSV из файла.
pub struct CsvDeliveryRepository<P: AsRef<Path>> {
    path: P,
}

impl<P: AsRef<Path>> CsvDeliveryRepository<P> {
    pub fn new(path: P) -> Self {
        Self { path }
    }
}

impl<P: AsRef<Path>> DeliveryRepository for CsvDeliveryRepository<P> {
    fn load_all(&self) -> Result<Vec<Delivery>, Box<dyn Error>> {
        let mut rdr = csv::Reader::from_path(&self.path)?;
        let mut out = Vec::new();
        for record in rdr.deserialize() {
            out.push(record?);
        }
        Ok(out)
    }
}

// =========================================================================
//  Типаж 2: Filter  (ISP — единственный метод; OCP — новые фильтры без правок)
// =========================================================================

pub trait DeliveryFilter {
    fn matches(&self, d: &Delivery) -> bool;
}

/// Выбор доставок конкретного ресторана.
pub struct RestaurantFilter {
    pub name: String,
}

impl DeliveryFilter for RestaurantFilter {
    fn matches(&self, d: &Delivery) -> bool {
        d.restaurant == self.name
    }
}

// =========================================================================
//  Типаж 3: StatisticsEngine  (SRP — только расчёт статистик)
// =========================================================================

#[derive(Debug, Clone)]
pub struct Stats {
    pub count: usize,
    pub median: f64,
    pub mean: f64,
    pub iqr: f64,
    pub std: f64,
    pub top_driver: String,
}

pub trait StatisticsEngine {
    fn compute(&self, deliveries: &[Delivery]) -> Option<Stats>;
}

/// Статистики, совместимые с pandas/numpy:
/// - медиана и квантили — линейная интерполяция (метод 'linear' numpy);
/// - СКО — выборочное (ddof = 1), как `pandas.Series.std()`;
/// - топ-водитель — мода, как `pandas.Series.mode()[0]` (среди значений с
///   максимальной частотой берётся лексикографически наименьшее — так же,
///   как сортирует `Series.mode()`).
pub struct PandasStatistics;

impl StatisticsEngine for PandasStatistics {
    fn compute(&self, deliveries: &[Delivery]) -> Option<Stats> {
        if deliveries.is_empty() {
            return None;
        }
        // Многопоточная сортировка минут (rayon).
        let mut minutes: Vec<f64> = deliveries.iter().map(|d| d.minutes).collect();
        minutes.par_sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = minutes.len();
        let median = quantile(&minutes, 0.5);
        // Параллельная сумма для среднего.
        let mean = minutes.par_iter().sum::<f64>() / count as f64;
        let iqr = quantile(&minutes, 0.75) - quantile(&minutes, 0.25);

        // Выборочное СКО (ddof = 1): делим на (n - 1).
        let std = if count > 1 {
            let sq_sum: f64 = minutes.par_iter().map(|m| (m - mean).powi(2)).sum();
            (sq_sum / (count - 1) as f64).sqrt()
        } else {
            0.0
        };

        let driver_refs: Vec<&str> = deliveries.iter().map(|d| d.driver.as_str()).collect();
        let top_driver = mode(&driver_refs);

        Some(Stats {
            count,
            median,
            mean,
            iqr,
            std,
            top_driver,
        })
    }
}

/// Квантиль методом линейной интерполяции — повторяет `numpy.percentile`
/// с методом по умолчанию ('linear'): h = (n - 1) * q, интерполяция между
/// соседними элементами отсортированного массива.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let h = (n - 1) as f64 * q;
    let lower = h.floor() as usize;
    let upper = h.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let frac = h - lower as f64;
    sorted[lower] + frac * (sorted[upper] - sorted[lower])
}

/// Мода как у `pandas.Series.mode()[0]`: среди значений с максимальной частотой
/// возвращается лексикографически наименьшее (так как `Series.mode()` сортирует
/// результат по возрастанию, а `[0]` берёт первый).
fn mode(values: &[&str]) -> String {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for v in values {
        *counts.entry(v).or_insert(0) += 1;
    }
    let max_count = *counts.values().max().unwrap_or(&0);
    let mut modes: Vec<&str> = counts
        .iter()
        .filter(|(_, c)| **c == max_count)
        .map(|(k, _)| *k)
        .collect();
    modes.sort();
    modes.first().map(|s| s.to_string()).unwrap_or_default()
}

/// Округление «к ближайшему чётному» (banker's rounding), как `numpy.round`.
fn round_half_to_even(x: f64, decimals: u32) -> f64 {
    let factor = 10f64.powi(decimals as i32);
    let scaled = x * factor;
    let half_away = scaled.round(); // Rust: половина — от нуля
    let diff = (scaled - half_away).abs();
    if diff == 0.5 {
        // Случай ровно 0.5: выбираем чётного соседа.
        let floor = scaled.floor();
        let even = if floor % 2.0 == 0.0 {
            floor
        } else {
            floor + 1.0
        };
        even / factor
    } else {
        half_away / factor
    }
}

/// Форматирование вещественного числа «как Python» (кратчайшее round-trip
/// представление, обязательно с десятичной точкой). `format!("{:?}", f64)`
/// даёт именно это, совпадая со `str(float)` в Python: 17.5 -> "17.5",
/// 19.2 -> "19.2", 5.81 -> "5.81".
fn py_float(x: f64) -> String {
    format!("{:?}", x)
}

// =========================================================================
//  Типаж 4: Reporter  (SRP — только вывод)
// =========================================================================

pub trait Reporter {
    fn report(&self, name: &str, stats: &Stats);
}

/// Вывод в stdout, повторяющий формат Python-скрипта.
pub struct ConsoleReporter;

impl Reporter for ConsoleReporter {
    fn report(&self, name: &str, stats: &Stats) {
        let mean_r = round_half_to_even(stats.mean, 1);
        let iqr_r = round_half_to_even(stats.iqr, 1);
        let std_r = round_half_to_even(stats.std, 2);

        println!("{}: {} deliveries", name, stats.count);
        println!(
            "Usual time: {} min (median), mean {}",
            py_float(stats.median),
            py_float(mean_r)
        );
        println!("Middle half spans {} min (IQR)", py_float(iqr_r));
        println!("Std: {} min", py_float(std_r));
        println!("Top driver: {}", stats.top_driver);
    }
}

// =========================================================================
//  Типаж 5: MlAnalyzer  (машинное обучение; SRP — только ML-анализ)
// =========================================================================

#[derive(Debug, Clone)]
pub struct ClusteringReport {
    pub fast_center: f64,
    pub fast_count: usize,
    pub slow_center: f64,
    pub slow_count: usize,
}

pub trait MlAnalyzer {
    fn analyze(&self, deliveries: &[Delivery]) -> Option<ClusteringReport>;
}

/// Кластеризация KMeans (smartcore) на одномерном признаке `minutes`
/// с k = 2: доставкам присваиваются метки кластеров, центры считаются как
/// среднее по точкам кластера. Кластер с меньшим центром — «быстрый».
pub struct KMeansAnalyzer {
    pub k: usize,
}

impl MlAnalyzer for KMeansAnalyzer {
    fn analyze(&self, deliveries: &[Delivery]) -> Option<ClusteringReport> {
        let n = deliveries.len();
        if n < self.k || self.k < 2 {
            return None;
        }

        // Матрица n x 1.
        let rows: Vec<Vec<f64>> = deliveries.iter().map(|d| vec![d.minutes]).collect();
        let slices: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        let x = DenseMatrix::from_2d_array(&slices);

        let model = KMeans::fit(&x, KMeansParameters::default().with_k(self.k)).ok()?;
        let labels: Vec<u8> = model.predict(&x).ok()?;

        // Центры кластеров = среднее минут по точкам каждого кластера.
        let mut sums: HashMap<u8, (f64, usize)> = HashMap::new();
        for (i, &lbl) in labels.iter().enumerate() {
            let entry = sums.entry(lbl).or_insert((0.0, 0));
            entry.0 += deliveries[i].minutes;
            entry.1 += 1;
        }
        let mut centers: Vec<(f64, usize)> =
            sums.values().map(|(s, c)| (*s / *c as f64, *c)).collect();
        if centers.len() < 2 {
            return None;
        }
        centers.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        Some(ClusteringReport {
            fast_center: centers[0].0,
            fast_count: centers[0].1,
            slow_center: centers[1].0,
            slow_count: centers[1].1,
        })
    }
}

// =========================================================================
//  Оркестратор (DIP — зависит только от типажей-объектов)
// =========================================================================

pub struct ReportService<'a> {
    repo: &'a dyn DeliveryRepository,
    filter: &'a dyn DeliveryFilter,
    stats: &'a dyn StatisticsEngine,
    reporter: &'a dyn Reporter,
    ml: &'a dyn MlAnalyzer,
}

impl<'a> ReportService<'a> {
    pub fn new(
        repo: &'a dyn DeliveryRepository,
        filter: &'a dyn DeliveryFilter,
        stats: &'a dyn StatisticsEngine,
        reporter: &'a dyn Reporter,
        ml: &'a dyn MlAnalyzer,
    ) -> Self {
        Self {
            repo,
            filter,
            stats,
            reporter,
            ml,
        }
    }

    pub fn run(&self, name: &str) -> Result<(), Box<dyn Error>> {
        let all = self.repo.load_all()?;

        let filtered: Vec<Delivery> = all
            .iter()
            .filter(|d| self.filter.matches(d))
            .cloned()
            .collect();
        if filtered.is_empty() {
            println!("No deliveries found for restaurant '{}'.", name);
            return Ok(());
        }

        // Основной отчёт (5 строк, как в Python-скрипте).
        let stats = self.stats.compute(&filtered).ok_or("no statistics")?;
        self.reporter.report(name, &stats);

        // Дополнительный ML-раздел: контекст + кластеризация.
        println!();
        println!("--- ML analysis (smartcore KMeans, k = 2) ---");
        if let Some((rank, total)) = parallel_rank_among_restaurants(&all, name) {
            println!(
                "Context: among {} restaurants, {} ranks #{} by mean delivery time (1 = fastest).",
                total, name, rank
            );
        }
        match self.ml.analyze(&filtered) {
            Some(cl) => {
                println!(
                    "Fast cluster center: {} min ({} deliveries)",
                    py_float(round_half_to_even(cl.fast_center, 1)),
                    cl.fast_count
                );
                println!(
                    "Slow cluster center: {} min ({} deliveries)",
                    py_float(round_half_to_even(cl.slow_center, 1)),
                    cl.slow_count
                );
            }
            None => println!("Clustering not available for this restaurant (too few deliveries)."),
        }

        Ok(())
    }
}

/// Параллельно (rayon) считает среднее время доставки для каждого ресторана
/// и возвращает ранг запрошенного ресторана по возрастанию среднего времени
/// (1 = самый быстрый). Демонстрирует многопоточность «по ресторанам».
fn parallel_rank_among_restaurants(all: &[Delivery], target: &str) -> Option<(usize, usize)> {
    let names: Vec<String> = {
        let mut set: Vec<String> = Vec::new();
        for d in all {
            if !set.iter().any(|n| n == &d.restaurant) {
                set.push(d.restaurant.clone());
            }
        }
        set
    };
    let total = names.len();

    // Параллельный расчёт среднего по каждому ресторану.
    let means: Vec<(String, f64)> = names
        .par_iter()
        .map(|n| {
            let vals: Vec<f64> = all
                .iter()
                .filter(|d| &d.restaurant == n)
                .map(|d| d.minutes)
                .collect();
            let m = vals.iter().sum::<f64>() / vals.len() as f64;
            (n.clone(), m)
        })
        .collect();

    let mut sorted = means;
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let rank = sorted
        .iter()
        .position(|(n, _)| n == target)
        .map(|i| i + 1)?;
    Some((rank, total))
}

// =========================================================================
//  main
// =========================================================================

fn main() {
    // Чтение названия ресторана из stdin — аналог Python input().
    let name: String = io::stdin()
        .lock()
        .lines()
        .next()
        .and_then(|r| r.ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    if name.is_empty() {
        eprintln!("Usage: echo \"<restaurant>\" | delivery_report");
        std::process::exit(1);
    }

    let csv_path = std::env::var("DELIVERIES_CSV").unwrap_or_else(|_| "deliveries.csv".to_string());

    // Конкретные реализации (можно подставлять любые другие, не трогая сервис).
    let repo = CsvDeliveryRepository::new(csv_path);
    let filter = RestaurantFilter { name: name.clone() };
    let stats = PandasStatistics;
    let reporter = ConsoleReporter;
    let ml = KMeansAnalyzer { k: 2 };

    let service = ReportService::new(&repo, &filter, &stats, &reporter, &ml);
    if let Err(e) = service.run(&name) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
