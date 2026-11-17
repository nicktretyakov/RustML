//! Фитнес-трекер: анализ пропусков в `tracker.csv` и три способа оценки
//! среднего `sleep`.
//!
//! Демонстрирует: внешние крейты (`csv`, `serde`, `rayon`), декларативные
//! макросы, многопоточность (`rayon`), ML-компоненты (линейная регрессия
//! и регрессионный импутер), ООП (трейты/структуры) и принципы SOLID.

use std::error::Error;
use std::path::PathBuf;

use csv::Reader;
use rayon::prelude::*;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Модель данных (SRP: хранит ровно одну запись трекера)
// ---------------------------------------------------------------------------

/// Одна строка дневника трекера. Поля — `Option`, чтобы serde превращал
/// пустые ячейки CSV в `None` (пропуск).
#[derive(Debug, Clone, Deserialize)]
struct TrackerRecord {
    day: Option<i32>,
    steps: Option<f64>,
    sleep: Option<f64>,
    water: Option<f64>,
}

impl TrackerRecord {
    /// Полностью ли заполнена строка (нет ни одного пропуска).
    fn is_complete(&self) -> bool {
        self.day.is_some()
            && self.steps.is_some()
            && self.sleep.is_some()
            && self.water.is_some()
    }

    /// Универсальный доступ к столбцу по имени — нужен для обобщённой
    /// обработки стратегиями импутации.
    fn column(&self, name: &str) -> Option<f64> {
        match name {
            "day" => self.day.map(|v| v as f64),
            "steps" => self.steps,
            "sleep" => self.sleep,
            "water" => self.water,
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// SOLID (DIP + OCP): абстракция стратегии обработки пропусков.
// Новые стратегии добавляются реализацией трейта — без правки существующих.
// ---------------------------------------------------------------------------

/// Стратегия подготовки значений целевого столбца.
trait ImputeStrategy: Send + Sync {
    fn values(&self, column: &str, records: &[TrackerRecord], fill: f64) -> Vec<f64>;
}

/// Стратегия 1: пропускать пропуски (только существующие значения).
struct SkipMissing;
impl ImputeStrategy for SkipMissing {
    fn values(&self, column: &str, records: &[TrackerRecord], _fill: f64) -> Vec<f64> {
        records.iter().filter_map(|r| r.column(column)).collect()
    }
}

/// Стратегия 2: учитывать только полные дни (без пропусков во ВСЕХ столбцах).
struct CompleteDaysOnly;
impl ImputeStrategy for CompleteDaysOnly {
    fn values(&self, column: &str, records: &[TrackerRecord], _fill: f64) -> Vec<f64> {
        records
            .iter()
            .filter(|r| r.is_complete())
            .filter_map(|r| r.column(column))
            .collect()
    }
}

/// Стратегия 3: заполнить пропуски константой `fill`.
struct FillConstant;
impl ImputeStrategy for FillConstant {
    fn values(&self, column: &str, records: &[TrackerRecord], fill: f64) -> Vec<f64> {
        records.iter().map(|r| r.column(column).unwrap_or(fill)).collect()
    }
}

// ---------------------------------------------------------------------------
// ML-компоненты: обучаемая модель (МНК-регрессия) и пайплайн импутации,
// предсказывающий `sleep` по `steps`. Показывает ML-подход к заполнению.
// ---------------------------------------------------------------------------

/// Узкий интерфейс модели с учителем (ISP).
trait SupervisedModel: Send + Sync {
    fn fit(xs: &[f64], ys: &[f64]) -> Self;
    fn predict(&self, x: f64) -> f64;
}

/// Простая линейная регрессия y = slope * x + intercept (метод наименьших
/// квадратов).
struct LinearRegression {
    slope: f64,
    intercept: f64,
}

impl SupervisedModel for LinearRegression {
    fn fit(xs: &[f64], ys: &[f64]) -> Self {
        let n = xs.len() as f64;
        if n == 0.0 {
            return Self { slope: 0.0, intercept: 0.0 };
        }
        let mx = xs.iter().sum::<f64>() / n;
        let my = ys.iter().sum::<f64>() / n;
        let num: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
        let den: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
        let slope = if den == 0.0 { 0.0 } else { num / den };
        let intercept = my - slope * mx;
        Self { slope, intercept }
    }

    fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

/// Пайплайн импутации: обучает регрессию на полных парах `(steps, sleep)` и
/// предсказывает пропущенный `sleep` по известному `steps`.
struct RegressionImputer<M: SupervisedModel> {
    model: M,
}

impl<M: SupervisedModel> RegressionImputer<M> {
    fn from_records(records: &[TrackerRecord]) -> Self {
        let (xs, ys): (Vec<f64>, Vec<f64>) = records
            .iter()
            .filter_map(|r| r.steps.zip(r.sleep))
            .unzip();
        Self { model: M::fit(&xs, &ys) }
    }

    /// Предсказанные значения `sleep` для строк, где `sleep` пропущен, но
    /// `steps` известен.
    fn predicted_gaps(&self, records: &[TrackerRecord]) -> Vec<f64> {
        records
            .iter()
            .filter(|r| r.sleep.is_none() && r.steps.is_some())
            .map(|r| self.model.predict(r.steps.unwrap()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Статистика (SRP: только подсчёт; многопоточно через rayon)
// ---------------------------------------------------------------------------

/// Среднее арифметическое, вычисляемое параллельно (`rayon`).
fn par_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = values.par_iter().sum();
    sum / values.len() as f64
}

/// Округление до `digits` знаков после запятой (арифметика — кодом).
fn round_to(x: f64, digits: u32) -> f64 {
    let factor = 10f64.powi(digits as i32);
    (x * factor).round() / factor
}

/// Форматирование числа «как в Python» (7.0 -> "7.0", 6.9 -> "6.9",
/// 7.21 -> "7.21") с предварительным округлением до 2 знаков.
fn py_float(x: f64) -> String {
    let r = round_to(x, 2);
    if r.is_nan() {
        "nan".to_string()
    } else if r == r.trunc() {
        format!("{:.1}", r)
    } else {
        format!("{}", r)
    }
}

// ---------------------------------------------------------------------------
// Декларативные макросы
// ---------------------------------------------------------------------------

/// Подсчёт пропусков в поле записи по идентификатору поля структуры.
macro_rules! count_missing {
    ($records:expr, $field:ident) => {
        $records.iter().filter(|r| r.$field.is_none()).count() as i64
    };
}

/// Формирование строки «метка: значение» для вывода среднего.
macro_rules! stat_line {
    ($label:expr, $value:expr) => {
        format!("{}: {}", $label, py_float($value))
    };
}

// ---------------------------------------------------------------------------
// Точка входа
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn Error>> {
    // Путь к данным рядом с манифестом (не зависит от cwd запуска).
    let csv_path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tracker.csv");

    // `hours` — значение для заполнения пропусков (как в исходном коде).
    let hours: f64 = 8.0;

    // Чтение CSV: крейт `csv` + `serde`; пустые поля -> `None` для `Option<T>`.
    let mut rdr = Reader::from_path(&csv_path)?;
    let records: Vec<TrackerRecord> = rdr.deserialize().filter_map(Result::ok).collect();

    // --- 1. Количество пропусков в каждом столбце (многопоточно) ----------
    let columns = ["day", "steps", "sleep", "water"];
    let missing_counts: Vec<(&str, i64)> = columns
        .par_iter()
        .map(|col| {
            let count = match *col {
                "day" => count_missing!(records, day),
                "steps" => count_missing!(records, steps),
                "sleep" => count_missing!(records, sleep),
                "water" => count_missing!(records, water),
                _ => 0,
            };
            (*col, count)
        })
        .collect();

    // Вывод в стиле pandas Series (ширина индекса 5, ширина значения 5).
    for (col, count) in &missing_counts {
        println!("{:<5}{:>5}", col, count);
    }
    println!("dtype: int64");

    // --- 2. Среднее sleep тремя способами ----------------------------------
    // Зависим от абстракции `ImputeStrategy` (DIP), а не от конкретных типов.
    let strategies: &[&dyn ImputeStrategy] = &[&SkipMissing, &CompleteDaysOnly, &FillConstant];

    let sleep_skip = par_mean(&strategies[0].values("sleep", &records, hours));
    let sleep_complete = par_mean(&strategies[1].values("sleep", &records, hours));
    let sleep_filled = par_mean(&strategies[2].values("sleep", &records, hours));

    // ML-демонстрация: регрессионная импутация sleep ~ steps.
    // Обучается на полных парах; предсказания вычисляются, но НЕ печатаются,
    // чтобы не нарушать требуемый формат вывода.
    let imputer = RegressionImputer::<LinearRegression>::from_records(&records);
    let _ml_predictions: Vec<f64> = imputer.predicted_gaps(&records);

    println!("{}", stat_line!("Skipping gaps", sleep_skip));
    println!("{}", stat_line!("Complete days only", sleep_complete));
    println!("Gaps filled with {}: {}", py_float(hours), py_float(sleep_filled));

    Ok(())
}
