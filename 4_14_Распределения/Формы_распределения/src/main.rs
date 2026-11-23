use std::error::Error;
use rayon::prelude::*;

// ------------------------------------------------------------
// 1. Абстракция (Interface Segregation + Dependency Inversion)
// ------------------------------------------------------------
trait DescriptiveStats {
    fn mean(&self, data: &[f64]) -> f64;
    fn median(&self, data: &[f64]) -> f64;
}

// ------------------------------------------------------------
// 2. Конкретная реализация (Single Responsibility)
// ------------------------------------------------------------
/// Ручная реализация: среднее и медиана без внешних крейтов.
struct ManualAnalyzer;

impl DescriptiveStats for ManualAnalyzer {
    fn mean(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return f64::NAN;
        }
        let sum: f64 = data.iter().sum();
        sum / data.len() as f64
    }

    fn median(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return f64::NAN;
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }
}

// ------------------------------------------------------------
// 3. Макрос для классификации асимметрии (Open/Closed)
// ------------------------------------------------------------
/// Возвращает кортеж (связка, метка), чтобы корректно
/// печатать "above", "below" или "close to" в зависимости
/// от знака отклонения среднего от медианы.
macro_rules! skew_info {
    ($mean:expr, $median:expr) => {{
        let diff = $mean - $median;
        if diff > 2.0 {
            ("above", "right-skewed")
        } else if diff < -2.0 {
            ("below", "left-skewed")
        } else {
            ("close to", "symmetric")
        }
    }};
}

// ------------------------------------------------------------
// 4. Чтение CSV (Single Responsibility)
// ------------------------------------------------------------
fn read_numeric_columns(path: &str) -> Result<Vec<(String, Vec<f64>)>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let n_cols = headers.len();

    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); n_cols];

    for result in rdr.records() {
        let record = result?;
        for (i, field) in record.iter().enumerate() {
            let value = field
                .parse::<f64>()
                .map_err(|_| format!("Non-numeric value '{}' in column '{}'", field, &headers[i]))?;
            columns[i].push(value);
        }
    }

    let named_columns = headers
        .iter()
        .zip(columns)
        .map(|(h, c)| (h.to_string(), c))
        .collect();

    Ok(named_columns)
}

// ------------------------------------------------------------
// 5. Главная функция (Dependency Inversion: analyzer — абстракция)
// ------------------------------------------------------------
fn main() -> Result<(), Box<dyn Error>> {
    let columns = read_numeric_columns("players.csv")?;
    let analyzer = ManualAnalyzer;

    // Параллельная обработка столбцов (rayon сохраняет порядок)
    let reports: Vec<String> = columns
        .par_iter()
        .map(|(name, data)| {
            let mean = analyzer.mean(data);
            let median = analyzer.median(data);
            let (relation, label) = skew_info!(mean, median);
            format!(
                "{}: mean {:.1} {} median {:.1}, {}",
                name, mean, relation, median, label
            )
        })
        .collect();

    for report in reports {
        println!("{}", report);
    }

    Ok(())
}
