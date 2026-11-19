//! # Анализ признаков breast_cancer: разделитель злокачественных образцов
//!
//! Воспроизводит эталонный Python-код:
//! ```text
//! for f in features:
//!     means = cancer.frame.groupby("target")[f].mean()
//!     ratio = means[0] / means[1]
//!     print(f"{f}: malignant is {round(ratio, 2)} times benign")
//! print(f"Best separator: {best}")
//! ```
//!
//! Демонстрирует запрошенные концепции:
//! * **ООП** — поведение инкапсулировано в типажах (traits) и реализующих их структурах.
//! * **SOLID** — `DatasetLoader`, `FeatureAnalyzer`, `Reporter` (Interface Segregation,
//!   Single Responsibility, Open/Closed, Dependency Inversion).
//! * **Многопоточность** — параллельный анализ признаков через крейт `rayon`.
//! * **Крейты** — `csv`, `serde`, `rayon`.
//! * **Макросы** — `round2!`, `ratio_line!`, `features!`.
//!
//! Соглашение датасета (как в `sklearn.datasets.load_breast_cancer`):
//! `target == 0` — злокачественный (malignant), `target == 1` — доброкачественный (benign).

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use csv::ReaderBuilder;
use rayon::prelude::*;

// ===========================================================================
// Макросы
// ===========================================================================

/// Округление до 2 знаков после запятой — аналог Python `round(x, 2)`.
///
/// Для значений этого датасета `f64::round` (округление половины вверх по модулю)
/// даёт тот же результат, что и банковское округление Python, т.к. третья цифра
/// в каждом случае >= 5.
macro_rules! round2 {
    ($x:expr) => {{
        let v: f64 = $x as f64;
        (v * 100.0).round() / 100.0
    }};
}

/// Формирует строку результата, идентичную Python f-string.
macro_rules! ratio_line {
    ($feature:expr, $ratio:expr) => {
        format!("{}: malignant is {} times benign", $feature, $ratio)
    };
}

/// Собирает `Vec<String>` из строковых литералов — удобно для тестов и значений
/// по умолчанию. Заодно демонстрирует декларативный макрос с повторением.
/// Используется в `#[cfg(test)]`, поэтому в основном билде помечен allow.
#[allow(unused_macros)]
macro_rules! features {
    ( $( $name:literal ),* $(,)? ) => {
        vec![ $( String::from($name), )* ]
    };
}

// ===========================================================================
// Доменные типы
// ===========================================================================

/// Одна запись датасета: имя признака -> числовое значение.
/// Колонка `target` тоже попадает сюда как `f64` (0.0 / 1.0).
type Record = HashMap<String, f64>;

/// Ошибка загрузки датасета.
#[derive(Debug)]
enum LoadError {
    Parse(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Parse(msg) => write!(f, "ошибка разбора CSV: {msg}"),
        }
    }
}

/// Результат анализа одного признака.
struct AnalysisResult {
    feature: String,
    ratio: Option<f64>,
}

// ===========================================================================
// SOLID — Interface Segregation + Dependency Inversion
// Загрузчик знает только, как получить записи; источник скрыт.
// ===========================================================================

/// Загрузчик датасета.
trait DatasetLoader {
    fn load(&self) -> Result<Vec<Record>, LoadError>;
}

/// Загрузчик из CSV, встроенного в бинарник на этапе компиляции (`include_str!`).
struct EmbeddedCsvLoader {
    csv_text: &'static str,
}

impl EmbeddedCsvLoader {
    fn new(csv_text: &'static str) -> Self {
        Self { csv_text }
    }
}

impl DatasetLoader for EmbeddedCsvLoader {
    fn load(&self) -> Result<Vec<Record>, LoadError> {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(self.csv_text.as_bytes());
        rdr.deserialize::<Record>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| LoadError::Parse(e.to_string()))
    }
}

// ===========================================================================
// SOLID — Single Responsibility
// Анализатор считает одну статистику по одному признаку: он не печатает,
// не загружает данные и не выбирает «лучший» признак.
// ===========================================================================

/// Анализатор признака.
///
/// `Send + Sync` требуется, чтобы ссылку на типаж-объект можно было
/// передавать в замыкания `rayon` (многопоточность).
trait FeatureAnalyzer: Send + Sync {
    /// Возвращает отношение mean(malignant) / mean(benign),
    /// либо `None`, если признака нет или доброкачественное среднее равно нулю.
    fn analyze(&self, feature: &str, records: &[Record]) -> Option<f64>;
}

/// Считает отношение средних: malignant / benign.
struct MalignancyRatioAnalyzer;

impl MalignancyRatioAnalyzer {
    /// Среднее значение признака `feature` по записям с `target == target_value`.
    fn mean(records: &[Record], feature: &str, target_value: f64) -> Option<f64> {
        let (sum, count) = records
            .iter()
            .filter(|r| r.get("target") == Some(&target_value))
            .filter_map(|r| r.get(feature).copied())
            .fold((0.0_f64, 0_usize), |(s, c), v| (s + v, c + 1));
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }
}

impl FeatureAnalyzer for MalignancyRatioAnalyzer {
    fn analyze(&self, feature: &str, records: &[Record]) -> Option<f64> {
        let malignant_mean = Self::mean(records, feature, 0.0)?;
        let benign_mean = Self::mean(records, feature, 1.0)?;
        if benign_mean == 0.0 {
            None
        } else {
            Some(malignant_mean / benign_mean)
        }
    }
}

// ===========================================================================
// SOLID — Single Responsibility (отдельно от анализа)
// Репортёр превращает результаты в текст. Замена на JSON/файловый репортёр —
// новая структура, не трогающая анализатор и загрузчик.
// ===========================================================================

/// Репортёр результатов.
trait Reporter: Send + Sync {
    fn report(&self, results: &[AnalysisResult]);
}

/// Вывод в stdout в формате эталонного Python-кода.
struct StdoutReporter;

impl Reporter for StdoutReporter {
    fn report(&self, results: &[AnalysisResult]) {
        let stdout = io::stdout();
        let mut out = stdout.lock();

        // Лучший разделитель — признак с наибольшим отношением.
        let best = results
            .iter()
            .filter_map(|r| r.ratio.map(|v| (v, r.feature.as_str())))
            .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for r in results {
            match r.ratio {
                Some(v) => writeln!(out, "{}", ratio_line!(r.feature, round2!(v))).unwrap(),
                None => writeln!(out, "{}: признак не найден в датасете", r.feature).unwrap(),
            }
        }

        match best {
            Some((_, name)) => writeln!(out, "Best separator: {}", name).unwrap(),
            None => writeln!(out, "Best separator: <нет данных>").unwrap(),
        }
    }
}

// ===========================================================================
// Оркестратор (Dependency Inversion): зависит от типажей, а не от конкретных типов.
// ===========================================================================

/// Приложение: связывает загрузчик, анализатор и репортёр.
struct BreastCancerApp<'a> {
    loader: &'a dyn DatasetLoader,
    analyzer: &'a dyn FeatureAnalyzer,
    reporter: &'a dyn Reporter,
}

impl<'a> BreastCancerApp<'a> {
    fn new(
        loader: &'a dyn DatasetLoader,
        analyzer: &'a dyn FeatureAnalyzer,
        reporter: &'a dyn Reporter,
    ) -> Self {
        Self { loader, analyzer, reporter }
    }

    fn run(&self, features: &[String]) -> Result<(), LoadError> {
        // Данные загружаются один раз и дальше разделяются между потоками
        // по неизменяемой ссылке (`&[Record]` — `Send + Sync`, т.к. `Record: Sync`).
        let records = self.loader.load()?;

        // Многопоточность: каждый признак анализируется параллельно через rayon.
        // `par_iter().map().collect()` сохраняет порядок входного вектора,
        // поэтому вывод идёт в порядке ввода.
        let results: Vec<AnalysisResult> = features
            .par_iter()
            .map(|f| AnalysisResult {
                feature: f.clone(),
                ratio: self.analyzer.analyze(f, &records),
            })
            .collect();

        self.reporter.report(&results);
        Ok(())
    }
}

// ===========================================================================
// Чтение ввода — повтор Python-цикла с input()
// ===========================================================================

fn read_features() -> io::Result<Vec<String>> {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    let n: usize = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "ожидается n"))??
        .trim()
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

    let mut features = Vec::with_capacity(n);
    for _ in 0..n {
        let line = lines
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "ожидается признак"))??
            .trim()
            .to_string();
        features.push(line);
    }
    Ok(features)
}

// ===========================================================================
// main
// ===========================================================================

fn main() {
    // Встраиваем датасет в бинарник на этапе компиляции — бинарник самодостаточен.
    const CSV: &str = include_str!("../data/breast_cancer.csv");

    let loader = EmbeddedCsvLoader::new(CSV);
    let analyzer = MalignancyRatioAnalyzer;
    let reporter = StdoutReporter;
    let app = BreastCancerApp::new(&loader, &analyzer, &reporter);

    let features = match read_features() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ошибка ввода: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = app.run(&features) {
        eprintln!("ошибка загрузки датасета: {e}");
        std::process::exit(2);
    }
}

// ===========================================================================
// Тесты
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const CSV: &str = include_str!("../data/breast_cancer.csv");

    /// Эталонные значения из условия задачи.
    #[test]
    fn worst_radius_is_1_58() {
        let analyzer = MalignancyRatioAnalyzer;
        let loader = EmbeddedCsvLoader::new(CSV);
        let records = loader.load().unwrap();
        let ratio = analyzer.analyze("worst radius", &records).unwrap();
        assert_eq!(format!("{}", round2!(ratio)), "1.58");
    }

    #[test]
    fn worst_texture_is_1_25() {
        let analyzer = MalignancyRatioAnalyzer;
        let loader = EmbeddedCsvLoader::new(CSV);
        let records = loader.load().unwrap();
        let ratio = analyzer.analyze("worst texture", &records).unwrap();
        assert_eq!(format!("{}", round2!(ratio)), "1.25");
    }

    /// Лучший разделитель среди двух признаков — worst radius.
    #[test]
    fn best_separator_is_worst_radius() {
        let a = MalignancyRatioAnalyzer;
        let loader = EmbeddedCsvLoader::new(CSV);
        let records = loader.load().unwrap();
        let feats = features!("worst radius", "worst texture");
        let mut best = "";
        let mut best_ratio = 0.0_f64;
        for f in &feats {
            let r = a.analyze(f, &records).unwrap();
            if r > best_ratio {
                best_ratio = r;
                best = f;
            }
        }
        assert_eq!(best, "worst radius");
    }

    /// Несуществующий признак даёт None.
    #[test]
    fn unknown_feature_returns_none() {
        let a = MalignancyRatioAnalyzer;
        let loader = EmbeddedCsvLoader::new(CSV);
        let records = loader.load().unwrap();
        assert!(a.analyze("no such feature", &records).is_none());
    }
}
