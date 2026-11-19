//! Модуль загрузки набора данных.
//!
//! SRP: единственная ответственность — чтение и хранение данных.
//! ISP: узкий трейт `DataSource` для доступа к данным, реализуемый
//! конкретным классом `CsvDataset` (ООП: поля приватны, доступ через методы).

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, anyhow};

/// Встроенный CSV-датасет breast_cancer (скомпилирован в бинарник через include_str!).
/// Гарантирует работоспособность программы даже без внешнего файла данных.
///
/// Файл `breast_cancer.csv` лежит в корне crate (рядом с `Cargo.toml` и `src/`),
/// поэтому из `src/dataset.rs` путь — `../breast_cancer.csv`.
const EMBEDDED_CSV: &str = include_str!("../breast_cancer.csv");

/// Абстракция источника данных (Interface Segregation Principle).
/// Любой источник (CSV, БД, linfa-датасет) реализует этот трейт.
pub trait DataSource {
    /// Названия признаков в исходном порядке.
    fn feature_names(&self) -> &[String];
    /// Названия классов (например, ["malignant", "benign"]).
    fn class_names(&self) -> &[String];
    /// Количество образцов (строк).
    fn n_samples(&self) -> usize;
    /// Значения конкретного признака по индексу.
    fn feature_column(&self, index: usize) -> Result<&[f64]>;
}

/// Конкретная реализация источника данных из CSV-файла.
///
/// Инкапсулирует:
/// - заголовок (названия признаков),
/// - матрицу признаков (строка = образец),
/// - метки классов,
/// - названия классов.
pub struct CsvDataset {
    feature_names: Vec<String>,
    class_names: Vec<String>,
    features: Vec<Vec<f64>>, // [n_samples][n_features]
}

impl CsvDataset {
    /// Создаёт датасет из CSV-файла.
    /// Формат: первая строка — заголовок, последняя колонка — `target` (0/1).
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)
            .with_context(|| format!("Не удалось открыть файл: {}", path_ref.display()))?;
        Self::from_reader(BufReader::new(file))
    }

    /// Создаёт датасет из встроенной строки CSV (скомпилированной в бинарник).
    /// Используется как fallback, когда внешний файл недоступен.
    pub fn from_embedded() -> Result<Self> {
        Self::from_reader(std::io::Cursor::new(EMBEDDED_CSV))
    }

    /// Универсальный конструктор: парсит CSV из любого `io::Read`.
    /// Формат: первая строка — заголовок, последняя колонка — `target` (0/1).
    pub fn from_reader<R: Read>(reader: R) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(false)
            .from_reader(reader);

        let headers = reader
            .headers()
            .context("Не удалось прочитать заголовок CSV")?;
        let header_vec: Vec<String> = headers.iter().map(String::from).collect();

        if header_vec.is_empty() {
            return Err(anyhow!("CSV-файл не содержит заголовков"));
        }
        if header_vec.last().map(|c| c.as_str()) != Some("target") {
            return Err(anyhow!(
                "Последний столбец должен называться 'target', найдено: {:?}",
                header_vec.last()
            ));
        }

        let feature_names: Vec<String> = header_vec[..header_vec.len() - 1].to_vec();
        let n_features = feature_names.len();

        // Читаем все записи в Vec<Vec<f64>>.
        let mut features: Vec<Vec<f64>> = Vec::new();
        // Классы: 0 -> malignant, 1 -> benign (по соглашению scikit-learn).
        let mut class_names_sorted: Vec<String> = Vec::new();

        for (row_idx, result) in reader.records().enumerate() {
            let record = result.with_context(|| format!("Ошибка чтения строки {row_idx}"))?;
            if record.len() != n_features + 1 {
                return Err(anyhow!(
                    "Строка {row_idx}: ожидалось {} значений, получено {}",
                    n_features + 1,
                    record.len()
                ));
            }
            let mut row = Vec::with_capacity(n_features);
            for (j, cell) in record.iter().take(n_features).enumerate() {
                let v: f64 = cell.parse().with_context(|| {
                    format!("Строка {row_idx}, признак '{j}': не число '{cell}'")
                })?;
                row.push(v);
            }
            // Целевая колонка: определяем множество классов.
            let target_str = record
                .get(n_features)
                .ok_or_else(|| anyhow!("Строка {row_idx}: отсутствует колонка target"))?;
            let _target: u32 = target_str
                .parse()
                .with_context(|| format!("Строка {row_idx}: target не целое '{target_str}'"))?;
            features.push(row);

            // Формируем уникальный список имён классов.
            let label = match target_str {
                "0" => "malignant",
                "1" => "benign",
                other => other,
            };
            if !class_names_sorted.iter().any(|c| c == label) {
                class_names_sorted.push(label.to_string());
            }
        }

        if features.is_empty() {
            return Err(anyhow!("CSV не содержит данных"));
        }

        // Сохраняем названия классов в порядке, заданном scikit-learn: malignant, benign.
        let mut class_names = vec!["malignant".to_string(), "benign".to_string()];
        // Если нашли другой набор классов — используем найденный, отсортированный.
        if class_names_sorted.iter().any(|c| c == "malignant")
            && class_names_sorted.iter().any(|c| c == "benign")
        {
            // стандартный случай — оставляем [malignant, benign]
        } else {
            class_names_sorted.sort();
            class_names = class_names_sorted;
        }

        Ok(Self {
            feature_names,
            class_names,
            features,
        })
    }

    /// Количество признаков (столбцов без target).
    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Доступ к сырой матрице признаков.
    #[allow(dead_code)]
    pub fn features_matrix(&self) -> &[Vec<f64>] {
        &self.features
    }
}

/// Стратегия разрешения источника данных (Chain of Responsibility).
/// Порядок поиска:
///   1. Переменная окружения `BC_DATA_PATH` (если задана).
///   2. Файл `breast_cancer.csv` в текущей рабочей директории.
///   3. Встроенный в бинарник CSV (никогда не падает).
///
/// Возвращает датасет и строку-описание использованного источника (для отладки).
pub fn load_with_fallback() -> Result<(CsvDataset, String)> {
    // 1. Переменная окружения.
    if let Ok(path) = std::env::var("BC_DATA_PATH") {
        match CsvDataset::from_path(&path) {
            Ok(ds) => return Ok((ds, format!("env BC_DATA_PATH={path}"))),
            Err(e) => eprintln!(
                "Предупреждение: не удалось прочитать BC_DATA_PATH={path}: {e}. \
                 Проверяю текущую директорию и встроенный датасет."
            ),
        }
    }

    // 2. Файл в текущей директории.
    let local = std::path::PathBuf::from("breast_cancer.csv");
    if local.exists() {
        match CsvDataset::from_path(&local) {
            Ok(ds) => return Ok((ds, format!("cwd file {}", local.display()))),
            Err(e) => eprintln!(
                "Предупреждение: файл {} найден, но не прочитан: {e}. \
                 Использую встроенный датасет.",
                local.display()
            ),
        }
    }

    // 3. Встроенный датасет.
    let ds = CsvDataset::from_embedded()?;
    Ok((ds, "embedded CSV (скомпилирован в бинарник)".to_string()))
}

impl DataSource for CsvDataset {
    fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    fn class_names(&self) -> &[String] {
        &self.class_names
    }

    fn n_samples(&self) -> usize {
        self.features.len()
    }

    fn feature_column(&self, index: usize) -> Result<&[f64]> {
        if index >= self.n_features() {
            return Err(anyhow!(
                "Индекс признака {index} вне диапазона (0..{})",
                self.n_features()
            ));
        }
        // Возвращаем "виртуальный" столбец — собираем на лету нельзя через slice.
        // Поэтому мы храним данные по строкам; для slice-доступа нужна transpose-матрица.
        // Чтобы сохранить API &[f64], мы транспонируем один раз лениво и кэшируем.
        // Но трейт не позволяет мутацию — используем отдельное хранилище.
        // Для простоты: возвращаем предвычисленную колонку из self.columns.
        // (См. реализацию ниже — мы используем ColumnarDataset в stats.)
        Err(anyhow!("feature_column не реализован для row-major CsvDataset; используйте ColumnarDataset"))
    }
}

/// Альтернативное column-major хранилище для эффективного slice-доступа.
/// Класс с инкапсуляцией: внешние части работают через методы, а не поля.
pub struct ColumnarDataset {
    feature_names: Vec<String>,
    class_names: Vec<String>,
    columns: Vec<Vec<f64>>, // [n_features][n_samples]
    n_samples: usize,
}

impl ColumnarDataset {
    /// Строит column-major датасет из row-major CsvDataset.
    pub fn from_row_major(row: &CsvDataset) -> Self {
        let n_features = row.n_features();
        let n_samples = row.n_samples();
        let mut columns = vec![Vec::with_capacity(n_samples); n_features];
        for r in 0..n_samples {
            for c in 0..n_features {
                columns[c].push(row.features[r][c]);
            }
        }
        Self {
            feature_names: row.feature_names().to_vec(),
            class_names: row.class_names().to_vec(),
            columns,
            n_samples,
        }
    }

    /// Конструктор для тестов и небольших примеров (ООП-инкапсуляция:
    /// внешние пользователи строят датасет через методы, а не литерал полей).
    #[allow(dead_code)]
    pub fn new(
        feature_names: Vec<String>,
        class_names: Vec<String>,
        columns: Vec<Vec<f64>>,
    ) -> Self {
        let n_samples = columns.first().map(|c| c.len()).unwrap_or(0);
        Self {
            feature_names,
            class_names,
            columns,
            n_samples,
        }
    }
}

impl DataSource for ColumnarDataset {
    fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    fn class_names(&self) -> &[String] {
        &self.class_names
    }

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn feature_column(&self, index: usize) -> Result<&[f64]> {
        self.columns
            .get(index)
            .map(|v| v.as_slice())
            .ok_or_else(|| anyhow!("Индекс признака {index} вне диапазона"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columnar_from_row_major_preserves_values() {
        let row = CsvDataset {
            feature_names: vec!["a".into(), "b".into()],
            class_names: vec!["malignant".into(), "benign".into()],
            features: vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        };
        let col = ColumnarDataset::from_row_major(&row);
        assert_eq!(col.n_samples(), 3);
        assert_eq!(col.feature_column(0).unwrap(), &[1.0, 3.0, 5.0]);
        assert_eq!(col.feature_column(1).unwrap(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn columnar_new_constructor() {
        let col = ColumnarDataset::new(
            vec!["a".into(), "b".into()],
            vec!["malignant".into(), "benign".into()],
            vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0]],
        );
        assert_eq!(col.n_samples(), 3);
        assert_eq!(col.feature_names().len(), 2);
        assert_eq!(col.class_names(), &["malignant", "benign"]);
        assert_eq!(col.feature_column(0).unwrap(), &[1.0, 3.0, 5.0]);
    }
}
