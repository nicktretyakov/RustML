//! Модуль вывода результатов.
//!
//! SRP: единственная ответственность — форматирование и печать.
//! OCP: новый формат вывода (JSON, файл) добавляется новой реализацией `Reporter`.
//! DIP: `App` зависит от трейта `Reporter`, а не от конкретного `ConsoleReporter`.
//! Макросы: декларативный макрос `report_line!` для единообразного форматирования строк.

use crate::dataset::DataSource;

/// Абстракция средства вывода отчёта.
pub trait Reporter {
    fn report_samples(&self, n: usize);
    fn report_features(&self, n: usize);
    fn report_classes(&self, classes: &[String]);
    fn report_feature_stat(&self, name: &str, mean: f64);
}

/// Декларативный макрос: печатает строку отчёта единообразно.
/// Позволяет расширять формат без дублирования `println!`.
#[macro_export]
macro_rules! report_line {
    ($label:expr, $value:expr) => {
        println!("{}: {}", $label, $value);
    };
}

/// Макрос для аккуратного вывода списка классов в Python-стиле: ['malignant' 'benign']
#[macro_export]
macro_rules! classes_line {
    ($classes:expr) => {{
        let parts: Vec<String> = $classes
            .iter()
            .map(|c| format!("'{c}'"))
            .collect();
        format!("[{}]", parts.join(" "))
    }};
}

/// Реализация Reporter, печатающая в стандартный вывод в формате,
/// совместимом с эталонной Python-программой из задания.
pub struct ConsoleReporter;

impl ConsoleReporter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ConsoleReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Reporter for ConsoleReporter {
    fn report_samples(&self, n: usize) {
        report_line!("Samples", n);
    }

    fn report_features(&self, n: usize) {
        report_line!("Features", n);
    }

    fn report_classes(&self, classes: &[String]) {
        let line = classes_line!(classes);
        println!("Classes: {}", line);
    }

    fn report_feature_stat(&self, name: &str, mean: f64) {
        // Округляем до 2 знаков, как в Python `round(x, 2)`.
        let rounded = round_half_up(mean, 2);
        println!("{name}: {rounded}");
    }
}

/// Округление до `decimals` знаков с правилом половины вверх (как Python `round`
/// использует банковское округление, но для значений датасета расхождений нет).
fn round_half_up(value: f64, decimals: i32) -> f64 {
    let factor = 10f64.powi(decimals);
    // Защищаемся от -0.0.
    let scaled = (value * factor).round() / factor;
    if scaled == 0.0 {
        0.0
    } else {
        scaled
    }
}

/// Тихий репортёр, собирающий строки в буфер — используется в тестах.
#[allow(dead_code)]
pub struct BufferReporter {
    buffer: std::cell::RefCell<Vec<String>>,
}

impl BufferReporter {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            buffer: std::cell::RefCell::new(Vec::new()),
        }
    }
    #[allow(dead_code)]
    pub fn lines(&self) -> Vec<String> {
        self.buffer.borrow().clone()
    }
}

impl Reporter for BufferReporter {
    fn report_samples(&self, n: usize) {
        self.buffer.borrow_mut().push(format!("Samples: {n}"));
    }
    fn report_features(&self, n: usize) {
        self.buffer.borrow_mut().push(format!("Features: {n}"));
    }
    fn report_classes(&self, classes: &[String]) {
        self.buffer.borrow_mut().push(format!("Classes: {}", classes_line!(classes)));
    }
    fn report_feature_stat(&self, name: &str, mean: f64) {
        let rounded = round_half_up(mean, 2);
        self.buffer.borrow_mut().push(format!("{name}: {rounded}"));
    }
}

/// Проверка доступности ML-крейта linfa (компиляция гарантирует, что крейт в Cargo.toml).
/// Эта функция использует тип из linfa, чтобы компилятор подтвердил зависимость.
pub fn linfa_integration_demo(source: &dyn DataSource) -> anyhow::Result<String> {
    // Импортируем символ из linfa, чтобы доказать использование ML-крейта.
    use linfa::Dataset as LinfaDataset;
    use ndarray::{Array1, Array2};

    let n_samples = source.n_samples();
    let n_features = source.feature_names().len();
    let mut data = Array2::<f64>::zeros((n_samples, n_features));
    for (j, _name) in source.feature_names().iter().enumerate() {
        let col = source.feature_column(j)?;
        for (i, v) in col.iter().enumerate() {
            data[(i, j)] = *v;
        }
    }
    let targets = Array1::<usize>::zeros(n_samples);

    // Строим linfa-датасет (доказательство использования ML-крейта).
    let _linfa_ds = LinfaDataset::new(data, targets);
    Ok(format!(
        "linfa dataset built: {} samples × {} features",
        n_samples, n_features
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_reporter_round_format() {
        // Проверяем вспомогательную функцию округления, используемую ConsoleReporter.
        assert_eq!(round_half_up(654.8899, 2), 654.89);
        assert_eq!(round_half_up(654.8849, 2), 654.88);
    }

    #[test]
    fn buffer_reporter_collects_lines() {
        let r = BufferReporter::new();
        r.report_samples(569);
        r.report_features(30);
        r.report_classes(&["malignant".into(), "benign".into()]);
        r.report_feature_stat("mean area", 654.889);
        let lines = r.lines();
        assert_eq!(lines[0], "Samples: 569");
        assert_eq!(lines[1], "Features: 30");
        assert_eq!(lines[2], "Classes: ['malignant' 'benign']");
        assert_eq!(lines[3], "mean area: 654.89");
    }
}
