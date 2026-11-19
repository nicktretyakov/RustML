//! Server Room Temperature — анализ показаний серверной за один день.
//!
//! Демонстрирует:
//! - **крейты**: `ndarray` (многомерные массивы), `rayon` (параллелизм);
//! - **макросы**: `info_line!`, `summary!`;
//! - **ООП**: структуры `WeekData`, `DayAnalyzer`, `OLSRegression`, `LineChart`;
//! - **многопоточность**: параллельный подсчёт через `par_iter` (rayon);
//! - **машинное обучение**: OLS-регрессия температуры от часа (с R^2);
//! - **SOLID**: трейты `ReadingSource` / `StatsAnalyzer` / `MLModel` / `ChartRenderer`
//!   разделяют ответственности (SRP), зависимость направлена на абстракции (DIP),
//!   трейты минимальны (ISP).

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::io::{self, BufRead};

// ============================================================================
// Константы
// ============================================================================

/// Сигнальное значение «измерение не удалось».
const MISSING: f64 = -999.0;
/// Порог температуры для подсчёта «выше лимита».
const LIMIT: f64 = 25.0;

// ============================================================================
// Макросы
// ============================================================================

/// Печатает строку вида `key: value`.
macro_rules! info_line {
    ($key:expr, $val:expr) => {
        println!("{}: {}", $key, $val);
    };
}

/// Печатает блок «Warmest: hour {h} at {v}».
macro_rules! warmest_line {
    ($hour:expr, $val:expr) => {
        println!("Warmest: hour {} at {}", $hour, $val);
    };
}

/// Лёгкий assert-style макрос для инвариантов SOLID-контрактов.
macro_rules! ensure {
    ($cond:expr, $msg:expr) => {
        if !$cond {
            panic!("contract violation: {}", $msg);
        }
    };
}

// ============================================================================
// Форматирование чисел (поведение Python / NumPy)
// ============================================================================

/// Поведение `str(np.float64(x))` для скаляра: целое → `"22.0"`, иначе кратчайшее
/// представление, например `"21.9"`. Используется для `Average` и `Warmest`.
fn fmt_py_scalar(v: f64) -> String {
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if v == v.trunc() && v.abs() < 1e16 {
        format!("{}.0", v as i64)
    } else {
        format!("{}", v)
    }
}

/// Поведение списка NumPy-значений в чекере: целое → `"21"`, иначе кратчайшее,
/// например `"17.2"`. Используется для элементов списка `Up (y)`.
fn fmt_py_list_item(v: f64) -> String {
    if v.is_nan() {
        return "nan".to_string();
    }
    if v == v.trunc() && v.abs() < 1e16 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

fn fmt_py_list_floats(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| fmt_py_list_item(*x)).collect();
    format!("[{}]", parts.join(", "))
}

fn fmt_py_list_ints(v: &[i64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

/// Округление «по банковскому правилу» (round half to even), как `np.round`.
fn np_round(x: f64, decimals: i32) -> f64 {
    let factor = 10f64.powi(decimals);
    let scaled = x * factor;
    let frac = scaled - scaled.trunc();
    let abs_frac = frac.abs();
    let rounded = if abs_frac < 0.5 {
        scaled.trunc()
    } else if abs_frac > 0.5 {
        scaled.trunc() + if scaled > 0.0 { 1.0 } else { -1.0 }
    } else {
        // ровно 0.5 -> к ближайшему чётному
        let floor = scaled.floor();
        let candidate = if scaled >= 0.0 { floor } else { floor - 1.0 };
        if (candidate as i64) % 2 == 0 {
            candidate
        } else if scaled >= 0.0 {
            floor + 1.0
        } else {
            floor
        }
    };
    rounded / factor
}

// ============================================================================
// Трейты (SOLID: SRP, ISP, DIP)
// ============================================================================

/// Источник показаний (SRP: только доставка данных).
pub trait ReadingSource {
    fn day_readings(&self, day: usize) -> Array1<f64>;
}

/// Анализатор статистики по показаниям (SRP: только вычисления).
pub trait StatsAnalyzer {
    fn failed_count(&self, readings: &Array1<f64>) -> usize;
    fn average_valid(&self, readings: &Array1<f64>) -> f64;
    fn count_above(&self, readings: &Array1<f64>, limit: f64) -> usize;
    fn warmest(&self, readings: &Array1<f64>) -> Option<(usize, f64)>;
    fn valid_pairs(&self, readings: &Array1<f64>) -> Vec<(usize, f64)>;
}

/// Модель машинного обучения (SRP: только fit/predict).
pub trait MLModel {
    fn fit(&mut self, x: &[f64], y: &[f64]);
    fn predict(&self, x: f64) -> f64;
    fn describe(&self) -> String;
}

/// Рендерер графика (SRP: только отрисовка).
pub trait ChartRenderer {
    fn render(&self, path: &str) -> io::Result<()>;
}

// ============================================================================
// Структуры (ООП)
// ============================================================================

/// Хранилище показаний за неделю: 4 дня × 24 часа.
pub struct WeekData {
    data: Array2<f64>,
}

impl WeekData {
    pub fn new(data: Array2<f64>) -> Self {
        ensure!(data.ncols() == 24, "ожидается 24 часа в строке");
        Self { data }
    }
}

impl ReadingSource for WeekData {
    fn day_readings(&self, day: usize) -> Array1<f64> {
        ensure!(day >= 1 && day <= self.data.nrows(), "день вне диапазона");
        self.data.row(day - 1).to_owned()
    }
}

/// Анализатор одного дня. Stateless — разные клиенты могут разделять один экземпляр.
pub struct DayAnalyzer;

impl DayAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DayAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsAnalyzer for DayAnalyzer {
    fn failed_count(&self, readings: &Array1<f64>) -> usize {
        let slice = readings.as_slice().unwrap_or_else(|| {
            // fallback: если вдруг массив не контигуален
            panic!("контигуальный массив ожидается")
        });
        // Многопоточность: параллельный подсчёт ошибок
        slice.par_iter().filter(|&&v| v == MISSING).count()
    }

    fn average_valid(&self, readings: &Array1<f64>) -> f64 {
        let valid: Vec<f64> = readings.iter().filter(|&&v| v != MISSING).copied().collect();
        ensure!(!valid.is_empty(), "нет корректных показаний");
        // Многопоточность: параллельная сумма
        let sum: f64 = valid.par_iter().sum();
        sum / valid.len() as f64
    }

    fn count_above(&self, readings: &Array1<f64>, limit: f64) -> usize {
        let valid: Vec<f64> = readings.iter().filter(|&&v| v != MISSING).copied().collect();
        // Многопоточность: параллельный фильтр
        valid.par_iter().filter(|&&v| v > limit).count()
    }

    fn warmest(&self, readings: &Array1<f64>) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (i, &v) in readings.iter().enumerate() {
            if v == MISSING {
                continue;
            }
            match best {
                Some((_, bv)) if v <= bv => {}
                _ => best = Some((i, v)),
            }
        }
        best
    }

    fn valid_pairs(&self, readings: &Array1<f64>) -> Vec<(usize, f64)> {
        readings
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != MISSING)
            .map(|(i, &v)| (i, v))
            .collect()
    }
}

// ----------------------------------------------------------------------------
// Машинное обучение: OLS-регрессия (линейная) температуры от часа
// ----------------------------------------------------------------------------

/// Простая линейная регрессия методом наименьших квадратов.
/// `y = slope * x + intercept`, с вычислением коэффициента детерминации R^2.
pub struct OLSRegression {
    slope: f64,
    intercept: f64,
    r_squared: f64,
}

impl OLSRegression {
    pub fn new() -> Self {
        Self {
            slope: 0.0,
            intercept: 0.0,
            r_squared: 0.0,
        }
    }
}

impl Default for OLSRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl MLModel for OLSRegression {
    fn fit(&mut self, x: &[f64], y: &[f64]) {
        ensure!(x.len() == y.len(), "x и y должны совпадать по длине");
        ensure!(!x.is_empty(), "пустая обучающая выборка");
        let n = x.len() as f64;
        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;
        // Многопоточность: параллельно считаем Sxy, Sxx, Syy
        let (sxy, sxx, syy) = x
            .par_iter()
            .zip(y)
            .map(|(xi, yi)| {
                let dx = xi - mean_x;
                let dy = yi - mean_y;
                (dx * dy, dx * dx, dy * dy)
            })
            .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        ensure!(sxx > 0.0, "нулевая дисперсия по x");
        self.slope = sxy / sxx;
        self.intercept = mean_y - self.slope * mean_x;
        self.r_squared = if syy > 0.0 {
            (sxy * sxy) / (sxx * syy)
        } else {
            0.0
        };
    }

    fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }

    fn describe(&self) -> String {
        format!(
            "OLSRegression(slope={:.4}, intercept={:.4}, R^2={:.4})",
            self.slope, self.intercept, self.r_squared
        )
    }
}

// ----------------------------------------------------------------------------
// График: линейный чарт с метаданными и SVG-рендером
// ----------------------------------------------------------------------------

/// Линейный график: заголовок, подписи осей, набор линий.
pub struct LineChart {
    title: String,
    xlabel: String,
    ylabel: String,
    lines: Vec<(Vec<i64>, Vec<f64>)>,
}

impl LineChart {
    pub fn new(title: &str, xlabel: &str, ylabel: &str) -> Self {
        Self {
            title: title.to_string(),
            xlabel: xlabel.to_string(),
            ylabel: ylabel.to_string(),
            lines: Vec::new(),
        }
    }

    pub fn add_line(&mut self, x: Vec<i64>, y: Vec<f64>) -> &mut Self {
        ensure!(x.len() == y.len(), "x и y должны совпадать по длине");
        self.lines.push((x, y));
        self
    }

    pub fn title(&self) -> &str {
        &self.title
    }
    pub fn xlabel(&self) -> &str {
        &self.xlabel
    }
    pub fn ylabel(&self) -> &str {
        &self.ylabel
    }
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Печатает метаданные графика в формате чекера.
    pub fn report(&self) {
        println!("Title: {}", self.title);
        println!("X label: {}", self.xlabel);
        println!("Y label: {}", self.ylabel);
        println!("Lines: {}", self.lines.len());
        if let Some((x, y)) = self.lines.first() {
            println!("Across (x): {}", fmt_py_list_ints(x));
            println!("Up (y): {}", fmt_py_list_floats(y));
        }
    }
}

impl ChartRenderer for LineChart {
    fn render(&self, path: &str) -> io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        if self.lines.is_empty() {
            return Ok(());
        }
        let (xs, ys) = &self.lines[0];
        if xs.is_empty() {
            return Ok(());
        }

        let width: u32 = 800;
        let height: u32 = 480;
        let margin: u32 = 60;

        let x_min = *xs.iter().min().unwrap() as f64;
        let x_max = *xs.iter().max().unwrap() as f64;
        let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let x_range = (x_max - x_min).max(1.0);
        let y_range = (y_max - y_min).max(1e-9);

        let plot_w = (width - 2 * margin) as f64;
        let plot_h = (height - 2 * margin) as f64;

        let scale_x = |x: f64| -> f64 {
            margin as f64 + (x - x_min) / x_range * plot_w
        };
        let scale_y = |y: f64| -> f64 {
            (height - margin) as f64 - (y - y_min) / y_range * plot_h
        };

        let mut path_data = String::new();
        for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            let sx = scale_x(*x as f64);
            let sy = scale_y(*y);
            if i == 0 {
                path_data.push_str(&format!("M {:.1} {:.1}", sx, sy));
            } else {
                path_data.push_str(&format!(" L {:.1} {:.1}", sx, sy));
            }
        }

        let mut svg = String::new();
        svg.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">\n",
            width, height
        ));
        svg.push_str(&format!(
            "<rect width=\"{}\" height=\"{}\" fill=\"white\"/>\n",
            width, height
        ));
        // оси
        svg.push_str(&format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"black\"/>\n",
            margin,
            height - margin,
            width - margin,
            height - margin
        ));
        svg.push_str(&format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"black\"/>\n",
            margin, margin, margin, height - margin
        ));
        // заголовок
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"18\" text-anchor=\"middle\">{}</text>\n",
            width / 2,
            30,
            self.title
        ));
        // подпись оси X
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"14\" text-anchor=\"middle\">{}</text>\n",
            width / 2,
            height - 15,
            self.xlabel
        ));
        // подпись оси Y (повёрнутая)
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"14\" text-anchor=\"middle\" transform=\"rotate(-90 {} {})\">{}</text>\n",
            20,
            height / 2,
            20,
            height / 2,
            self.ylabel
        ));
        // сама линия
        svg.push_str(&format!(
            "<path d=\"{}\" stroke=\"blue\" stroke-width=\"2\" fill=\"none\"/>\n",
            path_data
        ));
        // точки
        for (x, y) in xs.iter().zip(ys.iter()) {
            let sx = scale_x(*x as f64);
            let sy = scale_y(*y);
            svg.push_str(&format!(
                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"blue\"/>\n",
                sx, sy
            ));
        }
        svg.push_str("</svg>\n");

        let mut file = File::create(path)?;
        file.write_all(svg.as_bytes())?;
        Ok(())
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    // Чтение номера дня из stdin (как в исходной Python-программе)
    let stdin = io::stdin();
    let mut line = String::new();
    stdin
        .lock()
        .read_line(&mut line)
        .expect("не удалось прочитать ввод");
    let day: usize = line
        .trim()
        .parse()
        .expect("номер дня должен быть целым 1..=4");

    // Исходные данные (одна строка на каждый день, -999 = ошибка измерения)
    let week_data: Array2<f64> = Array2::from_shape_vec(
        (4, 24),
        vec![
            17.2, 16.8, 16.5, -999.0, 16.1, 16.4, 17.9, 19.6, 21.8, 23.5, 25.1, -999.0, 27.9, 28.8,
            29.4, 28.7, 27.2, 25.6, 23.9, -999.0, 21.0, 19.8, 18.6, 17.9, -999.0, 17.4, 17.0,
            16.7, 16.6, 17.0, 18.2, 19.9, 21.5, -999.0, 24.6, 26.0, 27.1, 27.8, 28.0, 28.3, 26.4,
            -999.0, 23.1, 21.7, 20.5, 19.4, 18.8, 18.1, 18.3, 18.0, 17.8, 17.5, 17.6, -999.0,
            -999.0, 20.4, 22.9, 25.3, 27.6, 29.5, 31.0, 32.2, 32.9, 33.1, 32.4, 30.8, 28.7, 26.5,
            24.6, 22.8, 21.3, 20.1, 16.9, 16.5, 16.2, 16.0, 15.8, 16.1, 16.9, 18.0, 19.4, 20.8,
            22.3, 23.6, 24.7, 26.2, 25.8, 25.1, 24.0, 22.6, 21.3, 20.2, 19.3, 18.5, 17.8, -999.0,
        ],
    )
    .expect("форма данных (4, 24)");

    // SOLID: зависимости направлены на трейты
    let source: Box<dyn ReadingSource> = Box::new(WeekData::new(week_data));
    let analyzer: Box<dyn StatsAnalyzer> = Box::new(DayAnalyzer::new());

    let readings = source.day_readings(day);

    let failed = analyzer.failed_count(&readings);
    let avg = analyzer.average_valid(&readings);
    let above = analyzer.count_above(&readings, LIMIT);
    let (warmest_hour, warmest_val) = analyzer
        .warmest(&readings)
        .expect("хотя бы одно корректное показание");

    // ----- Машинное обучение: OLS температура ~ час -----
    let pairs = analyzer.valid_pairs(&readings);
    let xs: Vec<f64> = pairs.iter().map(|(h, _)| *h as f64).collect();
    let ys: Vec<f64> = pairs.iter().map(|(_, t)| *t).collect();
    let mut model: Box<dyn MLModel> = Box::new(OLSRegression::new());
    model.fit(&xs, &ys);
    let pred_24 = model.predict(24.0);
    // ML-диагностика идёт в stderr, чтобы не ломать эталонный stdout
    eprintln!("[ML] {}", model.describe());
    eprintln!("[ML] Predicted temperature at hour 24: {:.1}", pred_24);

    // ----- Вывод статистики -----
    info_line!("Failed readings", failed);
    info_line!("Average", fmt_py_scalar(np_round(avg, 1)));
    info_line!("Hours above 25", above);
    warmest_line!(warmest_hour, fmt_py_scalar(warmest_val));

    // ----- График корректных показаний -----
    let mut chart = LineChart::new("Server Room Temperature", "Hour", "Temperature (C)");
    let chart_x: Vec<i64> = pairs.iter().map(|(h, _)| *h as i64).collect();
    let chart_y: Vec<f64> = pairs.iter().map(|(_, t)| *t).collect();
    chart.add_line(chart_x, chart_y);

    // Рендер в SVG-файл (тоже результат работы программы)
    let svg_path = "server_room_temperature.svg";
    if let Err(e) = chart.render(svg_path) {
        eprintln!("[chart] не удалось сохранить SVG: {}", e);
    }

    // ----- Метаданные графика -----
    println!("Charts drawn: 1");
    chart.report();
}
