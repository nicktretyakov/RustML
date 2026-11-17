//! Смотритель парка: анализ зависимости высоты деревьев от возраста или толщины ствола.
//!
//! Программа читает `trees.csv` (поля `age_years`, `trunk_cm`, `height_m`),
//! параллельно считает корреляцию Пирсона высоты с каждым из двух столбцов,
//! затем спрашивает у пользователя имя столбца оси X и строит scatter-диаграмму:
//! значения выбранного столбца — по горизонтали, `height_m` — по вертикали,
//! ось X подписана именем этого столбца.
//!
//! ## Использованные механизмы Rust
//!
//! - **Крейты**: `csv` + `serde` (чтение данных), `rayon` (многопоточность),
//!   `plotters` (отрисовка PNG).
//! - **Макросы**: `getter!` (генерирует методы-аксессоры) и `column_type!`
//!   (генерирует структуру столбца + реализацию трейта `TreeColumn`).
//! - **ООП**: структуры с приватными полями и методами (инкапсуляция),
//!   трейты (полиморфизм), конструкторы, `impl`-блоки.
//! - **Многопоточность**: `rayon` параллельно извлекает значения столбца,
//!   считает границы осей и корреляцию Пирсона через `par_iter`/`reduce`.
//! - **SOLID**:
//!   - *SRP* — каждый компонент делает одну вещь (репозиторий грузит, селектор
//!     спрашивает, плоттер рисует, статистика считает).
//!   - *OCP* — новый источник данных/рендерер/столбец добавляется реализацией
//!     трейта, без правки существующего кода.
//!   - *LSP* — любая реализация трейта взаимозаменяема в оркестраторе.
//!   - *ISP* — трейты минимальны (`TreeRepository`, `ColumnSelector`,
//!     `ScatterPlotter`, `TreeColumn`).
//!   - *DIP* — `AnalysisApp` зависит от `Box<dyn Trait>`, а не от конкретных типов.

use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead};

use csv::ReaderBuilder;
use plotters::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;

// ───────────────────────────────────── Макросы ──────────────────────────────────────

/// Генерирует метод-геттер для приватного поля структуры `Tree`.
/// Убирает бойлерплейт: вместо трёх ручных методов — три однострочных вызова.
macro_rules! getter {
    ($field:ident) => {
        #[inline]
        pub fn $field(&self) -> f64 {
            self.$field
        }
    };
}

/// Генерирует тип столбца: одноимённую unit-структуру и реализацию трейта
/// `TreeColumn`, где `label` — имя столбца в CSV, а `value` берётся через
/// указанный метод-аксессор дерева.
macro_rules! column_type {
    ($name:ident, $label:literal, $accessor:ident) => {
        /// Столбец оси X, соответствующий полю дерева.
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;

        impl TreeColumn for $name {
            fn label(&self) -> &'static str {
                $label
            }
            fn value(&self, tree: &Tree) -> f64 {
                tree.$accessor()
            }
        }
    };
}

// ───────────────────────────────── Доменная модель (ООП) ───────────────────────────

/// Одно дерево парка. Поля приватны — наружу только через методы-геттеры,
/// сгенерированные макросом `getter!` (инкапсуляция).
#[derive(Debug, Clone, Deserialize)]
pub struct Tree {
    age_years: f64,
    trunk_cm: f64,
    height_m: f64,
}

impl Tree {
    /// Конструктор с проверкой неотрицательности измерений.
    pub fn new(age_years: f64, trunk_cm: f64, height_m: f64) -> Self {
        assert!(age_years >= 0.0 && trunk_cm >= 0.0 && height_m >= 0.0);
        Self { age_years, trunk_cm, height_m }
    }

    getter!(age_years);
    getter!(trunk_cm);
    getter!(height_m);
}

// ──────────────────────────── ISP/DIP: столбцы оси X ───────────────────────────────

/// Минимальный трейт «как получить значение столбца из дерева» (ISP).
/// Расширение новыми столбцами — через `column_type!` (OCP).
pub trait TreeColumn: Sync {
    fn label(&self) -> &'static str;
    fn value(&self, tree: &Tree) -> f64;
}

column_type!(AgeColumn, "age_years", age_years);
column_type!(TrunkColumn, "trunk_cm", trunk_cm);

/// Фабрика столбцов по имени (точка расширения OCP).
pub fn resolve_column(name: &str) -> Result<Box<dyn TreeColumn>, Box<dyn Error>> {
    match name.trim() {
        "age_years" => Ok(Box::new(AgeColumn)),
        "trunk_cm" => Ok(Box::new(TrunkColumn)),
        other => Err(format!("Неизвестный столбец: '{other}'. Допустимо: age_years, trunk_cm").into()),
    }
}

/// Все поддерживаемые столбцы оси X.
pub const ALLOWED_COLUMNS: &[&str] = &["age_years", "trunk_cm"];

// ──────────────────────────── ISP/DIP: источник данных ─────────────────────────────

/// Абстракция хранилища деревьев (DIP: зависим от абстракции, не от CSV).
pub trait TreeRepository {
    fn load_all(&self) -> Result<Vec<Tree>, Box<dyn Error>>;
}

/// Конкретный источник: CSV-файл по пути. Делает только чтение (SRP).
pub struct CsvTreeRepository {
    path: String,
}

impl CsvTreeRepository {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
}

impl TreeRepository for CsvTreeRepository {
    fn load_all(&self) -> Result<Vec<Tree>, Box<dyn Error>> {
        let file = File::open(&self.path)?;
        let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
        let mut trees = Vec::new();
        for result in rdr.deserialize() {
            trees.push(result?);
        }
        Ok(trees)
    }
}

// ──────────────────────────── ISP/DIP: выбор столбца ───────────────────────────────

/// Стратегия получения имени столбца оси X от пользователя (ISP).
pub trait ColumnSelector {
    fn select(&self, allowed: &[&str]) -> Result<String, Box<dyn Error>>;
}

/// Интерактивный выбор: читает одну строку из stdin (SRP).
pub struct InteractiveColumnSelector;

impl InteractiveColumnSelector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for InteractiveColumnSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl ColumnSelector for InteractiveColumnSelector {
    fn select(&self, allowed: &[&str]) -> Result<String, Box<dyn Error>> {
        let stdin = io::stdin();
        let mut line = String::new();
        stdin.lock().read_line(&mut line)?;
        let trimmed = line.trim().to_string();
        if allowed.iter().any(|c| *c == trimmed.as_str()) {
            Ok(trimmed)
        } else {
            Err(format!(
                "Неизвестный столбец: '{trimmed}'. Допустимо: {}",
                allowed.join(", ")
            )
            .into())
        }
    }
}

// ──────────────────────── Многопоточность: статистика ──────────────────────────────

/// Коэффициент корреляции Пирсона между двумя рядами.
/// Считается параллельно через `rayon` (один `par_iter().reduce(...)` по всем
/// точкам накапливает суммы Σx, Σy, Σxy, Σx², Σy²).
pub fn pearson(points: &[(f64, f64)]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let n = points.len() as f64;
    let (sx, sy, sxy, sx2, sy2) = points
        .par_iter()
        .map(|&(x, y)| (x, y, x * y, x * x, y * y))
        .reduce(
            || (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
        );
    let mx = sx / n;
    let my = sy / n;
    let cov = sxy / n - mx * my;
    let vx = (sx2 / n - mx * mx).max(0.0);
    let vy = (sy2 / n - my * my).max(0.0);
    let denom = vx.sqrt() * vy.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        cov / denom
    }
}

/// Параллельно считает корреляцию высоты с каждым столбцом и выбирает лидера.
pub struct CorrelationReport {
    pub age_r: f64,
    pub trunk_r: f64,
    pub winner: &'static str,
}

impl CorrelationReport {
    /// Считает обе корреляции независимо — rayon выполняет их параллельно.
    pub fn compute(trees: &[Tree]) -> Self {
        let (age_r, trunk_r) = (
            trees
                .par_iter()
                .map(|t| (t.age_years(), t.height_m()))
                .collect::<Vec<_>>(),
            trees
                .par_iter()
                .map(|t| (t.trunk_cm(), t.height_m()))
                .collect::<Vec<_>>(),
        );
        let (age_r, trunk_r) = (pearson(&age_r), pearson(&trunk_r));
        let winner = if trunk_r.abs() >= age_r.abs() {
            "trunk_cm (толщина ствола)"
        } else {
            "age_years (возраст)"
        };
        Self { age_r, trunk_r, winner }
    }

    pub fn print(&self) {
        println!("─ Корреляция Пирсона с height_m ─────────────────────");
        println!("  age_years : {:+.4}", self.age_r);
        println!("  trunk_cm  : {:+.4}", self.trunk_r);
        println!("  => С высотой сильнее связан: {}", self.winner);
        println!("─────────────────────────────────────────────────────");
    }
}

// ──────────────────────────── ISP/DIP: рендерер ────────────────────────────────────

/// Абстракция рендерера scatter-диаграммы (DIP, ISP).
pub trait ScatterPlotter {
    fn plot(
        &self,
        trees: &[Tree],
        column: &dyn TreeColumn,
        out_path: &str,
    ) -> Result<(), Box<dyn Error>>;
}

/// Рендерер на базе крейта `plotters` (SRP: только рисует).
pub struct PlottersScatterPlotter {
    width: u32,
    height: u32,
}

impl PlottersScatterPlotter {
    pub fn new() -> Self {
        Self { width: 900, height: 600 }
    }

    /// Параллельно извлекает пары (значение столбца, высота) по всем деревьям.
    fn extract_xy(trees: &[Tree], column: &dyn TreeColumn) -> Vec<(f64, f64)> {
        trees
            .par_iter()
            .map(|t| (column.value(t), t.height_m()))
            .collect()
    }

    /// Параллельно находит min/max по X и Y с небольшим запасом.
    fn bounds(points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
        let (x_min, x_max) = points
            .par_iter()
            .map(|(x, _)| *x)
            .fold(|| (f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| (lo.min(x), hi.max(x)))
            .reduce(|| (f64::INFINITY, f64::NEG_INFINITY), |a, b| (a.0.min(b.0), a.1.max(b.1)));
        let (y_min, y_max) = points
            .par_iter()
            .map(|(_, y)| *y)
            .fold(|| (f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), y| (lo.min(y), hi.max(y)))
            .reduce(|| (f64::INFINITY, f64::NEG_INFINITY), |a, b| (a.0.min(b.0), a.1.max(b.1)));
        let pad_x = ((x_max - x_min).abs().max(1.0)) * 0.05;
        let pad_y = ((y_max - y_min).abs().max(1.0)) * 0.05;
        (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)
    }
}

impl Default for PlottersScatterPlotter {
    fn default() -> Self {
        Self::new()
    }
}

impl ScatterPlotter for PlottersScatterPlotter {
    fn plot(
        &self,
        trees: &[Tree],
        column: &dyn TreeColumn,
        out_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        let points = Self::extract_xy(trees, column);
        if points.is_empty() {
            return Err("Нет данных для построения диаграммы".into());
        }
        let (x_min, x_max, y_min, y_max) = Self::bounds(&points);

        let root = BitMapBackend::new(out_path, (self.width, self.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Tree Size", ("sans-serif", 28))
            .margin(20)
            .x_label_area_size(48)
            .y_label_area_size(56)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        // Подпись оси X — имя выбранного столбца; ось Y — Height (m).
        chart
            .configure_mesh()
            .x_desc(column.label())
            .y_desc("Height (m)")
            .draw()?;

        // По одной точке на каждое дерево.
        chart.draw_series(
            points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 5_u32, RED.filled())),
        )?;

        root.present()?;
        Ok(())
    }
}

// ──────────────────────────────── Оркестратор (DIP) ────────────────────────────────

/// Высокоуровневое приложение: зависит только от трейтов, конкретные
/// реализации внедряются снаружи (внедрение зависимостей — DIP).
pub struct AnalysisApp {
    repo: Box<dyn TreeRepository>,
    selector: Box<dyn ColumnSelector>,
    plotter: Box<dyn ScatterPlotter>,
}

impl AnalysisApp {
    pub fn new(
        repo: Box<dyn TreeRepository>,
        selector: Box<dyn ColumnSelector>,
        plotter: Box<dyn ScatterPlotter>,
    ) -> Self {
        Self { repo, selector, plotter }
    }

    /// Загружает данные, печатает корреляционный отчёт, спрашивает столбец,
    /// рисует диаграмму в `out_path`.
    pub fn run(&self, out_path: &str) -> Result<(), Box<dyn Error>> {
        let trees = self.repo.load_all()?;
        println!("Загружено деревьев: {}", trees.len());

        CorrelationReport::compute(&trees).print();

        let col_name = self.selector.select(ALLOWED_COLUMNS)?;
        let column = resolve_column(&col_name)?;
        self.plotter.plot(&trees, column.as_ref(), out_path)?;
        println!("Диаграмма сохранена: {out_path} (ось X = {col_name})");
        Ok(())
    }
}

// ─────────────────────────────────── Точка входа ───────────────────────────────────

fn main() -> Result<(), Box<dyn Error>> {
    let csv_path = std::env::var("TREES_CSV").unwrap_or_else(|_| "trees.csv".to_string());

    let app = AnalysisApp::new(
        Box::new(CsvTreeRepository::new(csv_path)),
        Box::new(InteractiveColumnSelector::new()),
        Box::new(PlottersScatterPlotter::new()),
    );

    // Имя выходного файла берётся из переменной окружения OUT (по умолчанию scatter.png).
    let out = std::env::var("OUT").unwrap_or_else(|_| "scatter.png".to_string());
    app.run(&out)
}
