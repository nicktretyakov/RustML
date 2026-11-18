// Анализ урожая помидоров на Rust.
//
// Демонстрирует:
//  * крейты: csv + serde (чтение CSV), rayon (многопоточность), plotters (график);
//  * макрос: `mean!` для вычисления среднего;
//  * ООП: инкапсулированные структуры `Harvest` и `VarietyGroup` с методами;
//  * SOLID: маленькие сфокусированные трейты `GroupStats` и `ScatterSeries`,
//    renderer зависит от абстракции (`dyn ScatterSeries`), а не от конкретного типа,
//    новые сорта добавляются без изменения кода отрисовки (OCP/DIP).

use std::error::Error;

use plotters::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;

// ----------------------------------------------------------------------------
// Макрос: вычисление среднего по срезу f64. Срабатывает на этапе компиляции.
// ----------------------------------------------------------------------------
macro_rules! mean {
    ($vals:expr) => {{
        let v: &[f64] = $vals;
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }};
}

// ----------------------------------------------------------------------------
// Модель данных (читается из tomatoes.csv).
// ----------------------------------------------------------------------------
#[derive(Debug, Clone, Deserialize)]
struct TomatoRecord {
    variety: String,
    diameter_cm: f64,
    weight_g: f64,
}

// ----------------------------------------------------------------------------
// ООП: коллекция инкапсулирует сырые записи и владеет логикой группировки.
// Single Responsibility: только загрузка и доступ к данным.
// ----------------------------------------------------------------------------
struct Harvest {
    records: Vec<TomatoRecord>,
}

impl Harvest {
    /// Загрузить урожай из CSV-файла.
    fn from_csv(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut reader = csv::Reader::from_path(path)?;
        let records = reader
            .deserialize::<TomatoRecord>()
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { records })
    }

    /// Доступ только для чтения к записям (инкапсуляция).
    fn records(&self) -> &[TomatoRecord] {
        &self.records
    }

    /// Уникальные сорта в порядке первого появления (как pandas `.unique()`).
    fn varieties(&self) -> Vec<String> {
        let mut seen: Vec<String> = Vec::new();
        for r in &self.records {
            if !seen.iter().any(|v| v == &r.variety) {
                seen.push(r.variety.clone());
            }
        }
        seen
    }

    /// Разбить записи по сортам.
    fn group_by_variety(&self) -> Vec<VarietyGroup> {
        let varieties = self.varieties();
        varieties
            .into_iter()
            .map(|v| {
                let recs: Vec<TomatoRecord> = self
                    .records
                    .iter()
                    .filter(|r| r.variety == v)
                    .cloned()
                    .collect();
                VarietyGroup::new(v, recs)
            })
            .collect()
    }
}

// ----------------------------------------------------------------------------
// SOLID (Interface Segregation): трейт только для статистики группы.
// ----------------------------------------------------------------------------
trait GroupStats {
    fn count(&self) -> usize;
    fn mean_diameter(&self) -> f64;
    fn mean_weight(&self) -> f64;
}

// ООП: группа сорта инкапсулирует свои записи и реализует статистику.
struct VarietyGroup {
    variety: String,
    records: Vec<TomatoRecord>,
}

impl VarietyGroup {
    fn new(variety: String, records: Vec<TomatoRecord>) -> Self {
        Self { variety, records }
    }

    fn variety(&self) -> &str {
        &self.variety
    }

    fn diameters(&self) -> Vec<f64> {
        self.records.iter().map(|r| r.diameter_cm).collect()
    }

    fn weights(&self) -> Vec<f64> {
        self.records.iter().map(|r| r.weight_g).collect()
    }
}

impl GroupStats for VarietyGroup {
    fn count(&self) -> usize {
        self.records.len()
    }

    fn mean_diameter(&self) -> f64 {
        mean!(&self.diameters())
    }

    fn mean_weight(&self) -> f64 {
        mean!(&self.weights())
    }
}

// ----------------------------------------------------------------------------
// SOLID (Dependency Inversion / Open-Closed): трейт "рисуемый ряд точек".
// Renderer зависит от абстракции, а не от конкретного сорта.
// Новый источник точек (например, прошлогодний урожай) реализует этот трейт
// и сразу работает с renderer'ом без правок.
// ----------------------------------------------------------------------------
trait ScatterSeries {
    fn label(&self) -> &str;
    fn color(&self) -> RGBColor;
    fn points(&self) -> Vec<(f64, f64)>;
}

impl ScatterSeries for VarietyGroup {
    fn label(&self) -> &str {
        &self.variety
    }

    fn color(&self) -> RGBColor {
        // Стабильная палитра по названию сорта (как matplotlib color cycle).
        match self.variety.as_str() {
            "cherry" => RGBColor(231, 76, 60),   // красный
            "plum" => RGBColor(52, 152, 219),    // синий
            "beefsteak" => RGBColor(39, 174, 96),// зелёный
            _ => RGBColor(128, 128, 128),
        }
    }

    fn points(&self) -> Vec<(f64, f64)> {
        self.records
            .iter()
            .map(|r| (r.diameter_cm, r.weight_g))
            .collect()
    }
}

// ----------------------------------------------------------------------------
// Renderer: зависит только от `dyn ScatterSeries` (DIP).
// Воспроизводит matplotlib-эталон: title, xlabel, ylabel, scatter per variety,
// legend.
// ----------------------------------------------------------------------------
fn render_scatter(
    groups: &[Box<dyn ScatterSeries>],
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    // Поля графика с небольшим запасом по краям.
    let (xmin, xmax, ymin, ymax) = groups.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
        |(mnx, mxx, mny, mxy), g| {
            g.points().iter().fold((mnx, mxx, mny, mxy), |acc, &(x, y)| {
                (acc.0.min(x), acc.1.max(x), acc.2.min(y), acc.3.max(y))
            })
        },
    );
    let pad_x = (xmax - xmin) * 0.08 + 0.5;
    let pad_y = (ymax - ymin) * 0.08 + 5.0;

    let root = BitMapBackend::new(out_path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Tomato Harvest", ("sans-serif", 36))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(70)
        .build_cartesian_2d(
            (xmin - pad_x)..(xmax + pad_x),
            (ymin - pad_y)..(ymax + pad_y),
        )?;

    chart
        .configure_mesh()
        .x_desc("Diameter (cm)")
        .y_desc("Weight (g)")
        .draw()?;

    // Цикл по сортам: для каждого — отдельный scatter-ряд и запись в легенду.
    // Это прямой аналог Python-цикла `for variety in tomatoes["variety"].unique()`.
    for group in groups {
        let color = group.color();
        let points = group.points(); // продлём время жизни, иначе временный Vec выпадет
        let series = points
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 5, color.filled()));

        chart
            .draw_series(series)?
            .label(group.label().to_string())
            .legend(move |coord| Circle::new(coord, 5, color.filled()));
    }

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.9))
        .draw()?;

    root.present()?;
    Ok(())
}

// ----------------------------------------------------------------------------
// Многопоточность (rayon): статистика по сортам считается параллельно.
// ----------------------------------------------------------------------------
fn compute_stats_parallel(groups: &[VarietyGroup]) -> Vec<(&str, usize, f64, f64)> {
    groups
        .par_iter()
        .map(|g| {
            (
                g.variety(),
                g.count(),
                g.mean_diameter(),
                g.mean_weight(),
            )
        })
        .collect()
}

// ----------------------------------------------------------------------------
// Анализ разделения сортов: сорта считаются разделёнными, если ни одна пара
// не перекрывается ни по диаметру, ни по весу.
// ----------------------------------------------------------------------------
fn separation_report(groups: &[VarietyGroup]) -> (bool, String) {
    let mut report = String::new();
    let mut all_separated = true;

    let n = groups.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let a = &groups[i];
            let b = &groups[j];

            let a_d = a.diameters();
            let b_d = b.diameters();
            let a_w = a.weights();
            let b_w = b.weights();

            let (a_dmin, a_dmax) = (
                a_d.iter().cloned().fold(f64::INFINITY, f64::min),
                a_d.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            );
            let (b_dmin, b_dmax) = (
                b_d.iter().cloned().fold(f64::INFINITY, f64::min),
                b_d.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            );
            let (a_wmin, a_wmax) = (
                a_w.iter().cloned().fold(f64::INFINITY, f64::min),
                a_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            );
            let (b_wmin, b_wmax) = (
                b_w.iter().cloned().fold(f64::INFINITY, f64::min),
                b_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            );

            let diam_gap = (a_dmin - b_dmax).abs().min((b_dmin - a_dmax).abs());
            let weight_gap = (a_wmin - b_wmax).abs().min((b_wmin - a_wmax).abs());
            let separated = diam_gap > 0.0 && weight_gap > 0.0;
            if !separated {
                all_separated = false;
            }

            report.push_str(&format!(
                "  {} vs {}: gap diameter = {:.2} cm, gap weight = {:.2} g => {}\n",
                a.variety(),
                b.variety(),
                diam_gap,
                weight_gap,
                if separated { "разделены" } else { "перекрытие" }
            ));
        }
    }
    (all_separated, report)
}

fn main() -> Result<(), Box<dyn Error>> {
    let csv_path = "tomatoes.csv";
    let out_path = "tomato_harvest.png";

    println!("Загрузка данных из {} ...", csv_path);
    let harvest = Harvest::from_csv(csv_path)?;
    println!("Загружено помидоров: {}", harvest.records().len());

    let groups = harvest.group_by_variety();
    println!("Найдено сортов: {}", groups.len());

    // Параллельный (rayon) расчёт статистики по сортам.
    let stats = compute_stats_parallel(&groups);
    println!("\nСтатистика по сортам (параллельно, rayon):");
    println!("  {:<10} {:>6} {:>14} {:>14}", "variety", "count", "mean_diam(cm)", "mean_weight(g)");
    for (v, c, md, mw) in &stats {
        println!("  {:<10} {:>6} {:>14.2} {:>14.2}", v, c, md, mw);
    }

    // Анализ разделения.
    let (separated, report) = separation_report(&groups);
    println!("\nАнализ разделения сортов:");
    print!("{}", report);
    println!(
        "\nВывод: сорта {}.",
        if separated {
            "ЧЁТКО РАЗДЕЛЕНЫ — кластеры не перекрываются"
        } else {
            "перекрываются (разделение неполное)"
        }
    );

    // Подготовка рядов для отрисовки через абстракцию `dyn ScatterSeries` (DIP).
    let series: Vec<Box<dyn ScatterSeries>> = groups
        .iter()
        .map(|g| Box::new(g.clone()) as Box<dyn ScatterSeries>)
        .collect();

    println!("\nОтрисовка scatter-диаграммы в {} ...", out_path);
    render_scatter(&series, out_path)?;
    println!("Готово. Файл сохранён: {}", out_path);

    Ok(())
}

// Клонирование группы нужно, чтобы передать её в `Box<dyn ScatterSeries>`,
// не забирая оригиналы у `groups` (используются выше для отчёта).
impl Clone for VarietyGroup {
    fn clone(&self) -> Self {
        Self {
            variety: self.variety.clone(),
            records: self.records.clone(),
        }
    }
}
