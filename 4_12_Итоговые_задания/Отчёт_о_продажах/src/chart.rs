//! Модуль построения столбчатого графика через крейт `plotters`.
//!
//! Имитирует API matplotlib: функция `bar_chart` принимает заголовок,
//! подписи осей, категории и высоты — аналогично `fig, ax = plt.subplots();
//! ax.bar(...); ax.set_title(...); ax.set_xlabel(...); ax.set_ylabel(...)`.

use plotters::prelude::*;
use std::error::Error;
use std::path::Path;

/// Структура-аксиом (как axes в matplotlib): содержит настройки графика.
pub struct Axes {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub categories: Vec<String>,
    pub heights: Vec<f64>,
}

impl Axes {
    pub fn new(title: String) -> Self {
        Self {
            title,
            x_label: String::new(),
            y_label: String::new(),
            categories: Vec::new(),
            heights: Vec::new(),
        }
    }
    pub fn xlabel(mut self, s: impl Into<String>) -> Self {
        self.x_label = s.into();
        self
    }
    pub fn ylabel(mut self, s: impl Into<String>) -> Self {
        self.y_label = s.into();
        self
    }
    pub fn bars(mut self, cats: Vec<String>, heights: Vec<f64>) -> Self {
        self.categories = cats;
        self.heights = heights;
        self
    }
}

/// Строит столбчатую диаграмму и сохраняет в PNG. Возвращает количество столбцов.
pub fn draw_bar_chart<P: AsRef<Path>>(
    out_path: P,
    axes: &Axes,
) -> Result<usize, Box<dyn Error>> {
    let bars = axes.categories.len();
    let max_h = axes
        .heights
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max)
        .max(1.0);
    // Вертикальные отступы чтобы столбец не упирался в самый верх.
    let y_top = (max_h * 1.15).ceil();

    let root = BitMapBackend::new(out_path.as_ref(), (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_cats: Vec<String> = axes.categories.clone();

    let mut chart = ChartBuilder::on(&root)
        .caption(&axes.title, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (0..bars).into_segmented(),
            0.0..y_top,
        )?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .x_labels(bars)
        .y_labels(6)
        .x_label_formatter(&|idx| {
            // Сегментированная ось: середина сегмента — это индекс категории.
            if let SegmentValue::CenterOf(i) = idx {
                x_cats.get(*i).cloned().unwrap_or_default()
            } else {
                String::new()
            }
        })
        .x_desc(&axes.x_label)
        .y_desc(&axes.y_label)
        .draw()?;

    chart.draw_series(
        axes.heights
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                let x0 = SegmentValue::Exact(i);
                let x1 = SegmentValue::Exact(i + 1);
                Rectangle::new([(x0, 0.0), (x1, h)], BLUE.filled())
            }),
    )?;

    root.present()?;
    Ok(bars)
}
