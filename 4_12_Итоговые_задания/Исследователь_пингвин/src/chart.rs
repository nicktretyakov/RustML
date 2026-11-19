//! Модуль отрисовки гистограммы.
//!
//! **SOLID — DIP + ISP**: `main` зависит от узкого трейта `ChartRenderer`,
//! а не от конкретной библиотеки `plotters`. Реализация
//! `PlottersHistogramRenderer` инкапсулирует знание графического бэкенда.
//!
//! **ООП**: `HistogramDrawing` — объект-значение с конфигурацией одной
//! колонки (вид + значения + стиль), `PlottersHistogramRenderer` — объект
//! с состоянием (цветовая палитра и путь к файлу).

use std::path::Path;

use plotters::prelude::*;

use crate::stats::{HistogramBin, HistogramSpec};

/// Реэкспорт типа цвета, чтобы внешние модули (main.rs) не зависели
/// напрямую от крейта `plotters` (DIP: зависим от нашей абстракции, а не от
/// конкретной библиотеки).
pub mod plotters_re_export {
    pub use plotters::style::RGBColor;
}

/// Описание одной гистограммы на диаграмме (один вид пингвина).
#[derive(Debug, Clone)]
pub struct HistogramDrawing {
    pub label: String,
    pub values: Vec<f64>,
    pub color: RGBColor,
    pub spec: HistogramSpec,
}

impl HistogramDrawing {
    pub fn new(label: impl Into<String>, values: Vec<f64>, color: RGBColor) -> Self {
        Self {
            label: label.into(),
            values,
            color,
            spec: HistogramSpec::default(),
        }
    }
}

/// Полная диаграмма: заголовок, ось Y, набор гистограмм и флаг «прозрачные
/// столбцы» (соответствует `alpha=0.5`).
#[derive(Debug, Clone)]
pub struct ChartSpec {
    pub title: String,
    pub y_label: String,
    pub histograms: Vec<HistogramDrawing>,
    pub see_through_bars: bool,
}

impl ChartSpec {
    pub fn new(title: impl Into<String>, y_label: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            y_label: y_label.into(),
            histograms: Vec::new(),
            see_through_bars: true,
        }
    }

    pub fn push(&mut self, h: HistogramDrawing) {
        self.histograms.push(h);
    }
}

/// Трейт рендерера диаграмм (DIP/ISP). Конкретная реализация — ниже.
pub trait ChartRenderer {
    /// Отрисовать диаграмму и вернуть сводку, которая одновременно подходит
    /// для текстового вывода и автотестов.
    fn render(&self, spec: &ChartSpec) -> anyhow::Result<RenderSummary>;
}

/// Текстовое описание результата отрисовки — для удобства проверок.
#[derive(Debug, Clone, Default)]
pub struct RenderSummary {
    pub charts_drawn: usize,
    pub title: String,
    pub y_label: String,
    pub histograms: Vec<HistogramSummary>,
    pub see_through_bars: bool,
    pub legend: Vec<String>,
}

/// Описание одной отрисованной гистограммы.
#[derive(Debug, Clone)]
pub struct HistogramSummary {
    pub label: String,
    pub bars: usize,
    pub counts: Vec<usize>,
}

/// Конкретный рендерер на `plotters`. Сохраняет результат в PNG-файл.
pub struct PlottersHistogramRenderer<'a> {
    output_path: &'a Path,
    /// Палитра цветов для разных видов. Совпадает с порядком matplotlib
    /// по умолчанию (tab10): синий, оранжевый, зелёный.
    palette: Vec<RGBColor>,
}

impl<'a> PlottersHistogramRenderer<'a> {
    pub fn new(output_path: &'a Path) -> Self {
        Self {
            output_path,
            palette: vec![
                RGBColor(31, 119, 180),  // tab:blue
                RGBColor(255, 127, 14), // tab:orange
                RGBColor(44, 160, 44),  // tab:green
                RGBColor(214, 39, 40),  // tab:red
            ],
        }
    }

    fn color_for(&self, i: usize) -> RGBColor {
        self.palette.get(i).copied().unwrap_or(RGBColor(0, 0, 0))
    }
}

impl<'a> ChartRenderer for PlottersHistogramRenderer<'a> {
    fn render(&self, spec: &ChartSpec) -> anyhow::Result<RenderSummary> {
        // Расчёт бинов для каждой гистограммы — единый логический шаг.
        let mut bins_per_series: Vec<(HistogramDrawing, Vec<HistogramBin>)> = Vec::new();
        for h in &spec.histograms {
            let bins = crate::stats::Analyzer::histogram(&h.values, h.spec);
            bins_per_series.push((h.clone(), bins));
        }

        // Глобальные границы оси X: от min до max по всем сериям.
        let x_min = bins_per_series
            .iter()
            .filter_map(|(h, _)| h.values.iter().cloned().min_by(f64::total_cmp))
            .fold(f64::INFINITY, f64::min);
        let x_max = bins_per_series
            .iter()
            .filter_map(|(h, _)| h.values.iter().cloned().max_by(f64::total_cmp))
            .fold(f64::NEG_INFINITY, f64::max);
        let y_max = bins_per_series
            .iter()
            .flat_map(|(_, bins)| bins.iter().map(|b| b.count))
            .max()
            .unwrap_or(0);

        const W: u32 = 1000;
        const H: u32 = 600;
        let root = BitMapBackend::new(self.output_path, (W, H)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| anyhow::anyhow!("failed to fill background: {e:?}"))?;

        let caption_style = ("sans-serif", 28).into_font();
        let mut chart = ChartBuilder::on(&root)
            .caption(&spec.title, caption_style)
            .margin(20)
            .x_label_area_size(45)
            .y_label_area_size(55)
            .build_cartesian_2d(x_min..x_max, 0usize..y_max.max(1))?;

        chart
            .configure_mesh()
            .x_desc(&spec.title)
            .y_desc(&spec.y_label)
            .draw()?;

        // Каждая серия — отдельный набор прямоугольников с легендой.
        for (i, (h, bins)) in bins_per_series.iter().enumerate() {
            let color = self.color_for(i);
            let alpha = h.spec.alpha;
            let series_label = h.label.clone();
            chart
                .draw_series(bins.iter().map(|b| {
                    let fill = color.mix(alpha);
                    Rectangle::new([(b.lower, 0), (b.upper, b.count)], fill.filled())
                }))?
                .label(series_label)
                .legend(move |(x, y)| {
                    Rectangle::new(
                        [(x, y - 5), (x + 20, y + 5)],
                        color.mix(alpha).filled(),
                    )
                });
        }

        // Легенда справа сверху.
        chart
            .configure_series_labels()
            .border_style(BLACK)
            .background_style(WHITE.mix(0.8))
            .draw()?;

        // Сводка для текстового вывода (повторяет формат Python-эталона).
        let histograms: Vec<HistogramSummary> = bins_per_series
            .iter()
            .map(|(h, bins)| HistogramSummary {
                label: h.label.clone(),
                bars: bins.len(),
                counts: bins.iter().map(|b| b.count).collect(),
            })
            .collect();

        Ok(RenderSummary {
            charts_drawn: 1,
            title: spec.title.clone(),
            y_label: spec.y_label.clone(),
            histograms,
            see_through_bars: spec.see_through_bars,
            legend: spec.histograms.iter().map(|h| h.label.clone()).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_writes_png_and_summary() {
        let tmp = std::env::temp_dir().join("penguins_chart_test.png");
        let renderer = PlottersHistogramRenderer::new(&tmp);
        let mut spec = ChartSpec::new("bill_length_mm", "Penguins");
        spec.push(HistogramDrawing::new(
            "Adelie",
            vec![39.0, 40.0, 41.0],
            RGBColor(31, 119, 180),
        ));
        let summary = renderer.render(&spec).expect("render failed");
        assert_eq!(summary.charts_drawn, 1);
        assert_eq!(summary.title, "bill_length_mm");
        assert_eq!(summary.y_label, "Penguins");
        assert_eq!(summary.histograms.len(), 1);
        assert_eq!(summary.histograms[0].bars, 8);
        assert!(tmp.exists());
        let _ = std::fs::remove_file(&tmp);
    }
}
