//! # Точка входа: анализ датасета пингвинов.
//!
//! Программа считывает с stdin название числового столбца и воспроизводит
//! поведение Python-эталона: подсчёт пропусков, очистка, средние по видам
//! и гистограмма.
//!
//! ## Архитектура (SOLID)
//!
//! - **DIP**: `main` зависит от трейтов [`DatasetLoader`], [`DropStrategy`],
//!   [`ChartRenderer`], [`Classifier`]; конкретные реализации подставляются
//!   через переменные.
//! - **SRP**: каждый шаг (`load` → `clean` → `analyze` → `render` → `ml`)
//!   выделен в отдельный вызов.
//! - **OCP**: добавить новый рендерер или стратегию очистки можно новой
//!   реализацией трейта, не трогая `main`.
//!
//! ## Многопоточность и макросы
//!
//! Средние по видам считаются параллельно через `rayon` (модуль
//! [`crate::threading`]), а форматирующий вывод использует макрос `dbg_num!`.

use std::io::{self, BufRead, Write};
use std::path::Path;

use penguins_rust::chart::{
    ChartRenderer, ChartSpec, HistogramDrawing, PlottersHistogramRenderer,
};
use penguins_rust::io::{CsvDatasetLoader, DatasetLoader};
use penguins_rust::ml::{Classifier, KnnSpeciesClassifier};
use penguins_rust::stats::{Analyzer, DropAnyNaStrategy, DropStrategy};
use penguins_rust::threading::ParallelStats;

// Палитра matplotlib tab10 для первых трёх видов.
const PALETTE: [(u8, u8, u8); 3] = [
    (31, 119, 180),  // tab:blue  → Adelie
    (255, 127, 14), // tab:orange → Chinstrap
    (44, 160, 44),  // tab:green  → Gentoo
];

fn rgb((r, g, b): (u8, u8, u8)) -> penguins_rust::chart::plotters_re_export::RGBColor {
    penguins_rust::chart::plotters_re_export::RGBColor(r, g, b)
}

fn read_column_name() -> anyhow::Result<String> {
    // Поведение `input()` из Python: читаем одну строку из stdin и
    // отрезаем завершающий перевод строки.
    let stdin = io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line)?;
    Ok(line.trim().to_string())
}

fn main() -> anyhow::Result<()> {
    // Выводим явно в stdout без буферизации, чтобы порядок строк точно
    // соответствовал Python-эталону.
    let stdout = io::stdout();
    let mut out = stdout.lock();

    // 1. Считать название столбца с stdin.
    let column = read_column_name()?;
    if column.is_empty() {
        anyhow::bail!("column name is empty");
    }

    // 2. Загрузить датасет (DIP: через трейт `DatasetLoader`).
    let csv_path = Path::new("penguins.csv");
    let loader = CsvDatasetLoader::new(csv_path);
    let dataset = loader.load()?;

    // Проверить, что столбец известен, чтобы сообщить об ошибке раньше.
    dataset.require_column(&column)?;

    // 3. Вывести количество пингвинов и число пропусков в столбце.
    writeln!(out, "Penguins: {}", dataset.len())?;
    writeln!(out, "Missing {}: {}", column, dataset.missing_count(&column))?;

    // 4. Удалить все строки с любыми пропусками (DIP: через `DropStrategy`).
    let cleaned = DropAnyNaStrategy.clean(&dataset);
    writeln!(out, "Kept after cleaning: {}", cleaned.len())?;

    // 5. Параллельно посчитать средние по видам (многопоточность: rayon).
    let stats = ParallelStats::means(&cleaned, &column);
    for s in &stats.per_species {
        // Макрос `dbg_num!` форматирует значение до 1 знака после точки,
        // повторяя Python `round(value, 1)` + `print`.
        penguins_rust::dbg_num!(&s.species, Analyzer::round(s.mean, 1));
    }

    // 6. Построить гистограммы по видам на одной диаграмме (ООП: ChartSpec).
    let mut chart_spec = ChartSpec::new(column.clone(), "Penguins".to_string());
    for (i, s) in stats.per_species.iter().enumerate() {
        let values = cleaned.values_for(&s.species, &column);
        let color = rgb(PALETTE[i % PALETTE.len()]);
        chart_spec.push(HistogramDrawing::new(s.species.clone(), values, color));
    }

    // 7. Отрисовать диаграмму в PNG-файл, имя которого повторяет столбец.
    let output_name = format!("{}.png", column);
    let renderer = PlottersHistogramRenderer::new(Path::new(&output_name));
    let summary = renderer.render(&chart_spec)?;

    // 8. Дополнительно обучить KNN-классификатор и оценить точность на
    //    обучающей выборке (машинное обучение: smartcore).
    let mut clf = KnnSpeciesClassifier::new(5);
    let ml_accuracy = match clf.fit(&cleaned) {
        Ok(()) => clf.accuracy(&cleaned).unwrap_or(0.0),
        Err(e) => {
            eprintln!("ML fit failed: {e}");
            0.0
        }
    };

    // 9. Вывести текстовое описание диаграммы — ровно в формате Python-эталона.
    writeln!(out, "Charts drawn: {}", summary.charts_drawn)?;
    writeln!(out, "Title: {}", summary.title)?;
    writeln!(out, "Y label: {}", summary.y_label)?;
    writeln!(out, "Histograms: {}", summary.histograms.len())?;
    for h in &summary.histograms {
        let counts_str = h
            .counts
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            out,
            "Histogram {}: {} bars, counts [{}]",
            h.label, h.bars, counts_str
        )?;
    }
    writeln!(out, "See-through bars: {}", if summary.see_through_bars { "yes" } else { "no" })?;
    writeln!(
        out,
        "Legend: {}",
        summary.legend.join(", ")
    )?;

    // 10. Краткая сводка по ML — это финальная часть вывода, не входящая
    //     в Python-эталон, но демонстрирующая ML-компоненту.
    eprintln!(
        "ML: KNN k=5 accuracy on training set = {:.3} ({} training rows)",
        ml_accuracy,
        cleaned.len()
    );

    Ok(())
}
