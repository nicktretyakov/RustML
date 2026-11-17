//! Минимальный аналог `pandas.Series` для красивой печати результата.
//!
//! Хранит имя столбца и упорядоченный список пар (имя бегуна, среднее значение).
//! Метод `print()` выводит серию в стиле pandas:
//!
//! ```text
//! runner
//! Ana    6.1
//! Bo     8.4
//! Name: distance_km, dtype: float64
//! ```
//!
//! Уровень точности: одно значение после запятой, как в примере задачи
//! (8.4, 6.1). Подгонка ширины колонки — по самому длинному имени.

use std::cmp::max;

/// Столбец-результат: среднее значение выбранного числового столбца по бегунам.
pub struct Series {
    /// Имя исходного числового столбца ("distance_km", "minutes", "climb_m").
    pub column: String,
    /// Пары (имя бегуна, среднее значение), в порядке первого появления бегуна.
    pub entries: Vec<(String, f64)>,
}

impl Series {
    pub fn new(column: &str, entries: Vec<(String, f64)>) -> Self {
        Self {
            column: column.to_string(),
            entries,
        }
    }

    /// Длина имени бегуна-индекса (в pandas это имя группирующего столбца).
    /// В примере это "runner".
    const INDEX_LABEL: &'static str = "runner";

    /// Печать в стиле pandas Series.
    pub fn print(&self) {
        // Ширина колонки с именами: не меньше метки индекса и самого длинного имени.
        let name_width = self
            .entries
            .iter()
            .map(|(name, _)| name.len())
            .max()
            .unwrap_or(0);
        let width = max(Self::INDEX_LABEL.len(), name_width);

        println!("{}", Self::INDEX_LABEL);
        for (name, value) in &self.entries {
            // Имя выравнивается по левому краю, значение — с одним знаком после запятой.
            println!("{:<width$} {:.1}", name, value, width = width);
        }
        println!("Name: {}, dtype: float64", self.column);
    }

    /// Возвращает срез пар (имя, значение) — удобен для дальнейшей обработки,
    /// например, для поиска бегуна с максимальным средним.
    pub fn as_pairs(&self) -> Vec<(&str, f64)> {
        self.entries
            .iter()
            .map(|(n, v)| (n.as_str(), *v))
            .collect()
    }
}
