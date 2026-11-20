//! Модуль `data` — источник данных о расходе воды (SRP).
//!
//! Trait [`DataSource`] абстрагирует способ получения выборки расхода воды,
//! чтобы движок не зависел от конкретного хранения (DIP). Реализация
//! [`StreetConsumption`] хранит значения в `ndarray::Array1`.

use ndarray::Array1;

/// Источник данных расхода воды.
pub trait DataSource {
    /// Возвращает наблюдения расхода воды в литрах.
    fn usage(&self) -> &Array1<f64>;
    /// Количество домов в выборке.
    fn size(&self) -> usize;
}

/// Расход воды по домам на улице.
#[derive(Debug, Clone)]
pub struct StreetConsumption {
    usage: Array1<f64>,
}

impl StreetConsumption {
    /// Создаёт выборку из вектора значений. Возвращает ошибку при пустом вводе.
    pub fn new(values: Vec<f64>) -> Self {
        let usage = Array1::from_vec(values);
        assert!(!usage.is_empty(), "sample must not be empty");
        Self { usage }
    }
}

impl DataSource for StreetConsumption {
    fn usage(&self) -> &Array1<f64> {
        &self.usage
    }

    fn size(&self) -> usize {
        self.usage.len()
    }
}
