//! Доменные трейты (интерфейсы) — основа SOLID: ISP (узкие интерфейсы),
//! OCP (новые реализации расширяют систему), DIP (main зависит от абстракций).
//!
//! Каждый трейт отвечает ровно за одну обязанность (SRP) и не знает,
//! как устроены конкретные хранилища, предобработчики и классификаторы.

use crate::ml::Sample;
use ndarray::Array2;

/// Источник датасета: умеет только отдавать матрицу признаков X и вектор меток y.
/// (Interface Segregation: читателю всё равно, CSV ли это, JSON или БД.)
pub trait DatasetRepository {
    fn load(&self) -> anyhow::Result<Dataset>;
}

/// Итог загрузки: ровно 64 признака-пикселя (метка НЕ подмешана в X).
#[derive(Debug, Clone)]
pub struct Dataset {
    pub x: Array2<f64>,
    pub y: Vec<u8>,
}

/// Предобработчик признаков. Преобразует матрицу «как есть».
pub trait Preprocessor {
    fn fit_transform(&self, x: &Array2<f64>) -> anyhow::Result<Array2<f64>>;
}

/// Классификатор: обучается и предсказывает (typical ML contract).
pub trait Classifier: Send + Sync {
    fn fit(&mut self, x: &Array2<f64>, y: &[u8]) -> anyhow::Result<()>;
    fn predict(&self, x: &Array2<f64>) -> anyhow::Result<Vec<u8>>;
}

/// Счётчик меток: отвечает только за агрегацию количества образцов класса.
pub trait Counter {
    fn count_for(&self, label: u8) -> anyhow::Result<usize>;
    fn distribution(&self) -> anyhow::Result<Vec<(u8, usize)>>;
}

/// Репортёр: отвечает только за печать результата в нужном формате.
pub trait Reporter {
    fn report(&self, d: u8) -> anyhow::Result<()>;
}

/// Переиспользуемый тип образца для классификатора.
impl Sample for (Vec<f64>, u8) {
    fn features(&self) -> &[f64] {
        &self.0
    }
    fn label(&self) -> u8 {
        self.1
    }
}

/// Доменный объект «образец рукописной цифры»: 64 признака-пикселя + метка.
/// Демонстрирует ООП (структура + инкапсуляция) и реализует трейт `Sample`,
/// чтобы классификатор мог обучаться напрямую на доменных объектах.
#[derive(Debug, Clone)]
pub struct DigitSample {
    pixels: Vec<f64>,
    label: u8,
}

impl DigitSample {
    pub fn new(pixels: Vec<f64>, label: u8) -> Self {
        Self { pixels, label }
    }
}

impl Sample for DigitSample {
    fn features(&self) -> &[f64] {
        &self.pixels
    }
    fn label(&self) -> u8 {
        self.label
    }
}
