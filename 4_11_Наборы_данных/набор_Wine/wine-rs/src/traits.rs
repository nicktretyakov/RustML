//! Трейты предметной области.
//!
//! Здесь сознательно применяются принципы SOLID:
//!
//! * **ISP** (Interface Segregation): трейты узкие и сфокусированные —
//!   `FeatureSource` отвечает только за доступ к данным, `StatsComputer` —
//!   только за вычисление метрик. Потребителю не навязываются лишние методы.
//! * **DIP** (Dependency Inversion): модуль `main` зависит от трейта
//!   `StatsComputer`, а не от конкретной `ParallelStats`.
//! * **OCP** (Open/Closed): новую стратегию расчёта (например, последовательную
//!   или основанную на GPU) можно добавить, реализовав `StatsComputer`,
//!   не трогая существующий код.
//! * **LSP** (Liskov Substitution): любая реализация `FeatureSource` может
//!   быть подставлена в `ParallelStats` без изменения поведения.

use std::collections::BTreeMap;

/// Источник данных о винах.
///
/// SRP: единственная ответственность — предоставлять значения признаков
/// и метки классов, скрывая способ хранения (linfa `Dataset`, CSV, БД...).
pub trait FeatureSource {
    /// Имена всех признаков в порядке столбцов.
    fn feature_names(&self) -> &[String];

    /// Уникальные метки классов в порядке возрастания (0, 1, 2 ...).
    fn class_labels(&self) -> Vec<usize>;

    /// Все значения признака `feature` (по всем образцам).
    fn all_feature_values(&self, feature: &str) -> Option<Vec<f64>>;

    /// Значения признака `feature` только для образцов класса `class`.
    fn class_feature_values(&self, class: usize, feature: &str) -> Option<Vec<f64>>;

    /// Число образцов класса `class`.
    fn class_count(&self, class: usize) -> usize {
        self.class_feature_values(class, self.feature_names().first().unwrap_or(&String::new()))
            .map(|v| v.len())
            .unwrap_or(0)
    }
}

/// Вычислитель трёх фактов, нужных лаборатории.
///
/// SRP: отвечает только за расчёт метрик на основе `FeatureSource`.
pub trait StatsComputer {
    /// Самый многочисленный класс: `(class, count)`.
    /// Если несколько классов равны, возвращается класс с наибольшим номером
    /// (как у `value_counts()` при стабильной сортировке по убыванию частоты).
    fn most_frequent_class(&self) -> Option<(usize, usize)>;

    /// Класс с наибольшим средним значением признака `feature`: `(class, mean)`.
    fn class_with_highest_mean(&self, feature: &str) -> Option<(usize, f64)>;

    /// Число образцов класса `class`, у которых `feature` выше
    /// общего среднего по всем винам.
    fn count_above_overall_mean(&self, class: usize, feature: &str) -> Option<usize>;
}

/// Вспомогательный тип для подсчёта частот классов через `rayon::fold`.
/// (SRP: хранит только агрегаты подсчёта.)
#[derive(Default, Clone)]
pub struct ClassCounts(pub BTreeMap<usize, usize>);

impl ClassCounts {
    pub fn merge(self, other: ClassCounts) -> ClassCounts {
        let mut a = self;
        for (k, v) in other.0 {
            *a.0.entry(k).or_insert(0) += v;
        }
        a
    }
}
