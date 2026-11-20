//! Контроль качества розлива: проверка равномерности наполнения бутылок соком.
//!
//! Программа считывает шесть значений объёма в мл, параллельно (многопоточно)
//! считает разброс «вручную» и сверяет результат с ML-эталоном на базе `ndarray`
//! (numpy-эквивалент в Rust — аналог `np.std`).
//!
//! # Архитектура (SOLID + ООП)
//!
//! * `VolumeSource`      — трейт-абстракция ввода (SRP: только чтение значений).
//! * `SpreadCalculator`  — трейт-абстракция расчёта разброса
//!   (OCP: новый алгоритм — новая реализация без правки старых; DIP: зависимость от абстракции).
//! * `ResultReporter`    — трейт-абстракция вывода (SRP: только отчёт).
//! * `QualityControlPipeline` — оркестратор: собирает реализации за трейтами и
//!   запускает «ручной» и «эталонный» расчёты параллельно в потоках
//!   (`std::thread::scope`), затем передаёт результаты репортёру.
//!
//! # Макросы
//!
//! * `macro_rules! parallel_mean_variance!` — авторский макрос слоя
//!   многопоточности: разворачивается в параллельный расчёт (среднее, дисперсия)
//!   через `rayon`.
//! * `#[derive(Serialize)]` (serde) и `#[derive(Error)]` (thiserror) — макросы
//!   внешних крейтов.
//!
//! # Машинное обучение
//!
//! `ndarray` — фундамент ML-экосаистемы Rust (на нём строится, например, `linfa`).
//! `NdarraySpreadCalculator` играет роль эталонной статистики — аналог `np.std`
//! (популяционное СКО, ddof = 0).

use ndarray::Array1;
use rayon::prelude::*;
use serde::Serialize;
use std::io::{self, BufRead, Write};
use thiserror::Error;

// ─────────────────────────── Доменные типы (ООП, value objects) ──────────────

/// Объём одной бутылки в миллилитрах.
#[derive(Debug, Clone, Copy, Serialize)]
struct MlVolume(f64);

impl MlVolume {
    #[inline]
    fn as_f64(&self) -> f64 {
        self.0
    }
}

/// Результат расчёта разброса: дисперсия и стандартное отклонение.
#[derive(Debug, Clone, Copy, Serialize)]
struct SpreadReport {
    variance: f64,
    std: f64,
}

// ───────────────────────── Ошибки (макрос thiserror::Error) ──────────────────

#[derive(Debug, Error)]
enum QcError {
    #[error("ошибка ввода-вывода: {0}")]
    Io(#[from] io::Error),
    #[error("не удалось разобрать целое число: {0}")]
    Parse(#[from] std::num::ParseIntError),
    #[error("ожидано {expected} значений объёма, получено {got}")]
    NotEnough { expected: usize, got: usize },
}

// ───────────────────── Трейты-абстракции (SRP / OCP / DIP / ISP) ─────────────

/// Источник значений объёма. Единственная ответственность — ввод (SRP).
trait VolumeSource {
    fn read_volumes(&self, n: usize) -> Result<Vec<MlVolume>, QcError>;
}

/// Калькулятор разброса. `Sync` — чтобы ссылку можно было передать в поток.
/// Аналог питоновской `spread_by_hand(a)`: возвращает дисперсию и СКО (OCP/DIP).
trait SpreadCalculator: Sync {
    fn spread_by_hand(&self, a: &[f64]) -> SpreadReport;
}

/// Репортёр. Единственная ответственность — вывод результата (SRP).
trait ResultReporter {
    fn report(&self, hand: &SpreadReport, reference: &SpreadReport);
}

// ──────────────────────────── Источник: stdin (SRP) ──────────────────────────

struct StdinVolumeSource;

impl VolumeSource for StdinVolumeSource {
    fn read_volumes(&self, n: usize) -> Result<Vec<MlVolume>, QcError> {
        let stdin = io::stdin();
        let mut out = Vec::with_capacity(n);
        for line in stdin.lock().lines().take(n) {
            let line = line?;
            let value: i64 = line.trim().parse()?;
            out.push(MlVolume(value as f64));
        }
        if out.len() < n {
            return Err(QcError::NotEnough {
                expected: n,
                got: out.len(),
            });
        }
        Ok(out)
    }
}

// ───────────── Калькулятор «вручную»: многопоточный, rayon + макрос ──────────

/// Авторский макрос слоя многопоточности.
/// Разворачивается в параллельный расчёт `(mean, variance)` для среза `&[f64]`:
///   * `mean` — параллельная сумма (rayon), делённая на N;
///   * `variance` — параллельная сумма квадратов отклонений, делённая на N
///     (дисперсия = среднее квадратов отклонений от среднего).
macro_rules! parallel_mean_variance {
    ($a:expr) => {{
        let a: &[f64] = $a;
        let n = a.len() as f64;
        let mean: f64 = a.par_iter().copied().sum::<f64>() / n;
        let sum_sq: f64 = a
            .par_iter()
            .copied()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>();
        (mean, sum_sq / n)
    }};
}

/// Ручной калькулятор: дисперсия = среднее квадратов отклонений,
/// СКО = sqrt(дисперсии). Расчёт многопоточный (rayon через макрос).
struct HandSpreadCalculator;

impl SpreadCalculator for HandSpreadCalculator {
    fn spread_by_hand(&self, a: &[f64]) -> SpreadReport {
        let (_mean, variance) = parallel_mean_variance!(a);
        SpreadReport {
            variance,
            std: variance.sqrt(),
        }
    }
}

// ───────────── ML-эталон: ndarray (numpy-эквивалент, аналог np.std) ─────────

/// Калькулятор-эталон на базе ML-библиотеки `ndarray`.
/// Популяционное СКО (ddof = 0) — как у `np.std` по умолчанию:
///   variance = mean((x - mean)^2);  std = sqrt(variance).
struct NdarraySpreadCalculator;

impl SpreadCalculator for NdarraySpreadCalculator {
    fn spread_by_hand(&self, a: &[f64]) -> SpreadReport {
        let arr = Array1::from_vec(a.to_vec());
        let mean = arr.mean().unwrap_or(0.0);
        let variance = arr.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
        SpreadReport {
            variance,
            std: variance.sqrt(),
        }
    }
}

// ──────────────────────────── Репортёр: консоль (SRP) ────────────────────────

struct ConsoleReporter;

impl ResultReporter for ConsoleReporter {
    fn report(&self, hand: &SpreadReport, reference: &SpreadReport) {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        let _ = writeln!(out, "Variance: {:.2} ml squared", hand.variance);
        let _ = writeln!(out, "Std: {:.2} ml", hand.std);
        // Сравнение округлённых до 2 знаков значений — как np.round(std,2) == np.round(np.std,2).
        let same = format!("{:.2}", hand.std) == format!("{:.2}", reference.std);
        let _ = writeln!(out, "Same as np.std: {}", if same { "True" } else { "False" });
    }
}

// ─────────────────────── Пайплайн (оркестрация, DIP) ─────────────────────────

/// Оркестратор контроля качества. Зависит только от трейтов-абстракций (DIP),
/// конкретные реализации подключаются извне (OCP).
struct QualityControlPipeline<'a> {
    source: &'a dyn VolumeSource,
    hand: &'a dyn SpreadCalculator,
    reference: &'a dyn SpreadCalculator,
    reporter: &'a dyn ResultReporter,
}

impl<'a> QualityControlPipeline<'a> {
    fn run(&self, n: usize) -> Result<(), QcError> {
        let volumes = self.source.read_volumes(n)?;
        let xs: Vec<f64> = volumes.iter().map(MlVolume::as_f64).collect();

        // Два независимых расчёта запускаем параллельно в потоках (std::thread::scope):
        // «ручной» (HandSpreadCalculator) и ML-эталон (NdarraySpreadCalculator).
        let hand_calc = self.hand;
        let ref_calc = self.reference;
        let xs_ref = &xs;
        let (hand_report, ref_report) = std::thread::scope(|s| {
            let h = s.spawn(move || hand_calc.spread_by_hand(xs_ref));
            let r = s.spawn(move || ref_calc.spread_by_hand(xs_ref));
            (
                h.join().expect("panic в потоке ручного расчёта"),
                r.join().expect("panic в потоке ML-эталона"),
            )
        });

        self.reporter.report(&hand_report, &ref_report);
        Ok(())
    }
}

// ─────────────────────────────────── main ────────────────────────────────────

fn main() -> Result<(), QcError> {
    const N: usize = 6;

    let source = StdinVolumeSource;
    let hand = HandSpreadCalculator;
    let reference = NdarraySpreadCalculator;
    let reporter = ConsoleReporter;

    QualityControlPipeline {
        source: &source,
        hand: &hand,
        reference: &reference,
        reporter: &reporter,
    }
    .run(N)
}
