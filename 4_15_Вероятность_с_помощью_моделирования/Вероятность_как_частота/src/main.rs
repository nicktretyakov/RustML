use anyhow::{Context, Result};
use rand::RngCore;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use std::io::{self, BufRead};
use std::str::FromStr;
use std::sync::Mutex;

// ============ Абстракции (SOLID: ISP — разделяем интерфейсы) ============

/// Источник случайных чисел в [0, 1). Абстракция (SOLID: D).
trait RandomSource: Send {
    fn next_f64(&mut self) -> f64;
}

/// Симулятор серии пенальти.
trait KickSimulator {
    fn simulate(&self, kicks: usize, chance: f64) -> SimulationResult;
}

/// Оценщик вероятности (ML-часть). Допускает разные стратегии (SOLID: O).
trait Estimator: Send + Sync {
    fn estimate(&self, goals: usize, kicks: usize) -> f64;
}

#[derive(Debug, Clone, Copy)]
struct SimulationResult {
    goals: usize,
    kicks: usize,
}

// ============ RNG: обёртка над rand_pcg ============

struct PcgRng {
    inner: Pcg64Mcg,
}

impl PcgRng {
    fn seeded(seed: u64) -> Self {
        Self { inner: Pcg64Mcg::new(seed as u128) }
    }
}

impl RandomSource for PcgRng {
    fn next_f64(&mut self) -> f64 {
        // 53 значащих бита — стандартный приём для получения [0, 1).
        let bits = self.inner.next_u64() >> 11;
        (bits as f64) * (1.0 / (1u64 << 53) as f64)
    }
}

// ============ Симулятор (SOLID: S, D) ============

struct PenaltySimulator<R: RandomSource> {
    rng: Mutex<R>,
}

impl<R: RandomSource> PenaltySimulator<R> {
    fn new(rng: R) -> Self {
        Self { rng: Mutex::new(rng) }
    }
}

impl<R: RandomSource + 'static> KickSimulator for PenaltySimulator<R> {
    fn simulate(&self, kicks: usize, chance: f64) -> SimulationResult {
        // 1) Детерминированная генерация чисел — последовательно под мьютексом.
        let randoms: Vec<f64> = {
            let mut guard = self.rng.lock().expect("rng mutex poisoned");
            (0..kicks).map(|_| guard.next_f64()).collect()
        };

        // 2) Параллельный подсчёт голов через rayon.
        let goals = randoms.par_iter().filter(|&&r| r < chance).count();

        SimulationResult { goals, kicks }
    }
}

// ============ ML-оценщики (Beta-Bernoulli / MLE) ============

/// Оценка максимального правдоподобия: goals / kicks.
struct MleEstimator;

impl Estimator for MleEstimator {
    fn estimate(&self, goals: usize, kicks: usize) -> f64 {
        if kicks == 0 { 0.0 } else { goals as f64 / kicks as f64 }
    }
}

/// Байесовский оценщик с равномерным Beta(1,1)-приором.
#[allow(dead_code)]
struct BayesianEstimator {
    alpha: f64,
    beta: f64,
}

#[allow(dead_code)]
impl BayesianEstimator {
    fn uniform() -> Self { Self { alpha: 1.0, beta: 1.0 } }
}

impl Estimator for BayesianEstimator {
    fn estimate(&self, goals: usize, kicks: usize) -> f64 {
        (self.alpha + goals as f64) / (self.alpha + self.beta + kicks as f64)
    }
}

// ============ Макрос вывода ============

macro_rules! report {
    ($result:expr, $estimator:expr) => {{
        let r   = $result;
        let est = $estimator.estimate(r.goals, r.kicks);
        // Аналог np.round(x, 3) + печать без хвостовых нулей.
        let rounded = (est * 1000.0).round() / 1000.0;
        println!("Goals: {} of {}", r.goals, r.kicks);
        println!("Estimated chance: {}", rounded);
    }};
}

// ============ Ввод ============

fn read_val<T: FromStr>() -> Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    line.trim().parse::<T>().context("failed to parse input")
}

// ============ main ============

fn main() -> Result<()> {
    let chance: f64 = read_val()?;
    let kicks:  usize = read_val()?;

    // Зерно 21, как в Python-скрипте (default_rng(21)).
    // NB: numpy использует SeedSequence поверх PCG64 — последовательность
    //     может не совпадать бит-в-бит с rand_pcg, но структура идентична.
    let simulator = PenaltySimulator::new(PcgRng::seeded(21));
    let result    = simulator.simulate(kicks, chance);

    report!(result, MleEstimator);
    Ok(())
}
