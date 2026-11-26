use derive_more::Display;
use ndarray::Array1;
use rand::seq::index::sample;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;                 // макрос-подобный трейт, даёт .into_par_iter()
use std::io::{self, Read};
use thiserror::Error;                  // #[derive(Error)] — макрос-крейт

// ---------- Ошибки (SOLID: SRP) ----------
#[derive(Error, Debug)]
enum PollError {
    #[error("IO error: {0}")]      Io(#[from] io::Error),
    #[error("Parse error: {0}")]   Parse(#[from] std::num::ParseIntError),
}

// ---------- DIP: абстракция источника случайности ----------
trait RandomSource: Send + Sync + Clone {
    fn new(seed: u64) -> Self;
    fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize>;
}

#[derive(Clone)]
struct ChaChaSource(ChaCha8Rng);

impl RandomSource for ChaChaSource {
    fn new(seed: u64) -> Self { Self(ChaCha8Rng::seed_from_u64(seed)) }
    fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        sample(&mut self.0, n, k).into_vec()   // без повторений, как np.choice(..., replace=False)
    }
}

// ---------- DTO опроса ----------
#[derive(Debug, Clone, Copy, Display)]
#[display(fmt = "{share:.3}, give or take {margin:.3}")]
struct PollResult {
    share:  f64,
    margin: f64,
}

// ---------- Стратегия опроса (SOLID: OCP, LSP) ----------
trait PollStrategy: Send + Sync {
    fn conduct<R: RandomSource>(&self, voters: &[u8], size: usize, rng: &mut R) -> PollResult;
}

struct SimpleRandomPoll;

impl PollStrategy for SimpleRandomPoll {
    fn conduct<R: RandomSource>(&self, voters: &[u8], size: usize, rng: &mut R) -> PollResult {
        let idx = rng.sample_indices(voters.len(), size);

        // ML-стек: считаем статистику через ndarray
        let poll: Array1<f64> = Array1::from_iter(idx.iter().map(|&i| voters[i] as f64));

        let n     = poll.len() as f64;
        let share = poll.mean().unwrap_or(0.0);
        let var   = poll.iter().map(|x| (x - share).powi(2)).sum::<f64>() / n; // ddof = 0
        let std   = var.sqrt();
        let margin = 2.0 * std / n.sqrt();

        PollResult { share, margin }
    }
}

// ---------- Оценщик (ML: evaluation-модуль) ----------
struct Evaluator { truth: f64 }

impl Evaluator {
    fn new(voters: &[u8]) -> Self {
        let truth = voters.iter().map(|&x| x as f64).sum::<f64>() / voters.len() as f64;
        Self { truth }
    }
    fn caught(&self, polls: &[PollResult]) -> usize {
        polls.iter()
            .filter(|p| p.share - p.margin < self.truth && self.truth < p.share + p.margin)
            .count()
    }
    fn called_wrong(&self, polls: &[PollResult]) -> usize {
        polls.iter().filter(|p| p.share <= 0.5).count()
    }
}

// ---------- Точка входа ----------
fn run() -> Result<(), PollError> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let size: usize = input.trim().parse()?;

    // Голоса: 1 — сторонник, 0 — остальные
    let mut voters = vec![1u8; 10800];
    voters.extend(std::iter::repeat(0u8).take(9200));

    let evaluator = Evaluator::new(&voters);
    println!("True support: {:.3}", evaluator.truth);

    let strategy  = SimpleRandomPoll;
    let n_polls   = 400usize;

    // Многопоточность через rayon: 400 опросов параллельно
    let polls: Vec<PollResult> = (0..n_polls)
        .into_par_iter()
        .map(|i| {
            let mut rng = ChaChaSource::new(23 + i as u64); // детерминированный seed
            strategy.conduct(&voters, size, &mut rng)
        })
        .collect();

    // Вывод
    println!("First poll: {}", polls[0]); // derive_more::Display — как в шаблоне
    println!(
        "Margins that caught the truth: {} of {}",
        evaluator.caught(&polls),
        n_polls
    );
    println!("Polls calling it wrong: {}", evaluator.called_wrong(&polls));

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
