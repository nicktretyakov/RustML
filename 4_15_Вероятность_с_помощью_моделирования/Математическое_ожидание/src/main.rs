// src/main.rs

use rand::prelude::*;
use rand::distributions::WeightedIndex;
use rand_distr::Distribution;
use rayon::prelude::*;
use ndarray::{Array1, Array2};
use linfa::dataset::Dataset;
use std::io::{self, Write};

// ------------------------------------------------------------
// 1. ООП и SOLID: трейты и структуры
// ------------------------------------------------------------

trait Wheel {
    fn prizes(&self) -> &[f64];
    fn chances(&self) -> &[f64];
    fn expected_value(&self) -> f64;
    fn simulate(&self, n: usize, seed: u64) -> Vec<f64>;
}

struct PrizeWheel {
    prizes: Vec<f64>,
    chances: Vec<f64>,
}

impl PrizeWheel {
    fn new(prizes: Vec<f64>, chances: Vec<f64>) -> Self {
        assert_eq!(prizes.len(), chances.len());
        assert!(chances.iter().all(|&c| c >= 0.0 && c <= 1.0));
        assert!((chances.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        Self { prizes, chances }
    }
}

impl Wheel for PrizeWheel {
    fn prizes(&self) -> &[f64] { &self.prizes }
    fn chances(&self) -> &[f64] { &self.chances }

    fn expected_value(&self) -> f64 {
        self.prizes.iter()
            .zip(self.chances.iter())
            .map(|(&p, &c)| p * c)
            .sum()
    }

    fn simulate(&self, n: usize, seed: u64) -> Vec<f64> {
        let dist = WeightedIndex::new(&self.chances).unwrap();
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
                let idx = dist.sample(&mut rng);
                self.prizes[idx]
            })
            .collect()
    }
}

// ------------------------------------------------------------
// 2. Макрос
// ------------------------------------------------------------

macro_rules! print_result {
    ($label:expr, $value:expr) => {
        println!("{}: {:.2} tokens", $label, $value);
    };
}

// ------------------------------------------------------------
// 3. "ML"-часть: обёртка данных в linfa::Dataset + ndarray
// ------------------------------------------------------------

fn mean_via_linfa(data: &[f64]) -> f64 {
    let n = data.len();

    // linfa::Dataset требует, чтобы признаки были двумерными (n × 1).
    let records = Array2::from_shape_vec((n, 1), data.to_vec())
        .expect("Не удалось создать Array2 из данных симуляции");

    // targets обязателен по типу, но для среднего он не нужен.
    let targets = Array1::<f64>::zeros(n);

    let dataset = Dataset::new(records, targets);

    // records() возвращает &Array2<f64>, у которого есть метод mean().
    dataset.records().mean().unwrap_or(0.0)
}

// ------------------------------------------------------------
// 4. main
// ------------------------------------------------------------

fn main() {
    // Ввод цены без приглашения — как в Python-версии.
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let price: f64 = input.trim().parse().expect("Введите число");

    let wheel = PrizeWheel::new(
        vec![0.0, 1.0, 4.0, 10.0],
        vec![0.5, 0.3, 0.15, 0.05],
    );

    let expected = wheel.expected_value();
    let spins = wheel.simulate(4000, 9);
    let simulated_avg = mean_via_linfa(&spins);

    let gain = expected - price;

    print_result!("Expected prize", expected);
    print_result!("Simulated average", simulated_avg);

    if gain >= 0.0 {
        println!("A player gains {:.2} tokens a spin", gain);
    } else {
        println!("A player loses {:.2} tokens a spin", -gain);
    }
}
