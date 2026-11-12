use ndarray::{Array2, Axis};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::io::{self, Write};

// Небольшой макрос для форматированного вывода строки таблицы
macro_rules! scoreline {
    ($round:expr, $highest:expr, $lowest:expr) => {
        println!("{:>5}\t{:>7}\t{:>6}", $round, $highest, $lowest);
    };
}

fn main() {
    print!("Enter seed: ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let seed: u64 = input.trim().parse().expect("seed must be an integer");

    let mut rng = StdRng::seed_from_u64(seed);

    // 15 бросков, как rng.integers(1, 7, size=15)
    let rolls: Vec<i64> = (0..15).map(|_| rng.gen_range(1..=6)).collect();

    // reshape(5, 3)
    let rounds: Array2<i64> = Array2::from_shape_vec((5, 3), rolls)
        .expect("shape mismatch");

    // highest / lowest по каждой строке
    let highest = rounds.map_axis(Axis(1), |row| *row.iter().max().unwrap());
    let lowest  = rounds.map_axis(Axis(1), |row| *row.iter().min().unwrap());

    println!("Round, highest, lowest:");
    for i in 0..5 {
        scoreline!(i + 1, highest[i], lowest[i]);
    }

    // (lowest >= 4).sum()
    let count = lowest.iter().filter(|&&x| x >= 4).count();
    println!("Rounds with no die under 4: {}", count);
}
