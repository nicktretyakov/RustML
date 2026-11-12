use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use std::io::{self, BufRead};

fn main() {
    let stdin = io::stdin();
    // Книги, взятые в главной библиотеке за шесть дней
    let books_vec: Vec<f64> = stdin
        .lock()
        .lines()
        .take(6)
        .map(|line| line.unwrap().trim().parse().expect("invalid number"))
        .collect();

    let books = Array1::from(books_vec);
    // Книги, взятые в сельском филиале за те же шесть дней
    let village = array![28.0, 35.0, 30.0, 33.0, 52.0, 32.0];

    let books_mean = books.mean().unwrap();
    let village_mean = village.mean().unwrap();

    if books_mean > village_mean {
        println!("Busier: main, {:.2} books a day", books_mean);
    } else {
        println!("Busier: village, {:.2} books a day", village_mean);
    }

    // numpy std по умолчанию использует ddof = 0 (population std) — совпадает с ndarray::std(0.0)
    let books_std = books.std(0.0);
    let village_std = village.std(0.0);

    if books_std < village_std {
        println!("Steadier: main, std {:.2}", books_std);
    } else {
        println!("Steadier: village, std {:.2}", village_std);
    }

let best = books.max().unwrap().max(*village.max().unwrap());
    println!("Best single day: {}", best);
}
