#[macro_use]
extern crate peroxide;
use peroxide::fuga::*;
use std::io::{self, BufRead};

fn read_three_f64() -> Vec<f64> {
    let stdin = io::stdin();
    let mut vals = Vec::with_capacity(3);
    for line in stdin.lock().lines().take(3) {
        vals.push(line.unwrap().trim().parse::<f64>().unwrap());
    }
    vals
}

fn main() {
    let km = read_three_f64();
    let minutes = read_three_f64();

    // Умножение вектора на скаляр
    let meters = km.mul_scalar(1000.0);
    // Поэлементное деление векторов
    let pace = minutes.div_v(&km);

    println!("Meters:");
    meters.print();
    println!("Minutes per km:");
    pace.print();
}
