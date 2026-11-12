use ndarray::array;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::io;

fn main() {
    // Чтение seed из stdin
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read line");
    let seed: u64 = input.trim().parse().expect("Please enter a valid integer");

    // --- Генератор A (seed) ---
    let mut rng_a = StdRng::seed_from_u64(seed);
    let customers_a: [u32; 7] = array![
        rng_a.gen_range(20..61),
        rng_a.gen_range(20..61),
        rng_a.gen_range(20..61),
        rng_a.gen_range(20..61),
        rng_a.gen_range(20..61),
        rng_a.gen_range(20..61),
        rng_a.gen_range(20..61),
    ].into_raw_vec().try_into().unwrap();

    let normal = Normal::new(15.0, 4.0).unwrap();
    let spend: [f64; 7] = array![
        normal.sample(&mut rng_a),
        normal.sample(&mut rng_a),
        normal.sample(&mut rng_a),
        normal.sample(&mut rng_a),
        normal.sample(&mut rng_a),
        normal.sample(&mut rng_a),
        normal.sample(&mut rng_a),
    ].into_raw_vec().try_into().unwrap();

    // --- Генератор B (тот же seed) ---
    let mut rng_b = StdRng::seed_from_u64(seed);
    let customers_b: [u32; 7] = array![
        rng_b.gen_range(20..61),
        rng_b.gen_range(20..61),
        rng_b.gen_range(20..61),
        rng_b.gen_range(20..61),
        rng_b.gen_range(20..61),
        rng_b.gen_range(20..61),
        rng_b.gen_range(20..61),
    ].into_raw_vec().try_into().unwrap();

    // --- Генератор C (seed + 1) ---
    let mut rng_c = StdRng::seed_from_u64(seed + 1);
    let customers_c: [u32; 7] = array![
        rng_c.gen_range(20..61),
        rng_c.gen_range(20..61),
        rng_c.gen_range(20..61),
        rng_c.gen_range(20..61),
        rng_c.gen_range(20..61),
        rng_c.gen_range(20..61),
        rng_c.gen_range(20..61),
    ].into_raw_vec().try_into().unwrap();

    // --- Вывод ---
    println!("Customers: {:?}", customers_a);
    // Округление до одного знака после запятой
    let spend_rounded: Vec<f64> = spend.iter().map(|x| (x * 10.0).round() / 10.0).collect();
    println!("Spend: {:?}", spend_rounded);

    // Подсчёт совпадений
    let same_seed_matches = customers_a.iter().zip(customers_b.iter()).filter(|(a, b)| a == b).count();
    println!("Same seed, days that match: {}", same_seed_matches);

    let seed_plus_one_matches = customers_a.iter().zip(customers_c.iter()).filter(|(a, b)| a == b).count();
    println!("Seed + 1, days that match: {}", seed_plus_one_matches);
}
