use std::io::{self, Read};
use ndarray::{arr2, Array2};

fn main() {
    // Читаем весь stdin и парсим числа
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut tokens = input.split_whitespace();

    let days: usize = tokens.next().unwrap().parse().unwrap();
    let showings: usize = tokens.next().unwrap().parse().unwrap();

    // Park: одна строка на каждый день, одно значение на каждый сеанс
    let mut park = Array2::<i32>::zeros((days, showings));
    for d in 0..days {
        for s in 0..showings {
            park[[d, s]] = tokens.next().unwrap().parse().unwrap();
        }
    }

    // Plaza
    let plaza = arr2(&[
        [ 34,  58,  91,  40],
        [ 29,  63,  85,  37],
        [ 41,  70,  96,  52],
        [ 38,  66, 102,  47],
        [ 55,  88, 120,  74],
        [ 72,  95, 131,  80],
        [ 60,  81,  99,  45],
    ]);

    let park_shape  = park.shape();   // &[usize] -> [days, showings]
    let plaza_shape = plaza.shape();
    let park_size   = park.len();     // общее число элементов
    let plaza_size  = plaza.len();

    println!(
        "Park: {} days, {} showings a day, {} values",
        park_shape[0], park_shape[1], park_size
    );
    println!(
        "Plaza: {} days, {} showings a day, {} values",
        plaza_shape[0], plaza_shape[1], plaza_size
    );
    println!("Same layout: {}", park_shape == plaza_shape);
}
