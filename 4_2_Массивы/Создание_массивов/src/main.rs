use ndarray::array;
use std::io;

fn read_line() -> String {
    let mut s = String::new();
    io::stdin().read_line(&mut s).expect("ошибка чтения");
    s.trim().to_string()
}

fn main() {
    let monday: i32 = read_line().parse().expect("целое число");
    let tuesday: i32 = read_line().parse().expect("целое число");
    let wednesday: i32 = read_line().parse().expect("целое число");
    let forecast: f64 = read_line().parse().expect("вещественное число");

    let rented = array![monday as f64, tuesday as f64, wednesday as f64];
    println!("{:?}", rented);

    let week = array![monday as f64, tuesday as f64, wednesday as f64, forecast];
    println!("{:?}", week);

    println!("{}", std::any::type_name_of_val(&week));
}
