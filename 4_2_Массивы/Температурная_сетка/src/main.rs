use ndarray::prelude::*;
use std::io;

fn main() {
    // Температуры в теплице в градусах Цельсия: одна строка на каждый день (дни с 1 по 5),
    // один столбец на каждое измерение, выполненное в 6, 12, 18 и 24 часа
    let readings = array![
        [12.5, 21.0, 24.5, 15.0],
        [13.0, 22.5, 23.5, 14.5],
        [11.5, 19.0, 21.0, 13.0],
        [14.0, 24.5, 26.0, 16.5],
        [12.0, 20.5, 22.0, 14.0],
    ];

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let day: usize = input.trim().parse().unwrap();

    input.clear();
    io::stdin().read_line(&mut input).unwrap();
    let hour: usize = input.trim().parse().unwrap();

    // Напишите свой код ниже
    let hours: Array1<i32> = (6..25).step_by(6).collect();
    println!("Hours: {:?}", hours);
    println!("Days recorded: {}", readings.shape()[0]);

    let column = hour / 6 - 1;
    println!(
        "Last 3 days at {}:00: {:?}",
        hour,
        readings.slice(s![-3.., column])
    );
    println!(
        "Day {} from noon on: {:?}",
        day,
        readings.slice(s![day - 1, 1..])
    );
}
