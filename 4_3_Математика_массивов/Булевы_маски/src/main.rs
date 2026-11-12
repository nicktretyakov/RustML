use itertools::Itertools;

macro_rules! read { ($t:ty) => {{
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).unwrap();
    s.trim().parse::<$t>().unwrap()
}}; }

fn main() {
    let times: Vec<i32> = vec![18, 34, 25, 41, 22, 29, 15, 38, 27, 31, 44, 20];
    let low  = read!(i32);
    let high = read!(i32);

    let on_time_vals: Vec<i32> = times.iter().copied()
        .filter(|t| *t >= low && *t <= high).collect();
    let on_time_count = on_time_vals.len();
    let outside_count = times.iter().filter(|t| **t < low || **t > high).count();

    println!("On time: {:?}", on_time_vals);
    println!("On time count: {}", on_time_count);
    println!("Outside count: {}", outside_count);
}
