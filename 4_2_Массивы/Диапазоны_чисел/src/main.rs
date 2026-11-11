// Cargo.toml:
// [dependencies]
// ndarray = "0.15"

use ndarray::Array1;
use std::io;

/// Аналог np.arange(first, last+1, gap): значения в [first, last] с шагом gap
fn arange(first: i64, last: i64, gap: i64) -> Array1<i64> {
    if gap == 0 {
        return Array1::from_vec(Vec::new());
    }
    let mut v = Vec::new();
    let mut x = first;
    if gap > 0 {
        while x <= last {
            v.push(x);
            x += gap;
        }
    } else {
        while x >= last {
            v.push(x);
            x += gap;
        }
    }
    Array1::from_vec(v)
}

/// Аналог np.linspace(first, last, n): n точек, включая оба конца
fn linspace(first: f64, last: f64, n: usize) -> Array1<f64> {
    match n {
        0 => Array1::zeros(0),
        1 => Array1::from_elem(1, first),
        _ => {
            let step = (last - first) / (n as f64 - 1.0);
            Array1::from_shape_fn(n, |i| first + step * i as f64)
        }
    }
}

/// Форматирование int-массива в стиле numpy: `[1 4 7]`
fn fmt_i64(a: &Array1<i64>) -> String {
    format!(
        "[{}]",
        a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
    )
}

/// Форматирование float-массива в стиле numpy: `[1. 4. 7.]`
fn fmt_f64(a: &Array1<f64>) -> String {
    let items: Vec<String> = a
        .iter()
        .map(|&x| {
            if x.fract() == 0.0 {
                format!("{:.0}.", x)   // 1.0 -> "1."
            } else {
                format!("{}", x)
            }
        })
        .collect();
    format!("[{}]", items.join(" "))
}

fn read_int() -> i64 {
    let mut s = String::new();
    io::stdin().read_line(&mut s).expect("read error");
    s.trim().parse().expect("not an integer")
}

fn main() {
    let first = read_int();
    let last = read_int();
    let gap = read_int();
    let stations = read_int();

    // cones = np.arange(first, last + 1, gap)
    let cones = arange(first, last, gap);
    println!("Cones: {}", fmt_i64(&cones));

    // np.linspace(first, last, stations)
    let stations_vals = linspace(first as f64, last as f64, stations as usize);
    println!("Stations: {}", fmt_f64(&stations_vals));

    // np.zeros(cones.size()) — в ndarray это Array1::zeros
    let times = Array1::<f64>::zeros(cones.len());
    println!("Times: {}", fmt_f64(&times));
}
