use ndarray::{array, Array1, Axis};
use ndarray_stats::QuantileExt; // даёт метод .argmax()

// Макрос для округления до 2 знаков — аналог np.round(x, 2)
macro_rules! round2 {
    ($x:expr) => {
        (($x) * 100.0).round() / 100.0
    };
}

fn main() {
    let days = ["Mon", "Tue", "Wed", "Thu", "Fri"];

    // 4×5 матрица посещений (f64, чтобы mean работал без приведения)
    let visits = array![
        [52.0_f64, 61.0, 48.0, 70.0, 66.0],
        [58.0,     65.0, 51.0, 74.0, 69.0],
        [49.0,     60.0, 47.0, 69.0, 71.0],
        [55.0,     57.0, 50.0, 72.0, 60.0],
    ];

    // Аналог visits.mean(axis=0) — средние по столбцам (по дням недели)
    let day_means  = visits.mean_axis(Axis(0)).unwrap();
    // Аналог visits.mean(axis=1) — средние по строкам (по неделям)
    let week_means = visits.mean_axis(Axis(1)).unwrap();

    // Округляем через макрос, собирая обратно в Array1
    let day_rounded: Array1<f64>  = day_means.iter().map(|&x| round2!(x)).collect();
    let week_rounded: Array1<f64> = week_means.iter().map(|&x| round2!(x)).collect();

    println!("Per weekday: {}", day_rounded);
    println!("Per week: {}",    week_rounded);

    // Аналог day_means.argmax()
    let busiest = day_means.argmax().unwrap();
    println!("Busiest day: {}", days[busiest]);
}
