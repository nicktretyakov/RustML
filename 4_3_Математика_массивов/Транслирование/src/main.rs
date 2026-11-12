use ndarray::{array, Array1, Array2, Axis};

/// Вычитает из каждого столбца его среднее значение (аналог `table - table.mean(axis=0)`).
fn center(table: &Array2<f64>) -> Array2<f64> {
    // mean_axis(Axis(0)) — среднее по столбцам, как `table.mean(axis=0)` в NumPy
    let means: Array1<f64> = table.mean_axis(Axis(0)).unwrap();
    // ndarray поддерживает broadcasting: (n, m) - (m,) → (n, m)
    table - &means
}

fn main() {
    // Одна строка на пользователя: возраст, пульс в покое, минуты упражнений в день
    let users = array![
        [25.0, 62.0, 30.0],
        [31.0, 70.0, 45.0],
        [40.0, 58.0, 20.0],
        [29.0, 66.0, 25.0],
    ];

    // Одна строка на тренировку: продолжительность в минутах, сожжённые калории
    let workouts = array![
        [30.0, 240.0],
        [45.0, 410.0],
        [20.0, 150.0],
        [60.0, 520.0],
        [35.0, 300.0],
    ];

    let users_c = center(&users);
    println!("Centered users:\n{:.2}", users_c);
    let users_means = users_c.mean_axis(Axis(0)).unwrap();
    println!("Column means after centering: {:.2}\n", users_means);

    let workouts_c = center(&workouts);
    println!("Centered workouts:\n{:.2}", workouts_c);
    let workouts_means = workouts_c.mean_axis(Axis(0)).unwrap();
    println!("Column means after centering: {:.2}", workouts_means);
}
