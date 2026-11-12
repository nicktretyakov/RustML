use ndarray::{array, concatenate, stack, Axis};

fn main() {
    // Первая группа: возраст каждого человека и ежедневное время перед экраном в часах
    let ages_1 = array![23, 35, 41];
    let hours_1 = array![6, 4, 3];
    // Вторая группа
    let ages_2 = array![19, 52];
    let hours_2 = array![7, 2];

    // Для исследования нужна одна строка на каждого человека: сначала возраст, затем количество часов.
    let ages = concatenate![Axis(0), ages_1, ages_2];
    let hours = concatenate![Axis(0), hours_1, hours_2];
    let table = stack![Axis(1), ages, hours];

    // Не меняйте строки ниже
    println!("{:?}", table);
    println!(
        "People: {}, measurements each: {}",
        table.shape()[0],
        table.shape()[1]
    );
}
