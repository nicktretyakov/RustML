use ndarray::prelude::*;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Количество минут чтения за каждый день (3 недели × 7 дней)
    let minutes = array![
        20, 35, 15,  0, 40, 60, 45,
        25, 30, 20, 10, 35, 50, 55,
        30, 30, 25, 20, 45, 40, 70,
    ];

    // Ввод номера недели
    print!("Enter week: ");
    io::stdout().flush()?;                       // макрос io::stdout + flush
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let week: usize = input.trim().parse()?;

    // reshape(-1, 7) → (3, 7), row-major порядок как в NumPy
    let weeks: Array2<i32> = minutes.into_shape_with_order((3, 7))?;

    println!("Weeks:\n{:?}", weeks);

    // weeks.sum(axis=1)
    let weekly_totals: Vec<i32> = weeks.outer_iter().map(|row| row.sum()).collect();
    println!("Weekly totals: {:?}", weekly_totals);

    // weeks[week - 1]
    println!("Week {}: {:?}", week, weeks.row(week - 1));

    Ok(())
}
