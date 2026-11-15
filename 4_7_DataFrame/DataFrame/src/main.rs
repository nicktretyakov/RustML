use polars::prelude::*;
use std::io::{self, Write};

/// Макрос для чтения числа из stdin с приглашением к вводу.
/// Убирает дублирование `print! + flush + read_line + parse`.
macro_rules! prompt_number {
    ($prompt:expr) => {{
        print!("{}", $prompt);
        io::stdout().flush().expect("failed to flush stdout");
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("failed to read line");
        input.trim().parse::<i64>().expect("expected an integer")
    }};
}

fn main() -> PolarsResult<()> {
    let plants = ["fern", "cactus", "palm"];
    let prices = [4.5_f64, 2.0, 12.0];

    let sold: Vec<i64> = plants
        .iter()
        .map(|name| prompt_number!(format!("sold {name}: ")))
        .collect();

    // Макрос `df!` из polars — аналог `pd.DataFrame({...})`
    let shop = df!(
        "plant" => &plants[..],
        "price" => &prices[..],
        "sold"  => &sold[..],
    )?;

    println!("{shop}");
    Ok(())
}
