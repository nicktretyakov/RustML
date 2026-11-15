use polars::prelude::*;
use std::io::{self, Write};

fn main() -> PolarsResult<()> {
    // Ввод названия дня (аналог input())
    let mut day = String::new();
    print!("Введите день: ");
    io::stdout().flush().unwrap(); // выталкиваем буфер, чтобы приглашение появилось сразу
    io::stdin().read_line(&mut day)?;
    let day = day.trim();

    // Путь к файлу через макрос format! (аналог f"{day}.csv")
    let path = format!("{day}.csv");

    // Чтение CSV в DataFrame (аналог pd.read_csv)
    let orders = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(path.into()))?
        .finish()?;

    // Печать таблицы (аналог print(orders))
    println!("{orders}");

    Ok(())
}
