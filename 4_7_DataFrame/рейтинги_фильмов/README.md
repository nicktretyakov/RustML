Что делает программа

    Читает файл movies.csv (с заголовком).

    В lazy-режиме вычисляет агрегаты по столбцам rating и votes:

        count — количество фильмов;

        min, max — минимальный и максимальный рейтинг;

        mean — средний рейтинг;

        median_votes — медиана числа голосов.

    Материализует результат (collect()).

    Извлекает скалярные значения и печатает их в консоль.

Требования

    Rust (edition 2021 или новее) — установка через rustup.

    Crate polars с фичами для чтения CSV и lazy-API.

Пример Cargo.toml:
toml

[package]
name = "movies-stats"
version = "0.1.0"
edition = "2021"

[dependencies]
polars = { version = "0.4x", features = ["lazy", "csv"] }

    Версию polars подберите актуальную — API между минорными версиями может отличаться.

Формат входного файла

Файл movies.csv должен лежать в рабочей директории (рядом с бинарником или в корне проекта) и содержать как минимум следующие столбцы:
Столбец	Тип	Описание
rating	float	Рейтинг фильма
votes	integer	Количество голосов

Остальные столбцы игнорируются. Пример:
csv

title,rating,votes
The Shawshank Redemption,9.3,2800000
The Godfather,9.2,1900000
Pulp Fiction,8.9,2100000

Сборка и запуск
bash

cargo build --release
cargo run --release

Или напрямую:
bash

cargo run --release -- movies.csv

    В текущей версии путь "movies.csv" захардкожен в коде — при необходимости поменяйте его на аргумент командной строки (см. раздел «Возможные улучшения»).

Пример вывода
text

Movies: 500
Ratings run from 1.1 to 9.8
Average rating: 6.85
Half the movies have over 12000 votes

Разбор кода
1. Чтение CSV
rust

let df = CsvReadOptions::default()
    .with_has_header(true)
    .try_into_reader_with_file_path(Some("movies.csv".into()))?
    .finish()?;

try_into_reader_with_file_path открывает файл и возвращает reader, а finish() читает его в DataFrame.
2. Lazy-агрегация
rust

let stats = df.lazy()
    .select([
        col("rating").count().alias("count"),
        col("rating").min().alias("min"),
        col("rating").max().alias("max"),
        col("rating").mean().alias("mean"),
        col("votes").median().alias("median_votes"),
    ])
    .collect()?;

df.lazy() включает ленивый движок Polars: выражения агрегации объединяются в один план и оптимизируются перед выполнением. collect() материализует результат в DataFrame из одной строки.
3. Извлечение значений
rust

let count = stats.column("count")?.get(0)?.try_extract::<u32>()?;
let min   = stats.column("min")?.get(0)?.try_extract::<f64>()?;
// ...
let median_votes = stats.column("median_votes")?.get(0)?.try_extract::<f64>()? as i64;

column(name) находит столбец по имени, get(0) берёт первое (единственное) значение, try_extract::<T>() приводит тип. median возвращает f64, поэтому для median_votes выполняется приведение к i64.
4. Вывод

println! с форматированием {:.2} для среднего рейтинга.

