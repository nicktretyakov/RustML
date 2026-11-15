//! Каталог приложений: читаем apps.csv, затем со стандартного ввода —
//! название столбца и число n. Печатаем n приложений с наибольшим
//! значением в этом столбце (по убыванию) и приложение с наименьшим
//! значением.
//!
//! Переписано с Python (pandas) на Rust. Две исходные ошибки сортировки
//! исправлены:
//!
//!  1. В pandas `df.sort_values(...)` возвращает НОВЫЙ DataFrame и не
//!     меняет исходный, поэтому `apps.head(n)` печатал неотсортированные
//!     данные. В Rust мы работаем с явно созданной отсортированной копией.
//!
//!  2. Исходный код искал «Lowest» через `sort_values(..., ascending=False)`
//!     и `iloc[0]` — то есть брал ПЕРВЫЙ элемент по убыванию, что есть
//!     максимум, а не минимум. Здесь минимум ищется напрямую (`min_by`),
//!     без путаницы с направлением сортировки.

use csv::ReaderBuilder;
use serde::Deserialize;
use std::io::{self, BufRead, Write};

/// Одна строка каталога — десериализуется крейтом `csv` + `serde`.
#[derive(Debug, Clone, Deserialize)]
struct App {
    app: String,
    downloads_k: i64,
    rating: f64,
}

/// Строка, дополненная её исходной позицией в CSV (0-индексация,
/// без учёта заголовка). Индекс нужен, чтобы воспроизвести левую
/// колонку вывода, как у pandas.
#[derive(Debug, Clone)]
struct Row {
    idx: usize,
    app: App,
}

/// `top_n!(rows, key, n)` — возвращает `Vec<Row>` из `n` строк с
/// наибольшим значением `key`, отсортированных по убыванию.
///
/// `key` — выражение, принимающее `&Row` и возвращающее `f64`.
/// Сравнение через `partial_cmp` (значения могут быть NaN).
macro_rules! top_n {
    ($rows:expr, $key:expr, $n:expr) => {{
        let mut sorted: Vec<&Row> = $rows.iter().collect();
        // По убыванию: key(b) vs key(a).
        sorted.sort_by(|a, b| {
            $key(b)
                .partial_cmp(&$key(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
            .into_iter()
            .take($n)
            .cloned()
            .collect::<Vec<Row>>()
    }};
}

/// `lowest!(rows, key)` — возвращает `Option<&Row>` с наименьшим `key`.
/// В отличие от исходного кода, здесь направление сортировки не играет
/// роли: минимум ищется напрямую, а не как «первый по убыванию».
macro_rules! lowest {
    ($rows:expr, $key:expr) => {{
        $rows.iter().min_by(|a, b| {
            $key(a)
                .partial_cmp(&$key(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }};
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "apps.csv".to_string());

    let mut rdr = ReaderBuilder::new()
        .from_path(&path)
        .unwrap_or_else(|e| {
            eprintln!("не удалось открыть {}: {}", path, e);
            std::process::exit(1);
        });

    // Читаем весь каталог и запоминаем исходный порядок строк.
    let rows: Vec<Row> = rdr
        .deserialize::<App>()
        .enumerate()
        .map(|(i, r)| Row {
            idx: i,
            app: r.expect("неверная строка CSV"),
        })
        .collect();

    if rows.is_empty() {
        eprintln!("каталог пуст");
        return;
    }

    // Столбец и n — со стандартного ввода, как в исходной программе.
    let stdin = io::stdin();
    let column = {
        let mut s = String::new();
        stdin.lock().read_line(&mut s).expect("введите столбец");
        s.trim().to_string()
    };
    let n: usize = {
        let mut s = String::new();
        stdin.lock().read_line(&mut s).expect("введите n");
        s.trim().parse().expect("n должно быть целым числом")
    };

    // Числовой экстрактор значения столбца. Выбирается по имени столбца.
    let key: Box<dyn Fn(&Row) -> f64> = match column.as_str() {
        "downloads_k" => Box::new(|r: &Row| r.app.downloads_k as f64),
        "rating" => Box::new(|r: &Row| r.app.rating),
        other => {
            eprintln!("неизвестный столбец: {}", other);
            std::process::exit(1);
        }
    };

    // Замыкание-обёртка, чтобы передать `key` в макросы как `Fn`.
    let top = top_n!(rows, |r| key(r), n);
    print_table(&top);

    if let Some(low) = lowest!(rows, |r| key(r)) {
        println!("Lowest: {}", low.app.app);
    }
}

/// Печать таблицы в стиле pandas: каждая колонка выравнивается по
/// правому краю по ширине заголовка и значений, разделитель — 2 пробела,
/// слева колонка с исходным индексом строки (пустой заголовок).
fn print_table(rows: &[Row]) {
    let idx_w = rows
        .iter()
        .map(|r| r.idx.to_string().len())
        .max()
        .unwrap_or(0);

    let app_w = rows
        .iter()
        .map(|r| r.app.app.len())
        .max()
        .unwrap_or(0)
        .max("app".len());

    let dl_w = rows
        .iter()
        .map(|r| r.app.downloads_k.to_string().len())
        .max()
        .unwrap_or(0)
        .max("downloads_k".len());

    let rt_w = rows
        .iter()
        .map(|r| format!("{}", r.app.rating).len())
        .max()
        .unwrap_or(0)
        .max("rating".len());

    let stdout = io::stdout();
    let mut out = stdout.lock();

    writeln!(
        out,
        "{:>iw$}  {:>aw$}  {:>dw$}  {:>rw$}",
        "", "app", "downloads_k", "rating",
        iw = idx_w, aw = app_w, dw = dl_w, rw = rt_w,
    )
    .ok();

    for r in rows {
        writeln!(
            out,
            "{:>iw$}  {:>aw$}  {:>dw$}  {:>rw$}",
            r.idx, r.app.app, r.app.downloads_k, r.app.rating,
            iw = idx_w, aw = app_w, dw = dl_w, rw = rt_w,
        )
        .ok();
    }
}
