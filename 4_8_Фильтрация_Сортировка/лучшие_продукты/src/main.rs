use anyhow::{Result, bail};
use polars::prelude::*;

/// Анализ выручки по товарам заданной категории.
///
/// Шаги (перенос pandas-задания на Rust + polars):
///   1. Загрузить products.csv.
///   2. Добавить столбец revenue = price * sold.
///   3. Оставить товары выбранной категории.
///   4. Отсортировать по revenue по убыванию.
///   5. Вывести топ-3 (только product и revenue), сохранив исходные индексы строк.
fn main() -> Result<()> {
    // Категория читается в `category` из первого аргумента командной строки,
    // по умолчанию — "garden" (как в примере задания).
    let args: Vec<String> = std::env::args().collect();
    let category = args.get(1).cloned().unwrap_or_else(|| String::from("garden"));
    let csv_path = args.get(2).cloned().unwrap_or_else(|| String::from("products.csv"));

    eprintln!("Категория: {category}");
    eprintln!("Файл:      {csv_path}");

    // 1. Загрузка CSV через polars lazy API.
    let lf = LazyCsvReader::new(PlRefPath::new(&csv_path))
        .with_has_header(true)
        .finish()
        .map_err(|e| anyhow::anyhow!("не удалось открыть CSV: {csv_path}: {e}"))?;

    // Категория должна быть непустой — иначе фильтр бессмысленен.
    if category.is_empty() {
        bail!("категория не задана: передайте её первым аргументом программы");
    }

    // 2-5. Вычисляем revenue, фильтруем по категории, сортируем по убыванию,
    //      берём нужные столбцы и ограничиваем топ-3.
    //
    // with_row_index сохраняет исходный порядковый номер строки файла,
    // благодаря чему после сортировки видны индексы 1, 2, 0 — как в pandas.
    let top: DataFrame = lf
        .with_row_index("idx", None)
        .with_columns(vec![(col("price") * col("sold")).alias("revenue")])
        .filter(col("category").eq(lit(category.clone())))
        .sort(
            vec!["revenue"],
            SortMultipleOptions::default().with_order_descending(true),
        )
        .select(vec![col("idx"), col("product"), col("revenue")])
        .limit(3)
        .collect()
        .map_err(|e| anyhow::anyhow!("ошибка при выполнении конвейера polars: {e}"))?;

    if top.height() == 0 {
        bail!("в категории `{category}` нет товаров");
    }

    // Печать в стиле pandas: левая колонка — индекс, затем product и revenue,
    // выровненные по правому краю по ширине заголовка/значений.
    print_pandas_style(&top)?;
    Ok(())
}

/// Печатает DataFrame (колонки idx, product, revenue) в виде, повторяющем
/// вывод `print(df.head(3))` в pandas:
///
///   ```text
///     product  revenue
///   1    Hose     1500
///   2   Seeds      800
///   0    Rake      600
///   ```
fn print_pandas_style(df: &DataFrame) -> Result<()> {
    let idx = df.column("idx")?.u32()?;
    let product = df.column("product")?.str()?;
    let revenue = df.column("revenue")?.i64()?;

    let n = df.height();
    let mut idx_strs: Vec<String> = Vec::with_capacity(n);
    let mut prod_strs: Vec<String> = Vec::with_capacity(n);
    let mut rev_strs: Vec<String> = Vec::with_capacity(n);

    for i in 0..n {
        idx_strs.push(idx.get(i).map(|v| v.to_string()).unwrap_or_default());
        prod_strs.push(product.get(i).map(|s| s.to_string()).unwrap_or_default());
        rev_strs.push(revenue.get(i).map(|v| v.to_string()).unwrap_or_default());
    }

    // Ширина колонки = max(длина заголовка, max длина значения).
    // У колонки индекса «заголовок» пустой (как в pandas для безымянного индекса).
    let w_i = max_len(std::iter::once("").chain(idx_strs.iter().map(String::as_str)));
    let w_p = max_len(std::iter::once("product").chain(prod_strs.iter().map(String::as_str)));
    let w_r = max_len(std::iter::once("revenue").chain(rev_strs.iter().map(String::as_str)));

    // Заголовок: пустой индекс, выровненный по правому краю, затем заголовки колонок.
    println!("{:>w_i$} {:>w_p$}  {:>w_r$}", "", "product", "revenue");

    // Строки данных: индекс, product, revenue, выровненные по правому краю.
    for i in 0..n {
        println!("{:>w_i$} {:>w_p$}  {:>w_r$}", idx_strs[i], prod_strs[i], rev_strs[i]);
    }
    Ok(())
}

/// Вычисляет максимальную длину строки среди переданных значений.
fn max_len<'a, I>(items: I) -> usize
where
    I: IntoIterator<Item = &'a str>,
{
    items.into_iter().map(str::len).max().unwrap_or(0)
}
