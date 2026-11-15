//! Магазин канцелярских товаров.
//!
//! Читает `stock.csv` (колонки product, price, quantity), добавляет колонку
//! `value = price * quantity`, печатает таблицу и общую стоимость запаса.
//!
//! Демонстрирует:
//! * крейты `csv` (чтение CSV) и `serde` (десериализация записей);
//! * декларативные макросы `line_value!` и `define_product!`.

use csv::Reader;
use serde::Deserialize;

/// Макрос: стоимость позиции = цена × количество.
macro_rules! line_value {
    ($price:expr, $qty:expr) => {
        ($price) * ($qty as f64)
    };
}

/// Макрос: порождает структуру товара с полями product / price / quantity,
/// поддержкой десериализации из CSV (через serde) и методом `value()`.
macro_rules! define_product {
    ($name:ident) => {
        #[derive(Debug, Clone, Deserialize)]
        struct $name {
            product: String,
            price: f64,
            quantity: i64,
        }

        impl $name {
            fn value(&self) -> f64 {
                line_value!(self.price, self.quantity)
            }
        }
    };
}

define_product!(Product);

fn fmt_float(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rdr = Reader::from_path("stock.csv")?;
    let mut items: Vec<Product> = Vec::new();
    for result in rdr.deserialize() {
        let record: Product = result?;
        items.push(record);
    }

    let price_strs: Vec<String> = items.iter().map(|i| fmt_float(i.price)).collect();
    let qty_strs: Vec<String> = items.iter().map(|i| i.quantity.to_string()).collect();
    let val_strs: Vec<String> = items.iter().map(|i| fmt_float(i.value())).collect();

    let idx_w = items.len().to_string().len().max(1);
    let product_w = items.iter().map(|i| i.product.len()).max().unwrap_or(0).max("product".len());
    let price_w = price_strs.iter().map(|s| s.len()).max().unwrap_or(0).max("price".len());
    let qty_w = qty_strs.iter().map(|s| s.len()).max().unwrap_or(0).max("quantity".len());
    let val_w = val_strs.iter().map(|s| s.len()).max().unwrap_or(0).max("value".len());

    println!(
        "{:>iw$} {:>pw$}  {:>prw$}  {:>qw$}  {:>vw$}",
        "", "product", "price", "quantity", "value",
        iw = idx_w, pw = product_w, prw = price_w, qw = qty_w, vw = val_w,
    );

    let mut total = 0.0_f64;
    for (i, it) in items.iter().enumerate() {
        total += it.value();
        println!(
            "{:>iw$} {:>pw$}  {:>prw$}  {:>qw$}  {:>vw$}",
            i, it.product, &price_strs[i], &qty_strs[i], &val_strs[i],
            iw = idx_w, pw = product_w, prw = price_w, qw = qty_w, vw = val_w,
        );
    }
    println!("Total value: {}", fmt_float(total));
    Ok(())
}
