// src/market.rs (без contextml — ручная регрессия, как ранее)
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Product {
    pub name: String,
    pub price: f64,
}

#[derive(Default)]
pub struct MarketService {
    products: std::collections::HashMap<String, Product>,
    histories: std::collections::HashMap<String, Vec<f64>>,
}

impl MarketService {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_product(&mut self, name: String, price: f64) {
        let product = Product { name: name.clone(), price };
        self.products.insert(name.clone(), product);
        self.histories.entry(name.clone()).or_insert_with(Vec::new).push(price);
        println!("Market: added product {} at price {}", name, price);
    }

    pub fn get_products(&self) -> Vec<Product> {
        self.products.values().cloned().collect()
    }

    pub fn predict_next_price(&self, name: &str) -> f64 {
        let hist = match self.histories.get(name) {
            Some(h) if h.len() >= 2 => h,
            Some(h) => return *h.last().unwrap_or(&0.0),
            None => return 0.0,
        };

        let n = hist.len() as f64;
        let x: Vec<f64> = (0..hist.len()).map(|i| i as f64).collect();
        let y = hist.clone();

        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|v| v * v).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return *y.last().unwrap_or(&0.0);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        let next_x = n;
        let prediction = slope * next_x + intercept;

        println!("Market: predicted next price for {}: {:.2}", name, prediction);

        prediction
    }
}