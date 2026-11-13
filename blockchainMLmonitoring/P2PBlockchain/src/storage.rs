// src/storage.rs
// Модуль: Децентрализованное хранилище (SOLID: Single Responsibility — key-value storage)
// ООП: Инкапсуляция данных, методы для store/retrieve
// Параллельность: Arc<Mutex<>> для потокобезопасности

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct DecentralizedStorage {
    data: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

impl DecentralizedStorage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn store_data(&self, key: String, value: Vec<u8>) {
        self.data.lock().unwrap().insert(key, value);
    }

    pub fn retrieve_data(&self, key: &str) -> Option<Vec<u8>> {
        self.data.lock().unwrap().get(key).cloned()
    }
}