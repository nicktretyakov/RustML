// src/tasks.rs
// Модуль: Децентрализованные вычисления (SOLID: Single Responsibility — задачи и ноды)
// ООП: Node инкапсулирует задачи, методы для добавления/обработки
// Параллельность: Обработка задач асинхронно, Arc<Mutex<>> для безопасного доступа

use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct Task {
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct Node {
    id: u64,
    tasks: Arc<Mutex<Vec<Task>>>,
}

impl Node {
    pub fn new(id: u64) -> Self {
        Self { id, tasks: Arc::new(Mutex::new(Vec::new())) }
    }

    pub fn add_task(&self, task: Task) {
        self.tasks.lock().unwrap().push(task);
    }

    pub async fn process_tasks(&self) {
        let tasks = self.tasks.lock().unwrap().drain(..).collect::<Vec<_>>();
        for task in tasks {
            println!("Node {} processing task: {:?} bytes", self.id, task.data.len());
            // Здесь может быть тяжёлая параллельная обработка (rayon, etc.)
            tokio::task::yield_now().await; // Симуляция асинхронной работы
        }
    }
}