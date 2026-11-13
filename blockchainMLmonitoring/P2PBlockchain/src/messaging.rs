// src/messaging.rs
// Модуль: Система чата/мессенджера (SOLID: Single Responsibility — только рассылка сообщений)
// ООП: Инкапсуляция, методы для добавления пользователей и broadcast
// Параллельность: Асинхронный broadcast через tokio::spawn + mpsc

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct Message {
    pub sender: String,
    pub content: String,
}

#[derive(Debug)]
struct User {
    pub name: String,
    pub sender: mpsc::Sender<Message>,
}

#[derive(Default)]
pub struct ServerState {
    users: HashMap<String, Arc<Mutex<User>>>,
}

impl ServerState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_user(&mut self, name: String, sender: mpsc::Sender<Message>) {
        let user = User { name: name.clone(), sender };
        self.users.insert(name, Arc::new(Mutex::new(user)));
    }

    pub fn broadcast(&self, message: Message) {
        for user in self.users.values() {
            let user = user.clone();
            let message = message.clone();
            tokio::spawn(async move {
                let sender = { user.lock().unwrap().sender.clone() };
                let _ = sender.send(message).await;
            });
        }
    }
}