use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct IoTNetwork {
    devices: HashMap<String, f64>, // device_id -> current temperature
}

impl IoTNetwork {
    pub fn new() -> Self {
        let mut devices = HashMap::new();
        devices.insert("sensor_1".to_string(), 23.5);
        Self { devices }
    }

    pub fn add_device(&mut self, device_id: &str) {
        self.devices.entry(device_id.to_string()).or_insert(20.0);
    }

    pub fn update_device_data(&mut self, device_id: &str, new_data: &str) {
        if let Ok(temp) = new_data.parse::<f64>() {
            self.devices.insert(device_id.to_string(), temp);
        }
    }

    pub fn get_current_temperature(&self) -> f64 {
        self.devices.get("sensor_1").copied().unwrap_or(20.0)
    }
}