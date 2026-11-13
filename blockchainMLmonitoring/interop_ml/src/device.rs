use std::thread;
use std::time::Duration;
use std::sync::Arc;

use crossbeam_channel::Sender;
use tracing::info;
use crate::sensor::TemperatureSensor;
use crate::app_config::AppConfig;

pub struct EdgeDevice {
    sensor: Arc<dyn TemperatureSensor>,
    raw_tx: Sender<f64>,
    config: AppConfig,
}

impl EdgeDevice {
    pub fn new(sensor: Arc<dyn TemperatureSensor>, raw_tx: Sender<f64>, config: AppConfig) -> Self {
        Self { sensor, raw_tx, config }
    }

    pub fn spawn_sensor_thread(&self) {
        let sensor = self.sensor.clone();
        let tx = self.raw_tx.clone();
        let interval = Duration::from_secs(self.config.system.read_interval_secs);

        thread::spawn(move || loop {
            match sensor.read() {
                Ok(temp) => {
                    info!("Считано: {:.2} °C", temp);
                    let _ = tx.send(temp);
                }
                Err(e) => tracing::error!("Ошибка сенсора: {:?}", e),
            }
            thread::sleep(interval);
        });
    }
}