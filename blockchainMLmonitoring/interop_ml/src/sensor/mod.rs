pub mod simulated;
pub mod ds18b20;

use std::sync::Arc;
use crate::app_config::{AppConfig, SensorConfig};

pub trait TemperatureSensor: Send + Sync + 'static {
    fn read(&self) -> Result<f64, SensorError>;
}

#[derive(Debug)]
pub enum SensorError {
    IoError(std::io::Error),
    ParseError(String),
    CrcError,
    DeviceNotFound,
    Timeout,
}

pub struct SensorFactory;

impl SensorFactory {
    pub fn create(config: &AppConfig) -> Arc<dyn TemperatureSensor> {
        match &config.sensor {
            SensorConfig::Simulated { base_range, noise_range } => {
                Arc::new(simulated::SimulatedTemperatureSensor::new(*base_range, *noise_range))
            }
            SensorConfig::Ds18b20 { device_id, retries } => {
                Arc::new(ds18b20::Ds18b20Sensor::new(device_id, *retries))
            }
        }
    }
}