use rand::Rng;
use super::{TemperatureSensor, SensorError};

pub struct SimulatedTemperatureSensor {
    base_range: (f64, f64),
    noise_range: (f64, f64),
}

impl SimulatedTemperatureSensor {
    pub fn new(base_range: (f64, f64), noise_range: (f64, f64)) -> Self {
        Self { base_range, noise_range }
    }
}

impl TemperatureSensor for SimulatedTemperatureSensor {
    fn read(&self) -> Result<f64, SensorError> {
        let mut rng = rand::thread_rng();
        let base = rng.gen_range(self.base_range.0..self.base_range.1);
        let noise = rng.gen_range(self.noise_range.0..self.noise_range.1);
        Ok(base + noise)
    }
}