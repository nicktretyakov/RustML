use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub sensor: SensorConfig,
    pub processing: ProcessingConfig,
    pub transmission: TransmissionConfig,
    pub system: SystemConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub enum SensorConfig {
    Simulated { base_range: (f64, f64), noise_range: (f64, f64) },
    Ds18b20 { device_id: String, retries: usize },
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProcessingConfig {
    pub batch_size: usize,
    pub worker_count: usize,
    pub kalman: KalmanConfig,
    pub anomaly_alpha: f64,
    pub anomaly_threshold: f64,
    pub anomaly_min_samples: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KalmanConfig {
    pub initial_estimate: f64,
    pub initial_error: f64,
    pub process_noise: f64,
    pub measurement_noise: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TransmissionConfig {
    pub url: String,
    pub device_id: String,
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SystemConfig {
    pub read_interval_secs: u64,
}