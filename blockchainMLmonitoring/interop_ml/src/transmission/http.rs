use reqwest::blocking::Client;
use serde::Serialize;
use chrono::Utc;
use crate::app_config::TransmissionConfig;

#[derive(Serialize)]
struct Payload {
    average_temperature: f64,
    timestamp: String,
    device: String,
    anomaly_count: u32,
}

pub struct HttpTransmitter {
    client: Client,
    url: String,
    device_id: String,
}

impl HttpTransmitter {
    pub fn new(config: &TransmissionConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .unwrap();

        Self {
            client,
            url: config.url.clone(),
            device_id: config.device_id.clone(),
        }
    }

    pub fn transmit(&self, temperature: f64, anomaly_count: u32) -> Result<(), reqwest::Error> {
        let payload = Payload {
            average_temperature: temperature,
            timestamp: Utc::now().to_rfc3339(),
            device: self.device_id.clone(),
            anomaly_count,
        };

        self.client
            .post(&self.url)
            .json(&payload)
            .send()?
            .error_for_status()?;
        Ok(())
    }
}