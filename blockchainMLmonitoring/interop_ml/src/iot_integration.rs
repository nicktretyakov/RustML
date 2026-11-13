use reqwest;
use serde_json::Value;

pub async fn fetch_temperature() -> Result<f64, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get("http://localhost:8081/iot/temperature") // Явно GET
        .send()
        .await
        .map_err(|e| format!("Ошибка передачи: {}", e))?;

    if resp.status().is_success() {
        let json: Value = resp.json().await.map_err(|e| format!("JSON error: {}", e))?;
        json["temperature"]
            .as_f64()
            .ok_or("No temperature field".to_string())
    } else {
        Err(format!("HTTP error: {}", resp.status()))
    }
}