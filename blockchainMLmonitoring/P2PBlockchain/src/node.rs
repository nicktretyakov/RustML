use reqwest::Client;

impl Node {
    async fn ml_validate_tx(&self, tx: &Transaction) -> Result<bool, ()> {
        let client = Client::new();
        let complex_data = tx_to_complex_data(tx); // Ваша функция преобразования

        let resp = client.post("http://interop_ml:3000/validate")
            .json(&json!({ "data": vec![complex_data] }))
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => {
                let body: ValidationResponse = r.json().await.unwrap_or_default();
                // Валидно, если все точки valid && !is_anomaly && lof_score <= 1.5
                Ok(body.results.iter().all(|res| res.valid && !res.is_anomaly && res.lof_score <= 1.5))
            }
            _ => Err(()), // Fallback: ML недоступен
        }
    }
}