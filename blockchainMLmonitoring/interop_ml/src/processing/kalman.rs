#[derive(Clone)]
pub struct KalmanFilter {
    estimate: f64,
    error_cov: f64,
    process_noise: f64,
    measurement_noise: f64,
}

impl KalmanFilter {
    pub fn new(initial_estimate: f64, initial_error: f64, process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            estimate: initial_estimate,
            error_cov: initial_error,
            process_noise,
            measurement_noise,
        }
    }

    pub fn update(&mut self, measurement: f64) -> f64 {
        self.error_cov += self.process_noise;
        let kalman_gain = self.error_cov / (self.error_cov + self.measurement_noise);
        self.estimate += kalman_gain * (measurement - self.estimate);
        self.error_cov *= 1.0 - kalman_gain;
        self.estimate
    }
}