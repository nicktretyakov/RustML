use super::AnomalyModel;

pub struct SimpleEWADetector {
    alpha: f64,
    mean: f64,
    variance: f64,
    threshold: f64,
    min_samples: usize,
    sample_count: usize,
}

impl SimpleEWADetector {
    pub fn new(alpha: f64, threshold: f64, min_samples: usize) -> Self {
        Self {
            alpha,
            mean: 0.0,
            variance: 0.0,
            threshold,
            min_samples,
            sample_count: 0,
        }
    }
}

impl AnomalyModel for SimpleEWADetector {
    fn update(&mut self, value: f64) -> bool {
        if self.sample_count == 0 {
            self.mean = value;
        } else {
            let diff = value - self.mean;
            self.mean += self.alpha * diff;
            let diff2 = value - self.mean;
            self.variance = (1.0 - self.alpha) * (self.variance + self.alpha * diff * diff2);
        }
        self.sample_count += 1;

        let std_dev = self.variance.sqrt();
        let is_anomaly = self.sample_count >= self.min_samples &&
            (value - self.mean).abs() > self.threshold * std_dev;

        is_anomaly
    }

    fn reset(&mut self) {
        self.mean = 0.0;
        self.variance = 0.0;
        self.sample_count = 0;
    }
}