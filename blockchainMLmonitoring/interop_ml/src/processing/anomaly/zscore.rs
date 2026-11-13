use super::AnomalyModel;

pub struct ZScoreDetector {
    threshold: f64,
    min_samples: usize,
    sample_count: usize,
    sum: f64,
    sum_sq: f64,
}

impl ZScoreDetector {
    pub fn new(threshold: f64, min_samples: usize) -> Self {
        Self {
            threshold,
            min_samples,
            sample_count: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }
}

impl AnomalyModel for ZScoreDetector {
    fn update(&mut self, value: f64) -> bool {
        self.sample_count += 1;
        self.sum += value;
        self.sum_sq += value * value;

        if self.sample_count < self.min_samples {
            return false;
        }

        let mean = self.sum / self.sample_count as f64;
        let variance = (self.sum_sq / self.sample_count as f64) - mean * mean;
        let std_dev = variance.sqrt().max(1e-9);
        let z = (value - mean).abs() / std_dev;

        z > self.threshold
    }

    fn reset(&mut self) {
        self.sample_count = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }
}