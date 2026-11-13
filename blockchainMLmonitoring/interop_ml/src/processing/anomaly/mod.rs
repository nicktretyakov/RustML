pub mod simple_ewa;
pub mod zscore;

use rayon::prelude::*;

pub trait AnomalyModel: Send + Sync {
    fn update(&mut self, value: f64) -> bool;
    fn reset(&mut self);
}

pub struct EnsembleDetector {
    models: Vec<Box<dyn AnomalyModel>>,
}

impl EnsembleDetector {
    pub fn new(models: Vec<Box<dyn AnomalyModel>>) -> Self {
        Self { models }
    }

    pub fn update(&mut self, value: f64) -> (u32, bool) {
        let flags: Vec<bool> = self.models.par_iter_mut()
            .map(|model| model.update(value))
            .collect();

        let count = flags.iter().filter(|&&f| f).count() as u32;
        let is_anomaly = count > self.models.len() as u32 / 2;
        (count, is_anomaly)
    }

    pub fn reset(&mut self) {
        self.models.par_iter_mut().for_each(|m| m.reset());
    }
}