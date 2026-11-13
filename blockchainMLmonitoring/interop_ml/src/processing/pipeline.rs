use std::sync::{Arc, atomic::{AtomicU32, Ordering}};
use std::thread;

use crossbeam_channel::{Receiver, Sender};
use crate::app_config::ProcessingConfig;
use super::kalman::KalmanFilter;
use super::anomaly::{EnsembleDetector, simple_ewa::SimpleEWADetector, zscore::ZScoreDetector};

struct Processor {
    kalman: KalmanFilter,
    anomaly: EnsembleDetector,
    buffer: Vec<f64>,
    batch_size: usize,
}

pub struct ProcessingPipeline;

impl ProcessingPipeline {
    pub fn spawn(config: ProcessingConfig, raw_rx: Receiver<f64>, agg_tx: Sender<(f64, u32)>) {
        let total_anomalies = Arc::new(AtomicU32::new(0));

        for _ in 0..config.worker_count {
            let raw_rx = raw_rx.clone();
            let agg_tx = agg_tx.clone();
            let total_anomalies = total_anomalies.clone();
            let config = config.clone();

            thread::spawn(move || {
                let mut processor = Processor {
                    kalman: KalmanFilter::new(
                        config.kalman.initial_estimate,
                        config.kalman.initial_error,
                        config.kalman.process_noise,
                        config.kalman.measurement_noise,
                    ),
                    anomaly: EnsembleDetector::new(vec![
                        Box::new(SimpleEWADetector::new(
                            config.anomaly_alpha,
                            config.anomaly_threshold,
                            config.anomaly_min_samples,
                        )),
                        Box::new(ZScoreDetector::new(
                            config.anomaly_threshold,
                            config.anomaly_min_samples,
                        )),
                    ]),
                    buffer: Vec::with_capacity(config.batch_size),
                    batch_size: config.batch_size,
                };

                loop {
                    match raw_rx.recv() {
                        Ok(raw) => {
                            let filtered = processor.kalman.update(raw);
                            let (new_anoms, is_anomaly) = processor.anomaly.update(filtered);

                            total_anomalies.fetch_add(new_anoms, Ordering::Relaxed);

                            if is_anomaly {
                                continue;
                            }

                            processor.buffer.push(filtered);

                            if processor.buffer.len() >= processor.batch_size {
                                let avg = processor.buffer.iter().sum::<f64>() / processor.batch_size as f64;
                                let current_total = total_anomalies.load(Ordering::Relaxed);
                                let _ = agg_tx.send((avg, current_total));
                                processor.buffer.clear();
                            }
                        }
                        Err(_) => break,
                    }
                }
            });
        }
    }
}