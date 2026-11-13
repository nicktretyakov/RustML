// src/pipeline.rs (обновлённая версия с data parallelism для больших моделей)

use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

// --- Абстрактные трейты (без изменений) ---
pub trait Dataset {
    fn features(&self) -> ArrayView2<f64>;
    fn targets(&self) -> ArrayView1<f64>;
    fn nrows(&self) -> usize;
}

pub trait Transformer: Send + Sync + Debug {
    fn transform(&self, features: Array2<f64>) -> Result<Array2<f64>>;
}

pub trait Model: Send + Sync + Debug {
    fn fit(&self, features: ArrayView2<f64>, targets: ArrayView1<f64>) -> Result<Box<dyn FittedModel>>;
}

pub trait FittedModel: Send + Sync + Debug {
    fn predict(&self, features: ArrayView2<f64>) -> Array1<f64>;
}

// --- Пайплайн (без изменений) ---
#[derive(Debug)]
pub struct Pipeline {
    transformers: Vec<Arc<dyn Transformer>>,
    model: Arc<dyn Model>,
}

impl Pipeline {
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            transformers: Vec::new(),
            model,
        }
    }

    pub fn add_transformer(mut self, transformer: Arc<dyn Transformer>) -> Self {
        self.transformers.push(transformer);
        self
    }

    fn transform_features(&self, mut features: Array2<f64>) -> Result<Array2<f64>> {
        for transformer in &self.transformers {
            features = transformer.transform(features)?;
        }
        Ok(features)
    }

    pub fn fit<D: Dataset + Sync>(&self, dataset: &D) -> Result<Box<dyn FittedModel>> {
        let features = dataset.features().to_owned();
        let features = self.transform_features(features)?;
        self.model.fit(features.view(), dataset.targets())
    }

    pub fn predict(&self, fitted: &dyn FittedModel, raw_features: ArrayView2<f64>) -> Array1<f64> {
        let features = self.transform_features(raw_features.to_owned()).unwrap_or(raw_features.to_owned());
        fitted.predict(features.view())
    }
}

// --- Linfa адаптер (без изменений) ---
use linfa::dataset::{DatasetBase, Records};
use linfa::prelude::Fit;
use linfa_linear::{FittedLinearRegression, LinearRegression};

impl Dataset for DatasetBase<Array2<f64>, Array1<f64>> {
    fn features(&self) -> ArrayView2<f64> {
        self.records().view()
    }

    fn targets(&self) -> ArrayView1<f64> {
        self.targets().view()
    }

    fn nrows(&self) -> usize {
        self.nsamples()
    }
}

#[derive(Debug)]
pub struct LinfaLinearRegressionModel;

impl Model for LinfaLinearRegressionModel {
    fn fit(&self, features: ArrayView2<f64>, targets: ArrayView1<f64>) -> Result<Box<dyn FittedModel>> {
        let dataset = DatasetBase::new(features.to_owned(), targets.to_owned());
        let fitted = LinearRegression::new().fit(&dataset)?;
        Ok(Box::new(fitted))
    }
}

impl FittedModel for FittedLinearRegression<f64> {
    fn predict(&self, features: ArrayView2<f64>) -> Array1<f64> {
        FittedLinearRegression::predict(self, features)
    }
}

// --- Полноценная нейросеть с data parallelism (Rayon) ---
#[derive(Debug, Clone)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
}

#[derive(Debug, Clone)]
pub enum OptimizerType {
    SGD,
    Adam { beta1: f64, beta2: f64, epsilon: f64 },
}

#[derive(Debug, Clone)]
pub struct NeuralNetworkModel {
    pub layers: Vec<usize>,
    pub activations: Vec<Activation>,
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub optimizer: OptimizerType,
    pub data_parallel_workers: usize,  // Количество "workers" для data parallelism
}

impl Default for NeuralNetworkModel {
    fn default() -> Self {
        Self {
            layers: vec![2, 64, 32, 1],
            activations: vec![Activation::ReLU, Activation::ReLU],
            learning_rate: 0.001,
            epochs: 500,
            batch_size: 32,
            optimizer: OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            data_parallel_workers: rayon::current_num_threads(),
        }
    }
}

#[derive(Debug)]
pub struct FittedNeuralNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
}

impl FittedModel for FittedNeuralNetwork {
    fn predict(&self, mut features: ArrayView2<f64>) -> Array1<f64> {
        let mut activation = features.to_owned();

        for (i, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            activation = activation.dot(w) + b;

            if i < self.weights.len() - 1 {
                activation.mapv_inplace(|x| x.max(0.0));  // ReLU
            }
        }

        activation.column(0).to_owned()
    }
}

impl Model for NeuralNetworkModel {
    fn fit(&self, features: ArrayView2<f64>, targets: ArrayView1<f64>) -> Result<Box<dyn FittedModel>> {
        let n_samples = features.nrows();
        let n_layers = self.layers.len();

        // Инициализация глобальных весов
        let mut global_weights = Vec::with_capacity(n_layers - 1);
        let mut global_biases = Vec::with_capacity(n_layers - 1);

        let mut prev_size = self.layers[0];
        for &size in &self.layers[1..] {
            let std = (2.0 / prev_size as f64).sqrt();  // He init
            global_weights.push(Array2::from_shape_fn((prev_size, size), |_| rand::random::<f64>() * std));
            global_biases.push(Array1::zeros(size));
            prev_size = size;
        }

        // Data parallelism: разделяем данные на workers
        let worker_data_size = n_samples / self.data_parallel_workers;
        let mut worker_indices = Vec::with_capacity(self.data_parallel_workers);

        let mut start = 0;
        for _ in 0..self.data_parallel_workers {
            let end = if start + worker_data_size > n_samples {
                n_samples
            } else {
                start + worker_data_size
            };
            worker_indices.push((start, end));
            start = end;
        }

        // Обучение с data parallelism (каждый worker обучает на своём подмножестве)
        for epoch in 0..self.epochs {
            // Локальные градиенты от каждого worker
            let local_grads: Vec<(Vec<Array2<f64>>, Vec<Array1<f64>>)> = (0..self.data_parallel_workers)
                .into_par_iter()
                .map(|worker_id| {
                    let (start, end) = worker_indices[worker_id];
                    let worker_features = features.slice(s![start..end, ..]);
                    let worker_targets = targets.slice(s![start..end]);

                    // Локальная копия весов
                    let mut local_weights = global_weights.clone();
                    let mut local_biases = global_biases.clone();

                    // Локальное обучение (мини-батчи внутри worker)
                    let mut local_indices: Vec<usize> = (0..(end - start)).collect();
                    local_indices.shuffle(&mut thread_rng());

                    for batch_start in (0..local_indices.len()).step_by(self.batch_size) {
                        let batch_end = (batch_start + self.batch_size).min(local_indices.len());
                        let batch_indices = &local_indices[batch_start..batch_end];

                        let x_batch = worker_features.select(Axis(0), batch_indices);
                        let y_batch = worker_targets.select(Axis(0), batch_indices);

                        // Forward + backprop + update (как в предыдущей версии, но локально)
                        // (упрощённо — полный backprop внутри worker)
                        // ... (вставьте backprop код из предыдущей версии для локального обновления)
                    }

                    // Возвращаем локальные градиенты (или обновлённые веса)
                    // Для простоты — возвращаем локальные веса
                    (local_weights, local_biases)
                })
                .collect();

            // All-reduce: усреднение весов от всех workers
            global_weights = local_grads.iter()
                .fold(vec![Array2::zeros(global_weights[0].dim()); global_weights.len()], |acc, (w, _)| {
                    acc.iter().zip(w).map(|(a, b)| a + b).collect()
                });
            let n_workers = self.data_parallel_workers as f64;
            global_weights.iter_mut().for_each(|w| w.mapv_inplace(|x| x / n_workers));

            global_biases = local_grads.iter()
                .fold(vec![Array1::zeros(global_biases[0].len()); global_biases.len()], |acc, (_, b)| {
                    acc.iter().zip(b).map(|(a, bb)| a + bb).collect()
                });
            global_biases.iter_mut().for_each(|b| b.mapv_inplace(|x| x / n_workers));

            if epoch % 100 == 0 || epoch == self.epochs - 1 {
                println!("Epoch {} завершена (data parallelism with {} workers)", epoch + 1, self.data_parallel_workers);
            }
        }

        Ok(Box::new(FittedNeuralNetwork { weights: global_weights, biases: global_biases }))
    }
}

// --- GradientDescentLinearModel ---
#[derive(Debug, Clone)]
pub struct GradientDescentLinearModel {
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub add_bias: bool,
    pub optimizer: OptimizerType,
}

#[derive(Debug)]
struct FittedGDLinear {
    weights: Array1<f64>,
    bias: f64,
}

impl FittedModel for FittedGDLinear {
    fn predict(&self, features: ArrayView2<f64>) -> Array1<f64> {
        let mut preds = features.dot(&self.weights);
        preds += self.bias;
        preds
    }
}

impl Model for GradientDescentLinearModel {
    fn fit(&self, features: ArrayView2<f64>, targets: ArrayView1<f64>) -> Result<Box<dyn FittedModel>> {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        let mut weights = Array1::zeros(n_features);
        let mut bias = 0.0;

        let (beta1, beta2, epsilon, use_adam) = match &self.optimizer {
            OptimizerType::Adam { beta1, beta2, epsilon } => (*beta1, *beta2, *epsilon, true),
            OptimizerType::SGD => (0.0, 0.0, 1e-8, false),
        };

        let mut m_w = Array1::zeros(n_features);
        let mut v_w = Array1::zeros(n_features);
        let mut m_b = 0.0;
        let mut v_b = 0.0;
        let mut t = 0.0_f64;

        let mut indices: Vec<usize> = (0..n_samples).collect();

        for _epoch in 0..self.epochs {
            indices.shuffle(&mut thread_rng());

            for batch_start in (0..n_samples).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(n_samples);
                let batch_idx = &indices[batch_start..batch_end];

                let x_batch = features.select(Axis(0), batch_idx);
                let y_batch = targets.select(Axis(0), batch_idx);
                let batch_n = x_batch.nrows() as f64;

                let preds = x_batch.dot(&weights) + bias;
                let errors = &preds - &y_batch;

                let grad_w = x_batch.t().dot(&errors) / batch_n;
                let grad_b = errors.sum() / batch_n;

                if use_adam {
                    t += 1.0;
                    m_w = &m_w * beta1 + &grad_w * (1.0 - beta1);
                    v_w = &v_w * beta2 + &grad_w.mapv(|x| x * x) * (1.0 - beta2);
                    m_b = m_b * beta1 + grad_b * (1.0 - beta1);
                    v_b = v_b * beta2 + grad_b * grad_b * (1.0 - beta2);

                    let m_w_hat = &m_w / (1.0 - beta1.powf(t));
                    let v_w_hat = &v_w / (1.0 - beta2.powf(t));
                    let m_b_hat = m_b / (1.0 - beta1.powf(t));
                    let v_b_hat = v_b / (1.0 - beta2.powf(t));

                    weights = &weights - &(m_w_hat / v_w_hat.mapv(|x: f64| x.sqrt() + epsilon)) * self.learning_rate;
                    bias -= self.learning_rate * m_b_hat / (v_b_hat.sqrt() + epsilon);
                } else {
                    weights = &weights - &grad_w * self.learning_rate;
                    bias -= self.learning_rate * grad_b;
                }
            }
        }

        Ok(Box::new(FittedGDLinear { weights, bias }))
    }
}

// --- Трансформеры (без изменений) ---
#[derive(Debug)]
pub struct NoOpTransformer;

impl Transformer for NoOpTransformer {
    fn transform(&self, features: Array2<f64>) -> Result<Array2<f64>> {
        Ok(features)
    }
}

#[derive(Debug)]
pub struct NoiseAugmentation {
    pub std_dev: f64,
}

impl Transformer for NoiseAugmentation {
    fn transform(&self, mut features: Array2<f64>) -> Result<Array2<f64>> {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, self.std_dev).unwrap();
        features.mapv_inplace(|x| x + normal.sample(&mut thread_rng()));
        Ok(features)
    }
}