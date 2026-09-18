//! Библиотека cat_rl_solid
//!
//! Содержит:
//! - Безопасную структуру `Cat` (Rust API)
//! - Генерацию синтетических данных
//! - Полноценную RL-среду (Gym-like)
//! - Функции активации и Layer Normalization
//! - Простую нейросеть (Dense + Policy Network)
//! - FFI-слой для вызова из Python / C
//!
//! Сборка:
//! ```bash
//! cargo build --release
//! # → target/release/libcat_rl_solid.so
//! ```

use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::os::raw::c_char;
use std::ptr;
use rand::Rng;
use rand::seq::SliceRandom;

// =============================================
// 0. ОБРАБОТКА ОШИБОК (AppError)
// =============================================

/// Централизованный тип ошибок ML-пайплайна
#[derive(Debug)]
pub enum AppError {
    Io(std::io::Error),
    Parse(String),
    Validation(String),
    DimensionMismatch(String),
    EmptyTensor(String),
    InvalidState(String),
    Preprocessing(String),
    Postprocessing(String),
    AugmentationError(String),
    SpaceError(String),
    SerializationError(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Io(e) => write!(f, "Ошибка ввода/вывода: {}", e),
            AppError::Parse(m) => write!(f, "Ошибка парсинга: {}", m),
            AppError::Validation(m) => write!(f, "Ошибка валидации: {}", m),
            AppError::DimensionMismatch(m) => write!(f, "Несовпадение размерностей: {}", m),
            AppError::EmptyTensor(m) => write!(f, "Пустой тензор: {}", m),
            AppError::InvalidState(m) => write!(f, "Некорректное состояние: {}", m),
            AppError::Preprocessing(m) => write!(f, "Ошибка предобработки: {}", m),
            AppError::Postprocessing(m) => write!(f, "Ошибка постобработки: {}", m),
            AppError::AugmentationError(m) => write!(f, "Ошибка аугментации: {}", m),
            AppError::SpaceError(m) => write!(f, "Ошибка пространства RL: {}", m),
            AppError::SerializationError(m) => write!(f, "Ошибка сериализации: {}", m),
        }
    }
}

impl Error for AppError {}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::Io(e)
    }
}

impl From<String> for AppError {
    fn from(s: String) -> Self {
        AppError::Validation(s)
    }
}

// =============================================
// 1. КОНСТАНТЫ И ИМЕНА ДЛЯ СИНТЕТИКИ
// =============================================

const NAMES: &[&str] = &[
    "Felix", "Tom", "Garfield", "Luna", "Simba",
    "Whiskers", "Murka", "Barsik", "Nala", "Oliver",
];

// =============================================
// 2. БЕЗОПАСНЫЙ RUST API — CAT
// =============================================

/// Основная структура кота (безопасная)
#[derive(Debug, Clone)]
pub struct Cat {
    pub name: String,
    pub lives: u8,
}

impl Cat {
    /// Создаёт кота с валидацией
    pub fn new(name: impl Into<String>, lives: u8) -> Result<Self, String> {
        let name = name.into();
        if name.is_empty() {
            return Err("Имя кота не может быть пустым".into());
        }
        if lives == 0 || lives > 9 {
            return Err(format!(
                "Количество жизней должно быть от 1 до 9, получено: {}",
                lives
            ));
        }
        Ok(Self { name, lives })
    }

    /// Случайный кот (для синтетических данных и RL)
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        let name = NAMES.choose(&mut rng).unwrap().to_string();
        let lives = rng.gen_range(1..=9);
        Self::new(name, lives).unwrap()
    }

    pub fn meow(&self) {
        println!("{} says meow! (lives: {})", self.name, self.lives);
    }

    pub fn is_alive(&self) -> bool {
        self.lives > 0
    }

    pub fn lives(&self) -> u8 {
        self.lives
    }

    /// Нормализованное состояние для RL / ML
    pub fn get_state(&self) -> Vec<f32> {
        vec![self.lives as f32 / 9.0]
    }

    /// Строка для CSV
    pub fn to_csv_line(&self) -> String {
        format!("{},{}\n", self.name, self.lives)
    }

    /// Интерпретация вероятностей (postprocessing)
    pub fn interpret_prediction(&self, probabilities: &[f32]) -> Result<String, String> {
        if probabilities.len() < 2 {
            return Err("Ожидалось минимум 2 вероятности [happy, grumpy]".into());
        }
        let (class, prob) = if probabilities[0] > probabilities[1] {
            ("happy", probabilities[0])
        } else {
            ("grumpy", probabilities[1])
        };
        Ok(format!(
            "{} is {} with confidence {:.2}",
            self.name, class, prob
        ))
    }

    // --- Действия среды ---
    pub fn feed(&mut self) -> (bool, f32) {
        if self.lives < 9 {
            self.lives += 1;
            (true, 1.0)
        } else {
            (false, -1.0)
        }
    }

    pub fn play(&mut self) -> (bool, f32) {
        if self.lives > 1 {
            self.lives -= 1;
            (true, 0.5)
        } else {
            (false, -2.0)
        }
    }
}

// =============================================
// 3. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ
// =============================================

/// Генератор синтетических котов для обучения / валидации
pub struct SyntheticDataGenerator {
    pub num_samples: usize,
}

impl SyntheticDataGenerator {
    pub fn new(num_samples: usize) -> Self {
        Self { num_samples }
    }

    pub fn generate(&self) -> Vec<Cat> {
        (0..self.num_samples).map(|_| Cat::random()).collect()
    }

    pub fn save_to_csv(&self, path: &str) -> Result<(), std::io::Error> {
        let mut csv_data = String::from("name,lives\n");
        for cat in self.generate() {
            csv_data.push_str(&cat.to_csv_line());
        }
        std::fs::write(path, csv_data)
    }

    pub fn to_csv_string(&self) -> String {
        let mut csv_data = String::from("name,lives\n");
        for cat in self.generate() {
            csv_data.push_str(&cat.to_csv_line());
        }
        csv_data
    }
}

// =============================================
// 4. ПРОСТРАНСТВА RL + СРЕДА (Gym-like)
// =============================================

/// Пространство наблюдений / действий (как в Gymnasium)
#[derive(Clone, Debug)]
pub enum Space {
    /// Дискретное: 0..n-1
    Discrete { n: usize },
    /// Непрерывное: [low, high]
    Box {
        low: Vec<f32>,
        high: Vec<f32>,
    },
}

impl Space {
    pub fn sample(&self) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        match self {
            Space::Discrete { n } => {
                let action = rng.gen_range(0..*n);
                vec![action as f32]
            }
            Space::Box { low, high } => {
                let mut sample = vec![0.0; low.len()];
                for (i, (&l, &h)) in low.iter().zip(high.iter()).enumerate() {
                    sample[i] = l + (h - l) * rng.gen::<f32>();
                }
                sample
            }
        }
    }

    pub fn contains(&self, value: &[f32]) -> bool {
        match self {
            Space::Discrete { n } => {
                value.len() == 1 && (value[0] as usize) < *n && value[0] >= 0.0
            }
            Space::Box { low, high } => {
                if value.len() != low.len() {
                    return false;
                }
                for (i, &v) in value.iter().enumerate() {
                    if v < low[i] || v > high[i] {
                        return false;
                    }
                }
                true
            }
        }
    }

    pub fn description(&self) -> String {
        match self {
            Space::Discrete { n } => format!("Discrete(n={})", n),
            Space::Box { low, high } => format!("Box(low={:?}, high={:?})", low, high),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Feed = 0,
    Play = 1,
}

impl Action {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Action::Feed),
            1 => Some(Action::Play),
            _ => None,
        }
    }

    pub fn as_usize(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone)]
pub struct StepResult {
    pub observation: Vec<f32>,
    pub reward: f32,
    pub done: bool,
    pub info: String,
}

#[derive(Clone)]
pub struct CatEnv {
    cat: Cat,
    step_count: usize,
    max_steps: usize,
    pub observation_space: Space,
    pub action_space: Space,
}

impl CatEnv {
    pub fn new(cat: Cat) -> Self {
        Self {
            cat,
            step_count: 0,
            max_steps: 50,
            observation_space: Space::Box {
                low: vec![0.0],
                high: vec![1.0],
            },
            action_space: Space::Discrete { n: 2 },
        }
    }

    pub fn random() -> Self {
        Self::new(Cat::random())
    }

    pub fn with_name(name: impl Into<String>) -> Result<Self, String> {
        Ok(Self::new(Cat::new(name, 9)?))
    }

    pub fn observation_space(&self) -> &Space {
        &self.observation_space
    }

    pub fn action_space(&self) -> &Space {
        &self.action_space
    }

    pub fn reset(&mut self) -> Vec<f32> {
        self.cat.lives = 9;
        self.step_count = 0;
        self.cat.get_state()
    }

    pub fn step(&mut self, action: Action) -> StepResult {
        // Проверка валидности действия
        if !self.action_space.contains(&[action.as_usize() as f32]) {
            return StepResult {
                observation: self.cat.get_state(),
                reward: -10.0,
                done: true,
                info: "Invalid action".to_string(),
            };
        }

        self.step_count += 1;

        let (success, reward) = match action {
            Action::Feed => self.cat.feed(),
            Action::Play => self.cat.play(),
        };

        let done = !self.cat.is_alive() || self.step_count >= self.max_steps;

        let info = format!(
            "action={:?}, success={}, lives={}, step={}",
            action, success, self.cat.lives, self.step_count
        );

        StepResult {
            observation: self.cat.get_state(),
            reward,
            done,
            info,
        }
    }

    pub fn observation(&self) -> Vec<f32> {
        self.cat.get_state()
    }

    pub fn is_done(&self) -> bool {
        !self.cat.is_alive() || self.step_count >= self.max_steps
    }

    pub fn cat(&self) -> &Cat {
        &self.cat
    }

    pub fn cat_mut(&mut self) -> &mut Cat {
        &mut self.cat
    }

    pub fn interactive_loop(&mut self) {
        use std::io::{self, Write};

        println!("=== Интерактивная RL-среда (CatEnv) ===");
        println!("Действия: 0 = feed, 1 = play");
        println!("Кот: {} (lives = {})\n", self.cat.name, self.cat.lives);

        while !self.is_done() {
            print!("State: lives = {} | Action (0/1): ", self.cat.lives);
            io::stdout().flush().unwrap();

            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() {
                break;
            }

            let action = match input.trim().parse::<u8>() {
                Ok(0) => Action::Feed,
                Ok(1) => Action::Play,
                _ => {
                    println!("Неизвестное действие, пропускаем");
                    continue;
                }
            };

            let result = self.step(action);
            println!(
                "  → reward = {:.1}, done = {}, info = {}",
                result.reward, result.done, result.info
            );
        }

        println!("\nЭпизод завершён после {} шагов.", self.step_count);
        self.cat.meow();
    }

    pub fn run_random_episode(&mut self, max_steps: usize) -> (f32, usize) {
        self.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;
        let mut rng = rand::thread_rng();

        while !self.is_done() && steps < max_steps {
            let action = if rng.gen_bool(0.5) {
                Action::Feed
            } else {
                Action::Play
            };
            let result = self.step(action);
            total_reward += result.reward;
            steps += 1;
            if result.done {
                break;
            }
        }
        (total_reward, steps)
    }
}

// =============================================
// 5. ФУНКЦИИ АКТИВАЦИИ
// =============================================

/// ReLU: max(0, x)
pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// Leaky ReLU: x if x > 0 else α·x
pub fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 { x } else { alpha * x }
}

/// Sigmoid: 1 / (1 + e^(-x))
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Tanh
pub fn tanh_activation(x: f32) -> f32 {
    x.tanh()
}

/// Softmax (численно устойчивый)
pub fn softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let max_val = *input
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum_exp).collect()
}

/// Применить функцию активации ко всему вектору
pub fn apply_activation(mut layer: Vec<f32>, activation: fn(f32) -> f32) -> Vec<f32> {
    for x in &mut layer {
        *x = activation(*x);
    }
    layer
}

// =============================================
// 6. LAYER NORMALIZATION
// =============================================

/// Layer Normalization:
/// 1. mean / variance по нейронам слоя
/// 2. (x - mean) / std
/// 3. gamma * norm + beta
pub fn layer_normalization(
    input: &[f32],
    gamma: f32,
    beta: f32,
    epsilon: f32,
) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let n = input.len() as f32;
    let mean: f32 = input.iter().sum::<f32>() / n;
    let variance: f32 = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std_dev = (variance + epsilon).sqrt();
    input
        .iter()
        .map(|&x| gamma * (x - mean) / std_dev + beta)
        .collect()
}

/// LayerNorm с параметрами по умолчанию (γ=1, β=0, ε=1e-5)
pub fn layer_norm_default(input: &[f32]) -> Vec<f32> {
    layer_normalization(input, 1.0, 0.0, 1e-5)
}

// =============================================
// 7. ТИПЫ ДАННЫХ ТЕНЗОРОВ (DType)
// =============================================

/// Тип данных тензора (эмуляция tch::Kind / burn::tensor::DType)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DType {
    /// Float32 — полная точность (4 байта)
    Float32,
    /// Float16 — половинная точность (2 байта, Tensor Cores)
    Float16,
    /// Int8 — квантизация (1 байт, inference на edge)
    Int8,
}

impl DType {
    /// Сколько байт занимает один элемент
    pub fn bytes_per_element(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float16 => 2,
            DType::Int8 => 1,
        }
    }

    /// Поддерживает ли ускорение на GPU / TPU / Metal
    pub fn supports_accelerator(&self) -> bool {
        match self {
            DType::Float32 | DType::Float16 | DType::Int8 => true,
        }
    }

    /// Является ли тип квантизированным
    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::Int8)
    }

    /// Симуляция приведения типа (cast)
    pub fn cast_value(&self, value: f32) -> f32 {
        match self {
            DType::Float32 => value,
            DType::Float16 => {
                // Грубая имитация half-precision
                (value * 2048.0).round() / 2048.0
            }
            DType::Int8 => {
                // Имитация INT8-квантизации
                value.clamp(-127.0, 127.0).round()
            }
        }
    }

    /// Привести весь вектор
    pub fn cast_vec(&self, data: &[f32]) -> Vec<f32> {
        data.iter().map(|&x| self.cast_value(x)).collect()
    }

    /// Описание для отладки
    pub fn description(&self) -> &'static str {
        match self {
            DType::Float32 => "Float32 (4 байта, полная точность)",
            DType::Float16 => "Float16 (2 байта, Tensor Cores / TPU)",
            DType::Int8 => "Int8 (1 байт, квантизация)",
        }
    }
}

// =============================================
// 8. ТЕНЗОР + АЛГОРИТМЫ ОПЕРАЦИЙ (как в tch-rs / Burn)
// =============================================

/// Мини-тензор с dtype и основными операциями
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub dtype: DType,
}

impl Tensor {
    /// Создать тензор (сразу приводит данные к dtype)
    pub fn new(data: Vec<f32>, dtype: DType) -> Self {
        let casted = dtype.cast_vec(&data);
        Tensor {
            data: casted,
            dtype,
        }
    }

    pub fn from_slice(slice: &[f32], dtype: DType) -> Self {
        Tensor::new(slice.to_vec(), dtype)
    }

    pub fn zeros(len: usize, dtype: DType) -> Self {
        Tensor::new(vec![0.0; len], dtype)
    }

    pub fn ones(len: usize, dtype: DType) -> Self {
        Tensor::new(vec![1.0; len], dtype)
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    // ========================================
    // ЭЛЕМЕНТНЫЕ ОПЕРАЦИИ (Element-wise)
    // ========================================

    /// a + b
    pub fn add(&self, other: &Tensor) -> Result<Tensor, AppError> {
        if self.data.len() != other.data.len() {
            return Err(AppError::DimensionMismatch(format!(
                "add: {} != {}",
                self.data.len(),
                other.data.len()
            )));
        }
        let result: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Ok(Tensor::new(result, self.dtype))
    }

    /// a - b
    pub fn sub(&self, other: &Tensor) -> Result<Tensor, AppError> {
        if self.data.len() != other.data.len() {
            return Err(AppError::DimensionMismatch(format!(
                "sub: {} != {}",
                self.data.len(),
                other.data.len()
            )));
        }
        let result: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Ok(Tensor::new(result, self.dtype))
    }

    /// a * b (поэлементно)
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, AppError> {
        if self.data.len() != other.data.len() {
            return Err(AppError::DimensionMismatch(format!(
                "mul: {} != {}",
                self.data.len(),
                other.data.len()
            )));
        }
        let result: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        Ok(Tensor::new(result, self.dtype))
    }

    /// a / b
    pub fn div(&self, other: &Tensor) -> Result<Tensor, AppError> {
        if self.data.len() != other.data.len() {
            return Err(AppError::DimensionMismatch(format!(
                "div: {} != {}",
                self.data.len(),
                other.data.len()
            )));
        }
        let result: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if b == 0.0 { 0.0 } else { a / b })
            .collect();
        Ok(Tensor::new(result, self.dtype))
    }

    /// a * scalar
    pub fn scalar_mul(&self, scalar: f32) -> Tensor {
        let result: Vec<f32> = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::new(result, self.dtype)
    }

    /// a + scalar
    pub fn scalar_add(&self, scalar: f32) -> Tensor {
        let result: Vec<f32> = self.data.iter().map(|&x| x + scalar).collect();
        Tensor::new(result, self.dtype)
    }

    // ========================================
    // РЕДУКЦИИ (Reduction operations)
    // ========================================

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            0.0
        } else {
            self.sum() / self.data.len() as f32
        }
    }

    pub fn max(&self) -> f32 {
        *self
            .data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0)
    }

    pub fn min(&self) -> f32 {
        *self
            .data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0)
    }

    // ========================================
    // ПРИВЕДЕНИЕ ТИПА
    // ========================================

    pub fn to_dtype(&self, target: DType) -> Tensor {
        if self.dtype == target {
            self.clone()
        } else {
            Tensor::new(self.data.clone(), target)
        }
    }
}

// =============================================
// 9. МАТРИЧНЫЕ ОПЕРАЦИИ И DENSE-СЛОЙ (с DType + Tensor)
// =============================================

/// Матричное умножение: matrix (out × in) × vector (in) → vector (out)
pub fn mat_vec_mul(matrix: &[Vec<f32>], input: &[f32]) -> Result<Vec<f32>, AppError> {
    let out_size = matrix.len();
    if out_size == 0 {
        return Ok(vec![]);
    }
    let in_size = matrix[0].len();
    if input.len() != in_size {
        return Err(AppError::DimensionMismatch(format!(
            "mat_vec_mul: input {} != weights {}",
            input.len(),
            in_size
        )));
    }
    let mut result = vec![0.0; out_size];
    for i in 0..out_size {
        for j in 0..in_size {
            result[i] += matrix[i][j] * input[j];
        }
    }
    Ok(result)
}

/// Полносвязный (Dense) слой с поддержкой DType
#[derive(Clone, Debug)]
pub struct DenseLayer {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    pub dtype: DType,
}

impl DenseLayer {
    /// Создаёт слой с указанным dtype
    pub fn new(input_size: usize, output_size: usize, dtype: DType) -> Self {
        let mut weights = vec![vec![0.0; input_size]; output_size];
        for i in 0..output_size {
            for j in 0..input_size {
                weights[i][j] = 0.05 * ((i + j) as f32 + 1.0);
            }
        }
        // Приводим веса к dtype при инициализации
        let weights = weights
            .into_iter()
            .map(|row| dtype.cast_vec(&row))
            .collect();
        let bias = dtype.cast_vec(&vec![0.0; output_size]);
        DenseLayer {
            weights,
            bias,
            dtype,
        }
    }

    /// Forward с использованием Tensor-операций (add) и обработкой ошибок
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, AppError> {
        if input.is_empty() {
            return Err(AppError::EmptyTensor(
                "DenseLayer::forward: пустой вход".into(),
            ));
        }

        // 1. Вход → Tensor → cast
        let input_tensor = Tensor::from_slice(input, DType::Float32).to_dtype(self.dtype);
        let input_casted = input_tensor.to_vec();

        // 2. Матричное умножение
        let output = mat_vec_mul(&self.weights, &input_casted)?;

        // 3. Добавляем bias через Tensor::add
        let bias_tensor = Tensor::from_slice(&self.bias, self.dtype);
        let output_tensor = Tensor::from_slice(&output, self.dtype).add(&bias_tensor)?;

        let mut output = output_tensor.to_vec();

        // 4. Квантизация при необходимости
        if self.dtype.is_quantized() {
            output = self.dtype.cast_vec(&output);
        }
        Ok(output)
    }
}

// =============================================
// 9. РЕДУКЦИЯ В ФУНКЦИЯХ ПОТЕРЬ (как в PyTorch)
// =============================================

/// Режим редукции функции потерь (аналог torch.nn.functional)
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Reduction {
    /// Не агрегировать — вернуть сырую сумму
    None,
    /// Среднее по батчу / траектории (стандарт в обучении)
    Mean,
    /// Сумма
    Sum,
}

/// REINFORCE loss = −logπ(a|s) × advantage
/// с поддержкой reduction (Mean / Sum / None)
pub fn compute_reinforce_loss(
    log_probs: &[f32],
    advantages: &[f32],
    reduction: Reduction,
) -> f32 {
    assert_eq!(
        log_probs.len(),
        advantages.len(),
        "Размеры log_probs и advantages должны совпадать"
    );

    let mut total_loss = 0.0f32;
    for (&logp, &adv) in log_probs.iter().zip(advantages.iter()) {
        total_loss += -logp * adv; // классический REINFORCE
    }

    match reduction {
        Reduction::None => total_loss,
        Reduction::Mean => {
            if log_probs.is_empty() {
                0.0
            } else {
                total_loss / log_probs.len() as f32
            }
        }
        Reduction::Sum => total_loss,
    }
}

// =============================================
// 10. POLICY NETWORK (MLP + REINFORCE + DType)
// =============================================

// =============================================
// КОНФИГУРАЦИИ ОПТИМИЗАТОРОВ, LR-SCHEDULERS, ГИПЕРПАРАМЕТРОВ
// =============================================

/// Тип оптимизатора
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizerType {
    SGD,
    Adam,
}

impl OptimizerType {
    pub fn description(&self) -> &'static str {
        match self {
            OptimizerType::SGD => "SGD (классический градиентный спуск)",
            OptimizerType::Adam => "Adam (адаптивный моментный оптимизатор)",
        }
    }
}

/// Планировщик learning rate
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SchedulerType {
    Constant,
    StepDecay { step_size: usize, decay: f32 },
    ExponentialDecay { decay: f32 },
    CosineAnnealing { t_max: usize },
}

impl SchedulerType {
    pub fn get_lr(&self, episode: usize, base_lr: f32) -> f32 {
        match self {
            SchedulerType::Constant => base_lr,
            SchedulerType::StepDecay { step_size, decay } => {
                let steps = episode / step_size;
                base_lr * decay.powi(steps as i32)
            }
            SchedulerType::ExponentialDecay { decay } => {
                base_lr * decay.powi(episode as i32)
            }
            SchedulerType::CosineAnnealing { t_max } => {
                let cos = ((episode as f32 / *t_max as f32) * std::f32::consts::PI).cos();
                base_lr * 0.5 * (1.0 + cos)
            }
        }
    }

    pub fn description(&self) -> String {
        match self {
            SchedulerType::Constant => "Constant (постоянный LR)".to_string(),
            SchedulerType::StepDecay { step_size, decay } => {
                format!("StepDecay (step={}, decay={})", step_size, decay)
            }
            SchedulerType::ExponentialDecay { decay } => {
                format!("ExponentialDecay (decay={})", decay)
            }
            SchedulerType::CosineAnnealing { t_max } => {
                format!("CosineAnnealing (t_max={})", t_max)
            }
        }
    }
}

/// Единая конфигурация гиперпараметров
#[derive(Clone, Debug)]
pub struct Hyperparams {
    pub base_lr: f32,
    pub gamma: f32,
    pub optimizer: OptimizerType,
    pub scheduler: SchedulerType,
    pub episodes: usize,
    pub eval_episodes: usize,
}

impl Hyperparams {
    pub fn default() -> Self {
        Hyperparams {
            base_lr: 0.008,
            gamma: 0.99,
            optimizer: OptimizerType::SGD,
            scheduler: SchedulerType::StepDecay {
                step_size: 10,
                decay: 0.9,
            },
            episodes: 30,
            eval_episodes: 10,
        }
    }

    pub fn print(&self) {
        println!("🔧 Гиперпараметры:");
        println!("   • Optimizer     : {}", self.optimizer.description());
        println!("   • Scheduler     : {}", self.scheduler.description());
        println!("   • Base LR       : {:.5}", self.base_lr);
        println!("   • Gamma (RL)    : {:.3}", self.gamma);
        println!("   • Train episodes: {}", self.episodes);
        println!("   • Eval episodes : {}", self.eval_episodes);
    }
}

// =============================================
// РЕЖИМЫ РАБОТЫ МОДЕЛИ (Train / Eval)
// =============================================

/// Режим работы модели (аналог model.train() / model.eval() в PyTorch)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelMode {
    Train,
    Eval,
}

impl ModelMode {
    pub fn description(&self) -> &'static str {
        match self {
            ModelMode::Train => "TRAIN (обучение + обновление весов)",
            ModelMode::Eval => "EVAL  (только инференс, обновления ЗАПРЕЩЕНЫ)",
        }
    }
}

// =============================================
// BACKEND / УСТРОЙСТВО
// =============================================

/// Устройство выполнения (как torch.device / tch::Device)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Device {
    Cpu,
    Gpu,
    Metal,
}

impl Device {
    pub fn description(&self) -> &'static str {
        match self {
            Device::Cpu => "CPU (стандартное выполнение)",
            Device::Gpu => "GPU (симуляция CUDA)",
            Device::Metal => "Metal (Apple Silicon / MPS)",
        }
    }

    pub fn is_accelerated(&self) -> bool {
        matches!(self, Device::Gpu | Device::Metal)
    }
}

// =============================================
// КОНФИГУРАЦИИ МОДЕЛЕЙ / АРХИТЕКТУР (закрытый набор)
// =============================================

/// Закрытый набор предопределённых архитектур
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelArchitecture {
    SimpleMLP,
    DeepMLP,
    ActorCritic,
    HybridTreeNN,
    Tiny,
}

impl ModelArchitecture {
    pub fn description(&self) -> &'static str {
        match self {
            ModelArchitecture::SimpleMLP => "SimpleMLP (1 → 8 → policy/value)",
            ModelArchitecture::DeepMLP => "DeepMLP (1 → 16 → 8 → policy/value)",
            ModelArchitecture::ActorCritic => "ActorCritic (shared backbone + heads)",
            ModelArchitecture::HybridTreeNN => "HybridTreeNN (NN + DecisionTree)",
            ModelArchitecture::Tiny => "Tiny (1 → 4 → policy/value)",
        }
    }

    pub fn hidden_sizes(&self) -> Vec<usize> {
        match self {
            ModelArchitecture::SimpleMLP => vec![8],
            ModelArchitecture::DeepMLP => vec![16, 8],
            ModelArchitecture::ActorCritic => vec![16, 8],
            ModelArchitecture::HybridTreeNN => vec![12, 8],
            ModelArchitecture::Tiny => vec![4],
        }
    }
}

/// Единая конфигурация модели
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub dtype: DType,
    pub device: Device,
    pub hyperparams: Hyperparams,
}

impl ModelConfig {
    pub fn default() -> Self {
        ModelConfig {
            architecture: ModelArchitecture::SimpleMLP,
            dtype: DType::Float16,
            device: Device::Cpu,
            hyperparams: Hyperparams::default(),
        }
    }

    pub fn print(&self) {
        println!("📋 Конфигурация модели:");
        println!("   • Architecture : {}", self.architecture.description());
        println!("   • Device       : {}", self.device.description());
        println!("   • DType        : {:?}", self.dtype);
        self.hyperparams.print();
    }
}

// =============================================
// АУГМЕНТАЦИИ ДАННЫХ (CV + NLP)
// =============================================

/// Типы аугментаций (адаптированы под 1D-состояние lives)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AugmentationType {
    // Computer Vision style
    GaussianNoise { std: f32 },
    RandomScale { min: f32, max: f32 },
    RandomShift { max_shift: f32 },
    // NLP style
    RandomMask { mask_prob: f32 },
    RandomReplace { replace_prob: f32, noise_std: f32 },
}

/// Конфигурируемый аугментатор
#[derive(Clone, Debug)]
pub struct DataAugmenter {
    pub augmentations: Vec<AugmentationType>,
    /// Вероятность применения каждой аугментации (0.0–1.0)
    pub probability: f32,
}

impl DataAugmenter {
    pub fn default() -> Self {
        DataAugmenter {
            augmentations: vec![
                AugmentationType::GaussianNoise { std: 0.05 },
                AugmentationType::RandomScale { min: 0.9, max: 1.1 },
                AugmentationType::RandomMask { mask_prob: 0.1 },
            ],
            probability: 0.7,
        }
    }

    /// Применить аугментации к состоянию
    pub fn augment(&self, state: &[f32]) -> Result<Vec<f32>, AppError> {
        if state.is_empty() {
            return Err(AppError::AugmentationError(
                "Пустое состояние для аугментации".into(),
            ));
        }

        let mut augmented = state.to_vec();
        let mut rng = rand::thread_rng();

        for aug in &self.augmentations {
            if rng.gen::<f32>() > self.probability {
                continue;
            }

            match aug {
                AugmentationType::GaussianNoise { std } => {
                    for x in &mut augmented {
                        let noise: f32 = rng.gen::<f32>() * 2.0 - 1.0;
                        *x += noise * std;
                    }
                }
                AugmentationType::RandomScale { min, max } => {
                    let scale = min + (max - min) * rng.gen::<f32>();
                    for x in &mut augmented {
                        *x *= scale;
                    }
                }
                AugmentationType::RandomShift { max_shift } => {
                    let shift = (rng.gen::<f32>() * 2.0 - 1.0) * max_shift;
                    for x in &mut augmented {
                        *x += shift;
                    }
                }
                AugmentationType::RandomMask { mask_prob } => {
                    for x in &mut augmented {
                        if rng.gen::<f32>() < *mask_prob {
                            *x = 0.0;
                        }
                    }
                }
                AugmentationType::RandomReplace {
                    replace_prob,
                    noise_std,
                } => {
                    for x in &mut augmented {
                        if rng.gen::<f32>() < *replace_prob {
                            let noise: f32 = (rng.gen::<f32>() * 2.0 - 1.0) * noise_std;
                            *x += noise;
                        }
                    }
                }
            }
        }

        // Клиппинг в [0, 1] для lives
        for x in &mut augmented {
            *x = x.clamp(0.0, 1.0);
        }

        Ok(augmented)
    }
}

// =============================================
// MULTI-TASK ВЫХОДЫ МОДЕЛИ
// =============================================

/// Результат multi-task предсказания
#[derive(Clone, Debug)]
pub struct MultiTaskOutput {
    /// Policy: [p_feed, p_play] после Softmax
    pub policy: Vec<f32>,
    /// Value: скалярная оценка состояния
    pub value: f32,
    /// Auxiliary: предсказание DecisionTree (если есть)
    pub tree_value: Option<f32>,
}

impl MultiTaskOutput {
    pub fn new(policy: Vec<f32>, value: f32, tree_value: Option<f32>) -> Self {
        MultiTaskOutput {
            policy,
            value,
            tree_value,
        }
    }
}

/// Multi-task MLP: shared backbone + policy head + value head + optional tree
/// Архитектура:
///   state(1) → Dense(1→8) → LayerNorm → ReLU
///              ├─ Dense(8→2) → Softmax  → policy
///              └─ Dense(8→1)            → value
#[derive(Clone, Debug)]
pub struct CatPolicyNetwork {
    pub layer1: DenseLayer,              // first hidden
    pub layer_hidden: Vec<DenseLayer>,   // additional hidden (depends on architecture)
    pub layer2_policy: DenseLayer,       // policy head → 2
    pub layer_value: DenseLayer,         // value head → 1
    pub dtype: DType,
    pub mode: ModelMode,
    pub device: Device,
    pub architecture: ModelArchitecture,
    pub hyperparams: Hyperparams,
    pub decision_tree: Option<DecisionTree>,
    pub augmenter: Option<DataAugmenter>,
}

impl CatPolicyNetwork {
    pub fn new() -> Self {
        Self::from_config(ModelConfig::default())
    }

    pub fn new_with_dtype(dtype: DType) -> Self {
        let mut cfg = ModelConfig::default();
        cfg.dtype = dtype;
        Self::from_config(cfg)
    }

    pub fn new_with_config(dtype: DType, hyperparams: Hyperparams) -> Self {
        let mut cfg = ModelConfig::default();
        cfg.dtype = dtype;
        cfg.hyperparams = hyperparams;
        Self::from_config(cfg)
    }

    /// Создать сеть из полной ModelConfig (архитектура + device + dtype + hyperparams)
    pub fn from_config(config: ModelConfig) -> Self {
        config.print();

        let hidden = config.architecture.hidden_sizes();
        let first_size = *hidden.first().unwrap_or(&8);
        let last_size = *hidden.last().unwrap_or(&8);

        // layer1: 1 → first hidden
        let layer1 = DenseLayer::new(1, first_size, config.dtype);

        // additional hidden layers
        let mut layer_hidden = Vec::new();
        for w in hidden.windows(2) {
            layer_hidden.push(DenseLayer::new(w[0], w[1], config.dtype));
        }

        // heads from last hidden size
        let layer2_policy = DenseLayer::new(last_size, 2, config.dtype);
        let layer_value = DenseLayer::new(last_size, 1, config.dtype);

        // DecisionTree for HybridTreeNN (and default for others as auxiliary)
        let use_tree = matches!(
            config.architecture,
            ModelArchitecture::HybridTreeNN | ModelArchitecture::ActorCritic | ModelArchitecture::SimpleMLP | ModelArchitecture::DeepMLP | ModelArchitecture::Tiny
        );
        let decision_tree = if use_tree {
            let mut tree = DecisionTree::new(4, 2, SplitQuality::Gini);
            let states = vec![0.1, 0.3, 0.5, 0.7, 0.9];
            let targets = vec![0.9, 0.7, 0.5, 0.3, 0.1];
            let _ = tree.fit(&states, &targets);
            Some(tree)
        } else {
            None
        };

        println!("🧬 Архитектура: {}", config.architecture.description());
        println!("   → hidden sizes: {:?}", hidden);

        CatPolicyNetwork {
            layer1,
            layer_hidden,
            layer2_policy,
            layer_value,
            dtype: config.dtype,
            mode: ModelMode::Train,
            device: config.device,
            architecture: config.architecture,
            hyperparams: config.hyperparams,
            decision_tree,
            augmenter: Some(DataAugmenter::default()),
        }
    }

    pub fn to_device(&mut self, device: Device) {
        self.device = device;
        println!("🔄 Модель перемещена на: {}", device.description());
    }

    pub fn architecture_description(&self) -> &'static str {
        self.architecture.description()
    }

    pub fn train(&mut self) {
        self.mode = ModelMode::Train;
        println!("🔥 Модель переведена в режим TRAIN");
    }

    pub fn eval(&mut self) {
        self.mode = ModelMode::Eval;
        println!("📊 Модель переведена в режим EVAL");
    }

    pub fn is_training(&self) -> bool {
        self.mode == ModelMode::Train
    }

    /// Обычный forward (только policy) — для совместимости
    pub fn forward(&self, state: &[f32]) -> Result<Vec<f32>, AppError> {
        let multi = self.forward_multi_task(state)?;
        Ok(multi.policy)
    }

    /// MULTI-TASK forward: policy + value + tree
    /// Аугментации применяются только в режиме Train
    pub fn forward_multi_task(&self, state: &[f32]) -> Result<MultiTaskOutput, AppError> {
        if state.is_empty() {
            return Err(AppError::EmptyTensor(
                "forward_multi_task: пустое состояние".into(),
            ));
        }

        // Проверка observation space (Box [0, 1])
        let obs_space = Space::Box {
            low: vec![0.0],
            high: vec![1.0],
        };
        if !obs_space.contains(state) {
            return Err(AppError::SpaceError(format!(
                "Состояние {:?} не входит в observation_space {}",
                state,
                obs_space.description()
            )));
        }

        // Логирование устройства (симуляция)
        if self.device.is_accelerated() {
            // В реальном tch-rs / burn здесь был бы tensor.to(device)
            let _ = self.device; // silence in hot path unless needed
        }

        // Аугментации только в Train
        let input_state = if self.is_training() {
            if let Some(ref aug) = self.augmenter {
                aug.augment(state)?
            } else {
                state.to_vec()
            }
        } else {
            state.to_vec()
        };

        // Shared backbone: layer1 + additional hidden layers
        let mut x = self.layer1.forward(&input_state)?;
        x = layer_norm_default(&x);
        x = apply_activation(x, relu);

        for layer in &self.layer_hidden {
            x = layer.forward(&x)?;
            x = layer_norm_default(&x);
            x = apply_activation(x, relu);
        }

        // Policy head
        let logits = self.layer2_policy.forward(&x)?;
        let policy = softmax(&logits);

        // Value head
        let value_raw = self.layer_value.forward(&x)?;
        let value = value_raw[0];

        // Auxiliary DecisionTree
        let tree_value = self
            .decision_tree
            .as_ref()
            .map(|t| t.predict(input_state[0]));

        Ok(MultiTaskOutput::new(policy, value, tree_value))
    }

    /// Выбрать действие по argmax policy
    pub fn select_action(&self, state: &[f32]) -> Result<Action, AppError> {
        let multi = self.forward_multi_task(state)?;
        Ok(if multi.policy[0] > multi.policy[1] {
            Action::Feed
        } else {
            Action::Play
        })
    }

    /// Один шаг Policy Gradient (упрощённый REINFORCE)
    /// В режиме Eval ничего не делает (веса не обновляются)
    pub fn reinforce_step(
        &mut self,
        state: &[f32],
        action: usize,
        advantage: f32,
        learning_rate: f32,
    ) -> Result<(), AppError> {
        if !self.is_training() {
            // В режиме EVAL обновления ЗАПРЕЩЕНЫ
            return Ok(());
        }

        // Forward с сохранением активаций
        let h_pre = self.layer1.forward(state)?;
        let h_norm = layer_norm_default(&h_pre);
        let h_act = apply_activation(h_norm.clone(), relu);
        let logits = self.layer2_policy.forward(&h_act)?;
        let probs = softmax(&logits);

        // Градиент d_logits = (probs - one_hot) * advantage
        // с использованием Tensor::scalar_mul
        let mut d_logits = probs.clone();
        d_logits[action] -= 1.0;
        let d_logits_tensor =
            Tensor::from_slice(&d_logits, DType::Float32).scalar_mul(advantage);
        let d_logits = d_logits_tensor.to_vec();

        // Обновление policy head
        for i in 0..2 {
            for j in 0..8 {
                self.layer2_policy.weights[i][j] -= learning_rate * d_logits[i] * h_act[j];
            }
            self.layer2_policy.bias[i] -= learning_rate * d_logits[i];
        }

        // Backprop через ReLU
        let mut d_h_act = vec![0.0; 8];
        for j in 0..8 {
            for i in 0..2 {
                d_h_act[j] += d_logits[i] * self.layer2_policy.weights[i][j];
            }
            if h_act[j] <= 0.0 {
                d_h_act[j] = 0.0;
            }
        }

        // Обновление layer1
        for i in 0..8 {
            for j in 0..1 {
                self.layer1.weights[i][j] -= learning_rate * d_h_act[i] * state[j];
            }
            self.layer1.bias[i] -= learning_rate * d_h_act[i];
        }
        Ok(())
    }

    /// Собрать траекторию и посчитать loss с разными reduction
    pub fn collect_and_compute_loss(
        &self,
        env: &mut CatEnv,
        max_steps: usize,
        gamma: f32,
    ) -> Result<(f32, f32, f32, f32, usize), AppError> {
        // (episode_reward, loss_mean, loss_sum, loss_none, steps)
        env.reset();
        let mut trajectory: Vec<(Vec<f32>, usize, f32, f32)> = Vec::new();
        let mut episode_reward = 0.0;
        let mut steps = 0;

        while !env.is_done() && steps < max_steps {
            let state = env.observation();
            let probs = self.forward(&state)?;
            let action = if probs[0] > probs[1] { 0usize } else { 1usize };
            let log_prob = probs[action].ln();

            let result = env.step(Action::from_u8(action as u8).unwrap());
            episode_reward += result.reward;
            trajectory.push((state, action, result.reward, log_prob));
            steps += 1;
            if result.done {
                break;
            }
        }

        // Дисконтированные returns G_t
        let mut returns = vec![0.0f32; trajectory.len()];
        let mut g = 0.0;
        for i in (0..trajectory.len()).rev() {
            g = trajectory[i].2 + gamma * g;
            returns[i] = g;
        }

        let log_probs: Vec<f32> = trajectory.iter().map(|(_, _, _, lp)| *lp).collect();
        let advantages = returns;

        let loss_mean = compute_reinforce_loss(&log_probs, &advantages, Reduction::Mean);
        let loss_sum = compute_reinforce_loss(&log_probs, &advantages, Reduction::Sum);
        let loss_none = compute_reinforce_loss(&log_probs, &advantages, Reduction::None);

        Ok((episode_reward, loss_mean, loss_sum, loss_none, steps))
    }

    /// Запустить эпизод + обновить веса (полный REINFORCE)
    pub fn train_episode(
        &mut self,
        env: &mut CatEnv,
        max_steps: usize,
        gamma: f32,
        lr: f32,
    ) -> Result<(f32, f32, usize), AppError> {
        env.reset();
        let mut trajectory: Vec<(Vec<f32>, usize, f32, f32)> = Vec::new();
        let mut episode_reward = 0.0;
        let mut steps = 0;

        while !env.is_done() && steps < max_steps {
            let state = env.observation();
            let probs = self.forward(&state)?;
            let action = if probs[0] > probs[1] { 0usize } else { 1usize };
            let log_prob = probs[action].ln();

            let result = env.step(Action::from_u8(action as u8).unwrap());
            episode_reward += result.reward;
            trajectory.push((state, action, result.reward, log_prob));
            steps += 1;
            if result.done {
                break;
            }
        }

        // Returns
        let mut returns = vec![0.0f32; trajectory.len()];
        let mut g = 0.0;
        for i in (0..trajectory.len()).rev() {
            g = trajectory[i].2 + gamma * g;
            returns[i] = g;
        }

        // Loss (для мониторинга)
        let log_probs: Vec<f32> = trajectory.iter().map(|(_, _, _, lp)| *lp).collect();
        let loss = compute_reinforce_loss(&log_probs, &returns, Reduction::Mean);

        // Обновление весов
        for (t, (state, action, _, _)) in trajectory.iter().enumerate() {
            self.reinforce_step(state, *action, returns[t], lr)?;
        }

        Ok((episode_reward, loss, steps))
    }

    /// Запустить эпизод без обучения
    pub fn run_episode(&self, env: &mut CatEnv, max_steps: usize) -> Result<(f32, usize), AppError> {
        env.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;

        while !env.is_done() && steps < max_steps {
            let state = env.observation();
            let action = self.select_action(&state)?;
            let result = env.step(action);
            total_reward += result.reward;
            steps += 1;
            if result.done {
                break;
            }
        }
        Ok((total_reward, steps))
    }
}

impl CatPolicyNetwork {
    /// Сохранить модель в текстовый файл
    pub fn save(&self, path: &str) -> Result<(), AppError> {
        let mut f = File::create(path).map_err(AppError::Io)?;

        writeln!(f, "# CatPolicyNetwork model file").map_err(AppError::Io)?;
        writeln!(f, "architecture: {:?}", self.architecture).map_err(AppError::Io)?;
        writeln!(f, "device: {:?}", self.device).map_err(AppError::Io)?;
        writeln!(f, "dtype: {:?}", self.dtype).map_err(AppError::Io)?;
        writeln!(f, "mode: {:?}", self.mode).map_err(AppError::Io)?;
        writeln!(f, "base_lr: {}", self.hyperparams.base_lr).map_err(AppError::Io)?;
        writeln!(f, "gamma: {}", self.hyperparams.gamma).map_err(AppError::Io)?;

        self.write_dense_layer(&mut f, "layer1", &self.layer1)?;
        for (i, layer) in self.layer_hidden.iter().enumerate() {
            self.write_dense_layer(&mut f, &format!("layer_hidden_{}", i), layer)?;
        }
        self.write_dense_layer(&mut f, "layer2_policy", &self.layer2_policy)?;
        self.write_dense_layer(&mut f, "layer_value", &self.layer_value)?;

        writeln!(f, "# End of model").map_err(AppError::Io)?;
        println!("✅ Модель сохранена в {}", path);
        Ok(())
    }

    fn write_dense_layer(
        &self,
        f: &mut File,
        name: &str,
        layer: &DenseLayer,
    ) -> Result<(), AppError> {
        writeln!(f, "[[{}]]", name).map_err(AppError::Io)?;
        writeln!(
            f,
            "shape: {} {}",
            layer.weights.len(),
            layer.weights.first().map(|r| r.len()).unwrap_or(0)
        )
        .map_err(AppError::Io)?;
        for row in &layer.weights {
            for &val in row {
                write!(f, "{} ", val).map_err(AppError::Io)?;
            }
            writeln!(f).map_err(AppError::Io)?;
        }
        for &b in &layer.bias {
            writeln!(f, "{}", b).map_err(AppError::Io)?;
        }
        Ok(())
    }

    /// Загрузить веса из файла (метаданные + веса)
    pub fn load(&mut self, path: &str) -> Result<(), AppError> {
        let file = File::open(path).map_err(AppError::Io)?;
        let lines: Vec<String> = io::BufReader::new(file)
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .map_err(AppError::Io)?;

        if lines.is_empty() {
            return Err(AppError::SerializationError(
                "Файл модели пустой".into(),
            ));
        }

        // Парсинг весов слоёв
        let mut idx = 0;
        while idx < lines.len() {
            let line = lines[idx].trim();
            if line.starts_with("[[") && line.ends_with("]]") {
                let name = &line[2..line.len() - 2];
                idx += 1;
                if idx >= lines.len() {
                    break;
                }
                // shape line
                let shape_parts: Vec<&str> = lines[idx].split_whitespace().collect();
                if shape_parts.len() < 3 {
                    idx += 1;
                    continue;
                }
                let out_size: usize = shape_parts[1].parse().unwrap_or(0);
                let in_size: usize = shape_parts[2].parse().unwrap_or(0);
                idx += 1;

                let mut weights = vec![vec![0.0; in_size]; out_size];
                for i in 0..out_size {
                    if idx >= lines.len() {
                        break;
                    }
                    let row: Vec<f32> = lines[idx]
                        .split_whitespace()
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    if row.len() == in_size {
                        weights[i] = row;
                    }
                    idx += 1;
                }
                let mut bias = vec![0.0; out_size];
                for i in 0..out_size {
                    if idx >= lines.len() {
                        break;
                    }
                    if let Ok(b) = lines[idx].trim().parse::<f32>() {
                        bias[i] = b;
                    }
                    idx += 1;
                }

                // Применяем к соответствующему слою
                match name {
                    "layer1" => {
                        self.layer1.weights = weights;
                        self.layer1.bias = bias;
                    }
                    "layer2_policy" => {
                        self.layer2_policy.weights = weights;
                        self.layer2_policy.bias = bias;
                    }
                    "layer_value" => {
                        self.layer_value.weights = weights;
                        self.layer_value.bias = bias;
                    }
                    n if n.starts_with("layer_hidden_") => {
                        if let Ok(hi) = n.trim_start_matches("layer_hidden_").parse::<usize>() {
                            if hi < self.layer_hidden.len() {
                                self.layer_hidden[hi].weights = weights;
                                self.layer_hidden[hi].bias = bias;
                            }
                        }
                    }
                    _ => {}
                }
            } else {
                idx += 1;
            }
        }

        println!("📥 Модель загружена из {}", path);
        Ok(())
    }
}

impl Default for CatPolicyNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================
// РЕКУРСИВНЫЕ СТРУКТУРЫ: ДЕРЕВЬЯ РЕШЕНИЙ
// =============================================

/// Критерий качества сплита (как в linfa-trees)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SplitQuality {
    Gini,
    Entropy,
    MSE,
}

/// Рекурсивный узел дерева решений
#[derive(Clone, Debug)]
pub enum TreeNode {
    Leaf {
        value: f32,
        samples: usize,
    },
    Split {
        feature_idx: usize,
        threshold: f32,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
        quality: f32,
        split_quality_type: SplitQuality,
    },
}

/// Decision Tree
#[derive(Clone, Debug)]
pub struct DecisionTree {
    pub root: Option<TreeNode>,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub quality: SplitQuality,
}

impl DecisionTree {
    pub fn new(max_depth: usize, min_samples_split: usize, quality: SplitQuality) -> Self {
        DecisionTree {
            root: None,
            max_depth,
            min_samples_split,
            quality,
        }
    }

    pub fn fit(&mut self, states: &[f32], targets: &[f32]) -> Result<(), AppError> {
        if states.len() != targets.len() {
            return Err(AppError::DimensionMismatch(
                "states.len() != targets.len()".into(),
            ));
        }
        if states.is_empty() {
            return Err(AppError::EmptyTensor(
                "Нет данных для обучения дерева".into(),
            ));
        }
        self.root = Some(self.build_tree(states, targets, 0));
        Ok(())
    }

    fn build_tree(&self, states: &[f32], targets: &[f32], depth: usize) -> TreeNode {
        if states.len() < self.min_samples_split || depth >= self.max_depth {
            let value = if targets.is_empty() {
                0.5
            } else {
                targets.iter().sum::<f32>() / targets.len() as f32
            };
            return TreeNode::Leaf {
                value,
                samples: states.len(),
            };
        }

        let mut sorted: Vec<(f32, f32)> = states
            .iter()
            .zip(targets.iter())
            .map(|(&s, &t)| (s, t))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let median_idx = sorted.len() / 2;
        let threshold = sorted[median_idx].0;

        let mut left_states = vec![];
        let mut left_targets = vec![];
        let mut right_states = vec![];
        let mut right_targets = vec![];

        for (&s, &t) in states.iter().zip(targets.iter()) {
            if s < threshold {
                left_states.push(s);
                left_targets.push(t);
            } else {
                right_states.push(s);
                right_targets.push(t);
            }
        }

        if left_states.is_empty() || right_states.is_empty() {
            let value = targets.iter().sum::<f32>() / targets.len() as f32;
            return TreeNode::Leaf {
                value,
                samples: states.len(),
            };
        }

        let left = Box::new(self.build_tree(&left_states, &left_targets, depth + 1));
        let right = Box::new(self.build_tree(&right_states, &right_targets, depth + 1));

        TreeNode::Split {
            feature_idx: 0,
            threshold,
            left,
            right,
            quality: 0.85,
            split_quality_type: self.quality,
        }
    }

    pub fn predict(&self, state: f32) -> f32 {
        let mut node = self.root.as_ref();
        while let Some(TreeNode::Split {
            threshold,
            left,
            right,
            ..
        }) = node
        {
            if state < *threshold {
                node = Some(left.as_ref());
            } else {
                node = Some(right.as_ref());
            }
        }
        match node {
            Some(TreeNode::Leaf { value, .. }) => *value,
            _ => 0.5,
        }
    }
}

// =============================================
// COMPUTATION GRAPH (рекурсивный граф)
// =============================================

#[derive(Clone, Debug)]
pub enum ComputationNode {
    Input { value: f32 },
    Linear {
        weight: f32,
        bias: f32,
        input: Box<ComputationNode>,
    },
    Activation {
        func: fn(f32) -> f32,
        input: Box<ComputationNode>,
    },
    Softmax { input: Box<ComputationNode> },
}

impl ComputationNode {
    pub fn forward(&self) -> f32 {
        match self {
            ComputationNode::Input { value } => *value,
            ComputationNode::Linear {
                weight,
                bias,
                input,
            } => {
                let x = input.forward();
                weight * x + bias
            }
            ComputationNode::Activation { func, input } => {
                let x = input.forward();
                func(x)
            }
            ComputationNode::Softmax { input } => {
                let x = input.forward();
                1.0 / (1.0 + (-x).exp())
            }
        }
    }
}

// =============================================
// FFI — C-совместимый слой
// =============================================

#[repr(C)]
pub struct FfiCat {
    name: *mut c_char,
    lives: u8,
}

#[no_mangle]
pub extern "C" fn cat_new(name: *const c_char, lives: u8) -> *mut FfiCat {
    if name.is_null() {
        return ptr::null_mut();
    }
    let name_str = unsafe {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s.to_owned(),
            Err(_) => return ptr::null_mut(),
        }
    };
    if name_str.is_empty() || lives == 0 || lives > 9 {
        return ptr::null_mut();
    }
    let c_name = match CString::new(name_str) {
        Ok(s) => s.into_raw(),
        Err(_) => return ptr::null_mut(),
    };
    let cat = FfiCat {
        name: c_name,
        lives,
    };
    Box::into_raw(Box::new(cat))
}

#[no_mangle]
pub extern "C" fn cat_meow(cat: *mut FfiCat) {
    if cat.is_null() {
        return;
    }
    let cat = unsafe { &*cat };
    if cat.name.is_null() {
        return;
    }
    let name = unsafe { CStr::from_ptr(cat.name).to_string_lossy() };
    println!("{} says meow! (lives: {})", name, cat.lives);
}

#[no_mangle]
pub extern "C" fn cat_get_lives(cat: *const FfiCat) -> u8 {
    if cat.is_null() {
        return 0;
    }
    unsafe { (*cat).lives }
}

#[no_mangle]
pub extern "C" fn cat_feed(cat: *mut FfiCat) -> f32 {
    if cat.is_null() {
        return -10.0;
    }
    let cat = unsafe { &mut *cat };
    if cat.lives < 9 {
        cat.lives += 1;
        1.0
    } else {
        -1.0
    }
}

#[no_mangle]
pub extern "C" fn cat_play(cat: *mut FfiCat) -> f32 {
    if cat.is_null() {
        return -10.0;
    }
    let cat = unsafe { &mut *cat };
    if cat.lives > 1 {
        cat.lives -= 1;
        0.5
    } else {
        -2.0
    }
}

#[no_mangle]
pub extern "C" fn cat_free(cat: *mut FfiCat) {
    if cat.is_null() {
        return;
    }
    unsafe {
        let cat = Box::from_raw(cat);
        if !cat.name.is_null() {
            let _ = CString::from_raw(cat.name);
        }
    }
}
