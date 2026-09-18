# Cat RL Solid — полный пайплайн

Учебный проект на Rust с полным ML/RL-стеком вокруг кота.

## Возможности

| Компонент | Описание |
|-----------|----------|
| **Preprocessing** | stdin → валидация → Tensor |
| **Postprocessing** | вероятности → текст + meow() |
| **Synthetic Data** | Cat::random + CSV |
| **RL Environment** | Gym-like CatEnv |
| **Activation Functions** | relu, sigmoid, tanh, softmax |
| **Layer Normalization** | layer_norm_default |
| **DenseLayer** | полносвязный слой |
| **CatPolicyNetwork** | MLP 1→8→2 |
| **Reduction / Loss** | REINFORCE loss с Mean / Sum / None |
| **DType** | Float32 / Float16 / Int8 (как tch::Kind) |
| **Tensor ops** | add, sub, mul, div, scalar_mul, sum, mean, max, to_dtype |
| **ModelMode** | Train / Eval (как model.train() / model.eval()) |
| **Hyperparams** | Optimizer (SGD/Adam) + LR-scheduler + gamma/episodes |
| **AppError** | Result-based ошибки (DimensionMismatch, EmptyTensor, …) |
| **DecisionTree** | Рекурсивное дерево решений (TreeNode + Box) |
| **ComputationGraph** | Рекурсивный граф вычислений (autograd-style) |
| **Multi-task** | policy + value + tree heads |
| **Augmentations** | CV (noise/scale/shift) + NLP (mask/replace), Train-only |
| **RL Spaces** | Discrete / Box (sample, contains) — Gym-like |
| **Device** | Cpu / Gpu / Metal (to_device) |
| **Architectures** | SimpleMLP / DeepMLP / ActorCritic / HybridTreeNN / Tiny |
| **Serialization** | save / load текстового .model файла |
| **FFI** | cdylib для Python |

## Reduction в функциях потерь

```rust
pub enum Reduction {
    None,  // сырая сумма
    Mean,  // среднее (как в PyTorch)
    Sum,   // сумма
}

let loss = compute_reinforce_loss(&log_probs, &advantages, Reduction::Mean);
```

Это точный аналог `reduction` в `torch.nn.functional`.

## DType (типы данных тензоров)

```rust
pub enum DType {
    Float32, // 4 байта, полная точность
    Float16, // 2 байта, Tensor Cores
    Int8,    // 1 байт, квантизация
}

let nn = CatPolicyNetwork::new_with_dtype(DType::Float16);
```

## REINFORCE

```rust
let mut nn = CatPolicyNetwork::new_with_dtype(DType::Float16);
let mut env = CatEnv::random();

// Собрать траекторию и посчитать loss всеми reduction
let (reward, loss_mean, loss_sum, loss_none, steps) =
    nn.collect_and_compute_loss(&mut env, 20, 0.99);

// Полный шаг обучения
let (reward, loss, steps) = nn.train_episode(&mut env, 25, 0.99, 0.01);
```

## Быстрый старт

```bash
cd cat_rl_solid
cargo run
```

## Архитектура сети

```
state (1)
  → Dense(1→8)
  → LayerNorm
  → ReLU
  → Dense(8→2)
  → Softmax → [p_feed, p_play]
```
