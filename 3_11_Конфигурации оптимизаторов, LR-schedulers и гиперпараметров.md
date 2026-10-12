# **Конфигурации оптимизаторов, LR-schedulers и гиперпараметров**
Оптимизатор, scheduler и гиперпараметры собраны в Hyperparams: base_lr, gamma, OptimizerType, SchedulerType, число эпизодов. Scheduler реально меняет LR через get_lr(episode, base_lr) (Constant / StepDecay / ExponentialDecay / CosineAnnealing). Оптимизатор в конфиге есть как enum, а обновление весов в текущей версии — простой SGD-шаг с этим динамическим lr.

В проекте это три связанных уровня:

OptimizerType — какой оптимизатор «заявлен»

SchedulerType — как меняется learning rate по эпизодам

Hyperparams — единый контейнер всех настроек обучения

Они хранятся в сети и в ModelConfig, печатаются при создании и участвуют в цикле обучения.

1. OptimizerType


```
pub enum OptimizerType {
    SGD,
    Adam,
}

```
SGD классический градиентный спуск

Adam адаптивный моментный оптимизатор

description() только для логов.

фактическое обновление весов в reinforce_step — это SGD-шаг:

 ← w − lr · grad

Моменты Adam (m, v) не хранятся. OptimizerType::Adam — метка конфигурации / задел под расширение; поведение шага сейчас одинаковое.

2. SchedulerType — планировщик LR


```
pub enum SchedulerType {
    Constant,
    StepDecay { step_size: usize, decay: f32 },
    ExponentialDecay { decay: f32 },
    CosineAnnealing { t_max: usize },
}
```


Главный метод:
```
pub fn get_lr(&self, episode: usize, base_lr: f32) -> f32

```


description() возвращает читаемую строку для print().

3. Hyperparams — всё в одном месте


```
pub struct Hyperparams {
    pub base_lr: f32,              // начальный learning rate
    pub gamma: f32,                // дисконт RL (returns)
    pub optimizer: OptimizerType,
    pub scheduler: SchedulerType,
    pub episodes: usize,           // число train-эпизодов
    pub eval_episodes: usize,      // число eval-эпизодов
}
```

Значения по умолчанию

base_lr:       0.008

gamma:         0.99

optimizer:     SGD

scheduler:     StepDecay { step_size: 10, decay: 0.9 }

episodes:      30

eval_episodes: 10

print()

Выводит блок:

🔧 Гиперпараметры:

   • Optimizer     : SGD (...)

   • Scheduler     : StepDecay (step=10, decay=0.9)

   • Base LR       : 0.00800

   • Gamma (RL)    : 0.990

   • Train episodes: 30

   • Eval episodes : 10

Вызывается при создании сети / ModelConfig::print().

4. Где это живёт в архитектуре

ModelConfig

├── architecture

├── dtype

├── device

└── hyperparams: Hyperparams

         ├── base_lr, gamma

         ├── optimizer

         └── scheduler

CatPolicyNetwork
└── hyperparams: Hyperparams  (копия из конфига)

Создание:

let mut hp = Hyperparams::default();
hp.optimizer = OptimizerType::Adam;
hp.scheduler = SchedulerType::CosineAnnealing { t_max: 30 };
hp.base_lr = 0.01;

let nn = CatPolicyNetwork::new_with_config(DType::Float16, hp);
// или через ModelConfig { hyperparams: hp, ... }

В save() пишутся как минимум base_lr (и связанные метаданные); scheduler/optimizer тоже могут логироваться в файл модели.

5. Как это используется в обучении

Типичный цикл (идея из design / train_episode):
```
for ep in 0..nn.hyperparams.episodes {
    let lr = nn.hyperparams.scheduler.get_lr(ep, nn.hyperparams.base_lr);

    // эпизод в среде, сбор trajectory, returns с gamma
    // ...
    nn.reinforce_step(state, action, advantage, lr)?;  // динамический LR
}

```

Связь с loss: compute_reinforce_loss считает скаляр для лога; величина шага задаётся именно lr от scheduler, а не reduction.

6. Схема потока конфигурации

Hyperparams::default()  или  ручная настройка

        │

        ▼

ModelConfig / new_with_config

        │

        ▼

CatPolicyNetwork.hyperparams

        │

        ├─ print() при старте

        ├─ gamma → подсчёт returns

        ├─ scheduler.get_lr(episode, base_lr) → lr

        └─ reinforce_step(..., lr) → w -= lr * grad



7. Где смотреть в коде

OptimizerType секция «КОНФИГУРАЦИИ ОПТИМИЗАТОРОВ…»

SchedulerType + get_lr там же

Hyperparams + default / print там же

Хранение в сети CatPolicyNetwork.hyperparams

В ModelConfig ModelConfig.hyperparams

Применение LR вызов reinforce_step(..., lr) в train-цикле



