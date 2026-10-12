# **Пространства действий и наблюдений в Reinforcement Learning**

Пространства RL — enum Space::Discrete / Space::Box с sample, contains и description.

 Среда объявляет observation_space = Box[0,1] и action_space = Discrete(2); step валидирует действия, сеть — наблюдения. Так агент и окружение разделяют один формальный контракт в стиле Gymnasium.

Пространства задают контракт между средой и агентом: что можно наблюдать и какие действия допустимы. Реализация следует идее Gymnasium (Discrete, Box, sample, contains).

1. Enum Space

```
pub enum Space {
    Discrete { n: usize },              // целые 0 .. n-1
    Box { low: Vec<f32>, high: Vec<f32> },  // непрерывный гиперпрямоугольник
}

```

Discrete { n } $n$ взаимоисключающих действий action space: n = 2 (feed / play)

Box { low, high } каждый компонент в $[low_i, high_i]$ observation space: [0.0] … [1.0]

2. Методы Space

sample() — случайный элемент

```
Discrete → vec![randint(0..n) as f32]
Box      → для каждой оси: low + (high - low) * U[0,1]

```

Нужен для exploration, отладки и random baseline (run_random_episode может опираться на похожий принцип).

contains(value: &[f32]) -> bool

Discrete длина 1 (как число)

Box та же размерность, что у low/high

Это валидация входа/действия до применения в среде или сети.

description()

Строки вида "Discrete(n=2)", "Box(low=[0.0], high=[1.0])" — для логов и ошибок (SpaceError).

3. Пространства в CatEnv


```
pub struct CatEnv {
    // ...
    pub observation_space: Space,  // Box { low: [0.0], high: [1.0] }
    pub action_space: Space,       // Discrete { n: 2 }
}
```

При CatEnv::new:


```
observation_space: Space::Box {
    low: vec![0.0],
    high: vec![1.0],
},
action_space: Space::Discrete { n: 2 },
```

Наблюдение всегда одно число: lives as f32 / 9.0 ∈ (0, 1] (после reset — 1.0).

Геттеры: observation_space(), action_space().

4. Действия: Action ↔ action_space


```
pub enum Action {
    Feed = 0,
    Play = 1,
}
```


as_usize() → индекс для contains и policy head;

from_u8 → разбор 0/1 (интерактив, trajectory).

Связь:

Action::Feed  ↔  0  ↔  policy[0]

Action::Play  ↔  1  ↔  policy[1]

action_space = Discrete(n=2)

5. Где пространства проверяются

В среде — step

```
if !self.action_space.contains(&[action.as_usize() as f32]) {
    return StepResult {
        observation: self.cat.get_state(),
        reward: -10.0,
        done: true,
        info: "Invalid action".into(),
    };
}
```

Невалидное действие → большой штраф и конец эпизода (не Err, а soft-fail внутри MDP).

В агенте — forward_multi_task


```
let obs_space = Space::Box { low: vec![0.0], high: vec![1.0] };
if !obs_space.contains(state) {
    return Err(AppError::SpaceError(...));
}
```

Сеть отказывается считать policy, если observation вне контракта → типизированная AppError::SpaceError.

6. Gym-подобный цикл с пространствами

reset() → obs ∈ observation_space

    │

    ▼

agent: policy(obs) → action ∈ action_space

    │

    ▼

env.step(action):

    contains(action)? 

        → feed/play → (obs', reward, done, info)

    │

    ▼

obs' снова в observation_space

StepResult — кортеж (observation, reward, done, info) в духе Gym (без отдельного truncated).

7. Связь с остальным пайплайном

Cat::get_state производит вектор под Box[0,1]

Preprocessing / clamp аугментаций держат state в bounds

Policy head (2 выхода) соответствует Discrete(n=2)

select_action argmax → Action

Ошибки AppError::SpaceError

Агент и среда согласованы: 1 continuous obs, 2 discrete actions.

8. Пример


```
let env = CatEnv::random();
println!("{}", env.observation_space().description());
// Box(low=[0.0], high=[1.0])

println!("{}", env.action_space().description());
// Discrete(n=2)

let obs = env.observation();           // e.g. [1.0]
assert!(env.observation_space().contains(&obs));

let a = Action::Feed;
assert!(env.action_space().contains(&[a.as_usize() as f32]));

let sample_a = env.action_space().sample(); // [0.0] или [1.0]
```



9. Где смотреть в коде

Space + sample/contains/description секция «ПРОСТРАНСТВА RL + СРЕДА»

Action, StepResult там же 

Присвоение spaces CatEnv::new

Проверка action CatEnv::step

Проверка obs forward_multi_task

Ошибка AppError::SpaceError

