# **Рекурсивные структуры (деревья решений, computation graphs)**

Рекурсивные структуры — это TreeNode с Box<TreeNode> (рекурсивный fit, обход до листа в predict) и ComputationNode с Box<ComputationNode> (рекурсивный forward по цепочке Input → Linear/Activation).

 Дерево встроено в multi-task выход сети как auxiliary; граф показывает, как в принципе устроен computation graph без полноценного autograd.

В коде два независимых рекурсивных типа:

TreeNode / DecisionTree — дерево решений (как linfa-trees / sklearn)
ComputationNode — мини computation graph (как autograd / упрощённый torch graph)

Оба используют Box<T>, потому что без указателя размер enum был бы бесконечным.

1. Почему нужна рекурсия и Box


```
enum TreeNode {
    Leaf { ... },
    Split {
        left: Box<TreeNode>,   // дочерний узел
        right: Box<TreeNode>,
        ...
    },
}
```

TreeNode содержит сам себя. Компилятор должен знать размер типа на стеке; Box хранит только указатель (фиксированный размер), а узел — в куче.

То же для графа:
```
Linear { input: Box<ComputationNode>, ... }

```

2. Дерево решений

SplitQuality

```
pub enum SplitQuality {
    Gini,     // классификация
    Entropy,  // information gain
    MSE,      // регрессия
}
```

Критерий «качества» сплита (как в linfa-trees). В текущем build_tree значение quality задаётся упрощённо (0.85), а не считается по полной формуле Gini/Entropy — enum готов к расширению.
TreeNode

Leaf value, samples предсказание (среднее targets)

Split feature_idx, threshold, left, right, quality, split_quality_type ветвление

Для этой задачи признак один (lives/9.0), поэтому feature_idx всегда 0.

DecisionTree

```
pub struct DecisionTree {
    pub root: Option<TreeNode>,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub quality: SplitQuality,
}
```

fit


```
pub fn fit(&mut self, states: &[f32], targets: &[f32]) -> Result<(), AppError>
```


проверка states.len() == targets.len() → иначе DimensionMismatch;

пустые данные → EmptyTensor;

self.root = Some(self.build_tree(states, targets, 0)).

build_tree — рекурсивное построение

Алгоритм:

Стоп, если len < min_samples_split или depth >= max_depth→ Leaf со средним targets.

Сортировка пар (state, target) по state.

Порог = медиана state (упрощённый поиск сплита, не полный grid search).

Разделение на left (s < threshold) и right.

Если одна сторона пустая → снова Leaf (защита от бесконечной рекурсии).

Иначе:

```
left  = Box::new(build_tree(left_..., depth + 1))
right = Box::new(build_tree(right_..., depth + 1))
→ TreeNode::Split { ... }
```


Это классический top-down recursive partitioning.

predict — обход до листа


```
pub fn predict(&self, state: f32) -> f32 {
    let mut node = self.root.as_ref();
    while let Some(TreeNode::Split { threshold, left, right, .. }) = node {
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
```

Итеративный спуск по Box-ссылкам (без рекурсии стека на predict).

Результат — скаляр (например, «вероятность feed» как auxiliary target).

3. Интеграция дерева в нейросеть

В CatPolicyNetwork:

pub decision_tree: Option<DecisionTree>,

При создании (особенно HybridTreeNN) дерево обучается на простых парах state → target и вызывается в multi-task forward:


```
let tree_value = self.decision_tree.as_ref().map(|t| t.predict(input_state[0]));
// → MultiTaskOutput { policy, value, tree_value }
```


Гибрид: MLP даёт policy/value, дерево — независимую оценку (auxiliary task / интерпретируемый сигнал).

4. Computation Graph


```
pub enum ComputationNode {
    Input { value: f32 },
    Linear {
        weight: f32,
        bias: f32,
        input: Box<ComputationNode>,
    },
    Activation {
        func: fn(f32) -> f32,   // например relu
        input: Box<ComputationNode>,
    },
    Softmax { input: Box<ComputationNode> },
}
```

Каждый узел знает только предыдущий (input: Box<...>). Цепочка — это граф (здесь — цепочка, DAG из одного пути).

forward — рекурсивный проход


```
pub fn forward(&self) -> f32 {
    match self {
        Input { value } => *value,
        Linear { weight, bias, input } => {
            weight * input.forward() + bias
        }
        Activation { func, input } => {
            func(input.forward())
        }
        Softmax { input } => {
            // упрощённо sigmoid для скаляра
            let x = input.forward();
            1.0 / (1.0 + (-x).exp())
        }
    }
}
```

Пример сборки:

Input(1.2)

   → Activation(relu)

        → Linear(w=2, b=0.5)

             → forward() = 2 * relu(1.2) + 0.5

Это демонстрация идеи autograd graph: операции — узлы, данные текут снизу вверх. Обратного прохода (backward) в коде нет — только forward.


5. Схема дерева


Split(threshold=0.5)

                   /                    \

          Split(t=0.25)              Leaf(value=0.2)

         /             \

   Leaf(0.9)        Leaf(0.6)

predict(0.3): 0.3 < 0.5 → left → 0.3 >= 0.25 → Leaf(0.6).

6. Схема computation graph

Input(x)

   │

   ▼

Activation(relu)     ← input: Box → Input

   │

   ▼

Linear(w, b)         ← input: Box → Activation

   │

   ▼

forward() → скаляр


8. Где смотреть в коде

SplitQuality, TreeNode, DecisionTree «РЕКУРСИВНЫЕ СТРУКТУРЫ: ДЕРЕВЬЯ РЕШЕНИЙ»

fit / build_tree / predict impl DecisionTreeComputationNode + forward«COMPUTATION GRAPH»

Использование CatPolicyNetwork.decision_tree, forward_multi_task → tree_value

