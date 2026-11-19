# wine-rs

Разведочный анализ датасета **Wine** (UCI / `sklearn.datasets.load_wine`)
на Rust с использованием:

* **`linfa`** — нативный Rust-аналог scikit-learn, здесь используется для
  хранения данных в формате `Dataset` (матрица `X` + вектор меток `y`);
* **`rayon`** — параллельные вычисления при подсчёте частот классов, средних
  по классам и количестве значений выше среднего;
* **декларативный макрос `report!`** — единообразный вывод строк отчёта,
  формат которого строго совпадает с эталоном `sklearn`;
* **CSV** — встроенный датасет `data/wine.csv` (тот же Wine, что и в sklearn,
  с заголовком `target, alcohol, malic_acid, …, proline`).

## Архитектура (SOLID)

| Принцип | Где соблюдён |
| --- | --- |
| **SRP** | `WineDataset` — только загрузка/доступ к данным; `ParallelStats` — только расчёт метрик; макрос `report!` — только вывод. |
| **OCP** | Новый источник данных или новая стратегия расчёта добавляются реализацией трейта `FeatureSource` / `StatsComputer`, не меняя существующий код. |
| **LSP** | Любая реализация `FeatureSource` подставляется в `ParallelStats` без изменения поведения. |
| **ISP** | Трейты узкие: `FeatureSource` не включает методов расчёта, `StatsComputer` — методов загрузки. |
| **DIP** | `main` зависит от трейтов `FeatureSource` / `StatsComputer`, а не от конкретных типов; `ParallelStats` получает owned-данные, а не ссылку на конкретный `WineDataset`. |

## Структура

```
wine-rs/
├── Cargo.toml
├── data/
│   └── wine.csv          # 178 строк, 13 признаков + target (0..2)
└── src/
    ├── main.rs           # точка входа: чтение stdin, связывание модулей
    ├── macros.rs         # макрос report!
    ├── traits.rs         # трейты FeatureSource / StatsComputer (DIP/ISP/OCP)
    ├── dataset.rs        # WineDataset: загрузка CSV через linfa::Dataset
    └── stats.rs          # ParallelStats: rayon-вычисления трёх фактов
```

## Запуск

```bash
cargo build --release
echo "malic_acid" | ./target/release/wine-rs
```

Ожидаемый вывод (совпадает с эталоном sklearn):

```
Most wines: class_1 (71)
Highest malic_acid: class_2
class_2 wines above the average malic_acid: 41
```

Путь к CSV можно переопределить переменной окружения `WINE_CSV_PATH`:

```bash
WINE_CSV_PATH=/path/to/wine.csv echo "hue" | ./target/release/wine-rs
```

## Проверка

Вывод сверен с эталонным `sklearn.datasets.load_wine` для четырёх признаков
(`malic_acid`, `hue`, `alcohol`, `proline`) — полное совпадение всех трёх строк.
