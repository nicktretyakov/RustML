Теория
Обучение с учителем
Обучение с учителем включает в себя обучение модели на наборе данных, содержащем как входные признаки, так и соответствующие правильные выходные данные (метки). Цель состоит в том, чтобы предсказать выходные данные для новых, ранее не встречавшихся входных данных на основе изученной функции. Существует два основных типа обучения с учителем: Классификация: цель состоит в том, чтобы предсказать метку класса, которая представляет собой выбор из предопределенного списка возможностей. Например, представьте, что вы пытаетесь отсортировать электронные письма на «спам» или «не спам». Или когда ваш телефон распознает ваше лицо для разблокировки, он использует классификацию, чтобы сказать: «Да, это вы!» Регрессия: цель состоит в том, чтобы предсказать величину. Например, предсказать цену дома на основе различных характеристик, таких как его размер и местоположение, или определить, сколько стоит автомобиль, исходя из его марки, модели и возраста. Это как играть в «Цена правильная». По мере изучения различных алгоритмов в этом курсе мы будем говорить о том, являются ли они больше похожими на обучение с учителем или на обучение без учителя.
Обучение без учителя 
Включает в себя обучение модели на неразмеченном наборе данных. Цель — обнаружить скрытые закономерности или структуры в данных. Это как дать ребенку большую коробку с кубиками LEGO без инструкции. Цель — увидеть, какие интересные структуры или закономерности он сам найдет. Примеры включают задачи кластеризации и уменьшения размерности: Кластеризация: Цель — сгруппировать похожие экземпляры в кластеры. Это как организовать свои игрушки в группы, не объясняя, как это сделать. Например, магазин может разделить покупателей на разные типы, чтобы определить наилучший способ рекламы для каждого типа. Или в социальных сетях это помогает определить, кто в какой группе друзей. Уменьшение размерности: Метод, который упрощает сложность многомерных данных, сохраняя при этом их основные характеристики. Представьте, что у вас есть огромный рюкзак, полный школьных принадлежностей, но вы хотите сделать его легче. Уменьшение размерности — это как определить, что важно, а что нет, чтобы вы носили только то, что вам действительно нужно. Это помогает справляться с избытком информации — например, когда вы пытаетесь упростить очень сложную картинку, чтобы её было легче понять. В ходе изучения различных алгоритмов в этом курсе мы будем обсуждать, относятся ли они больше к контролируемому или неконтролируемому обучению.
Алгоритм KNN
Метод KNN не имеет явной фазы обучения. Он просто сохраняет набор данных. Только на этапе прогнозирования он вычисляет расстояние между новой точкой и остальными.
Алгоритм KNN (K-ближайших соседей) — это алгоритм обучения с учителем, используемый как для классификации, так и для регрессии. Он прост, но мощен и часто используется в качестве базового алгоритма для более сложных моделей. Алгоритм KNN находит сходство признаков, используя метрики расстояния для определения ближайших соседей к точке. Например, он может сортировать вина, сравнивая их с ближайшими соседями, например, группируя вина со схожими вкусовыми качествами. Это как иметь гида, который помогает вам найти похожие вина на основе их наиболее общих характеристик. 
Идея KNN довольно проста: Для заданной тестовой точки алгоритм вычисляет расстояние между этой точкой и каждой другой точкой в ​​обучающем наборе данных. Расстояние может быть вычислено различными способами, например, евклидовым или манхэттенским расстоянием. Поиск ближайших соседей: Затем алгоритм выбирает k точек в обучающих данных, которые находятся ближе всего к тестовой точке. k — это определяемая пользователем константа, которая может быть любым целым числом. Предсказание: Для классификации алгоритм присваивает тестовой точке метку наиболее часто встречающегося класса среди k ближайших соседей. Для регрессии он присваивает среднее значение меток «k» ближайших соседей. KNN — это алгоритм ленивого обучения, то есть он фактически не обучается модели. Вместо этого он запоминает обучающие примеры и использует их для предсказания (в следующем уроке будет рассмотрена фаза обучения). Это делает KNN вычислительно затратным на этапе тестирования. Несмотря на свою простоту, KNN может показать удивительно хорошие результаты. Однако он чувствителен к выбору k и типу метрики расстояния, и плохо работает с многомерными данными из-за «проклятия размерности».
fn distances_option_calculator(X: &[i32], y: &[i32], distance_type: &str) -> f64 {
    match distance_type {
        "euclidian" => {
            let sum_sq: i32 = X.iter().zip(y).map(|(&a, &b)| (a - b).pow(2)).sum();
            (sum_sq as f64).sqrt()
        }
        "manhattan" => {
            let sum_abs: i32 = X.iter().zip(y).map(|(&a, &b)| (a - b).abs()).sum();
            sum_abs as f64
        }
        "hamming" => {
            let diff_count: i32 = X.iter().zip(y).filter(|(&a, &b)| a != b).count() as i32;
            diff_count as f64
        }
        _ => panic!("distance_type must be 'euclidian', 'manhattan', or 'hamming'"),
    }
}
use std::f64;

struct KNN<T> {
    points: Vec<Vec<f64>>,
    labels: Vec<T>,
    k: usize,
}

impl<T> KNN<T> {
    // Конструктор
    fn new(k: usize) -> Self {
        KNN {
            points: Vec::new(),
            labels: Vec::new(),
            k,
        }
    }

    // Евклидово расстояние между двумя точками (векторами одинаковой длины)
    fn distance(&self, point_a: &[f64], point_b: &[f64]) -> f64 {
        let sum_sq: f64 = point_a
            .iter()
            .zip(point_b)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        sum_sq.sqrt()
    }

    // Сохраняет обучающие точки и метки
    fn fit(&mut self, X_train: Vec<Vec<f64>>, y_train: Vec<T>) {
        self.points = X_train;
        self.labels = y_train;
    }

    // Заглушка для predict (как в исходном коде)
    fn predict(&self, _X_test: Vec<Vec<f64>>) -> Vec<T> {
        todo!("Метод predict пока не реализован")
    }
}

fn main() {
    // Пример использования 
    let mut knn = KNN::new(3);
    
    // Обучающие точки: (1,2), (2,3), (3,5)
    let X_train = vec![
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 5.0],
    ];
    // Метки: A, B, A
    let y_train = vec!["A", "B", "A"];
    
    // Вызов fit
    knn.fit(X_train, y_train);
    println!("Fit method pass");
    
    // Проверка distance на двух точках
    let dist = knn.distance(&[1.0, 2.0], &[2.0, 3.0]);
    println!("Distance method pass (distance = {})", dist);
}
use std::collections::HashMap;

struct KNN {
    points: Vec<Vec<f64>>,
    labels: Vec<String>,
    k: usize,
}

impl KNN {
    fn new(k: usize) -> Self {
        KNN {
            points: Vec::new(),
            labels: Vec::new(),
            k,
        }
    }

    // Евклидово расстояние между двумя точками (векторами одинаковой длины)
    fn distance(&self, point_a: &[f64], point_b: &[f64]) -> f64 {
        let sum_sq: f64 = point_a
            .iter()
            .zip(point_b)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        sum_sq.sqrt()
    }

    // Запоминает обучающие точки и метки
    fn fit(&mut self, X_train: Vec<Vec<f64>>, y_train: Vec<String>) {
        self.points = X_train;
        self.labels = y_train;
    }

    // Предсказывает метки для тестовых точек
    fn predict(&self, X_test: Vec<Vec<f64>>) -> Vec<String> {
        let mut predictions = Vec::new();

        for test_point in X_test {
            // Вычисляем расстояния до всех обучающих точек с сохранением индексов
            let mut distances: Vec<(usize, f64)> = self.points
                .iter()
                .enumerate()
                .map(|(i, train_point)| (i, self.distance(&test_point, train_point)))
                .collect();

            // Сортируем по возрастанию расстояния
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Берём индексы k ближайших соседей
            let k_nearest_indices: Vec<usize> = distances
                .iter()
                .take(self.k)
                .map(|&(idx, _)| idx)
                .collect();

            // Получаем метки этих соседей
            let k_nearest_labels: Vec<&String> = k_nearest_indices
                .iter()
                .map(|&idx| &self.labels[idx])
                .collect();

            // Подсчитываем частоту каждой метки
            let mut freq = HashMap::new();
            for label in &k_nearest_labels {
                *freq.entry(*label).or_insert(0) += 1;
            }

            // Находим метку с максимальной частотой (при равных - берём первую по порядку)
            let mut best_label = k_nearest_labels[0];
            let mut max_count = 0;
            for label in k_nearest_labels {
                let count = freq[label];
                if count > max_count {
                    max_count = count;
                    best_label = label;
                }
            }

            predictions.push(best_label.clone());
        }

        predictions
    }
}

// Пример использования 
fn main() {
    let mut knn = KNN::new(3);

    let X_train = vec![
        vec![1.0, 5.0],
        vec![5.0, 0.0],
        vec![3.0, 2.0],
        vec![8.0, 1.0],
        vec![4.0, 4.0],
        vec![7.0, 7.0],
    ];
    let y_train = vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "A".to_string(),
    ];

    knn.fit(X_train, y_train);

    let X_test = vec![vec![3.0, 3.0]];
    let predictions = knn.predict(X_test);
    println!("{:?}", predictions); // Выведет ["A"]
}
Введение в алгоритм K-средних 
Алгоритм K-средних без учителя — цель разбить n наблюдений на k кластеров, в каждом из которых каждое наблюдение принадлежит кластеру с ближайшим средним значением (центроидом), который служит прототипом кластера. Допустим, у вас есть набор разных фруктов, и вы хотите рассортировать их по корзинам в зависимости от типа, но у вас нет меток. Алгоритм K-средних поможет вам сделать это, но с точками данных вместо фруктов!
Алгоритм использует итеративную процедуру для разделения набора данных на k кластеров. 
Инициализация центроидов: Сначала выберите k точек в качестве начальных центроидов из набора данных. Эти точки могут быть выбраны случайным образом или на основе определенной стратегии.
Назначение кластеров: Для каждой точки в наборе данных найдите ближайший центроид (используя меры расстояния, такие как евклидово расстояние) и назначьте точку этому кластеру. 
Обновление центроидов: После того, как все точки назначены кластерам, пересчитайте центроиды, взяв среднее значение всех точек в каждом кластере. Повторение шагов 2 и 3: Вышеописанные шаги повторяются до тех пор, пока центроиды перестанут существенно изменяться. Это означает, что алгоритм сошелся, и кластеры стали стабильными.
fn euclidian_distance(point_a: &[f64], point_b: &[f64]) -> f64 {
    point_a.iter()
        .zip(point_b)
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn wcss(points: &[Vec<f64>]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    let n_points = points.len();
    let n_dims = points[0].len();

    // Compute centroid (mean of all points)
    let mut centroid = vec![0.0; n_dims];
    for point in points {
        for i in 0..n_dims {
            centroid[i] += point[i];
        }
    }
    for i in 0..n_dims {
        centroid[i] /= n_points as f64;
    }

    // Sum Euclidean distances from each point to centroid
    let total_distance: f64 = points.iter()
        .map(|point| euclidian_distance(point, &centroid))
        .sum();

    // Return average distance
    total_distance / n_points as f64
}

// Пример использования
fn main() {
    let points = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ];
    let result = wcss(&points);
    println!("WCSS: {}", result);
}
fn euclidian_distance(point_a: &[f64], point_b: &[f64]) -> f64 {
    point_a.iter()
        .zip(point_b)
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

struct KMeans {
    k: usize,
    centroids: Vec<Vec<f64>>,
}

impl KMeans {
    fn new(k: usize) -> Self {
        KMeans {
            k,
            centroids: Vec::new(),
        }
    }

    fn find_cluster_centroid(&self, cluster_points: &[Vec<f64>]) -> Option<Vec<f64>> {
        if cluster_points.is_empty() {
            return None;
        }
        let n_dims = cluster_points[0].len();
        let n_points = cluster_points.len() as f64;
        let mut centroid = vec![0.0; n_dims];
        for point in cluster_points {
            for i in 0..n_dims {
                centroid[i] += point[i];
            }
        }
        for i in 0..n_dims {
            centroid[i] /= n_points;
        }
        Some(centroid)
    }

    fn fit(&mut self, X_train: Vec<Vec<f64>>) {
        // Step 1: Initialize centroids using 3rd, 5th, 7th, ... points (indices 2,4,6,...)
        let mut indices = Vec::new();
        for i in 0..self.k {
            indices.push(2 + 2 * i);
        }
        self.centroids = indices.iter()
            .map(|&idx| X_train[idx].clone())
            .collect();

        // Iterate until convergence
        loop {
            // Step 2: Assign each point to nearest centroid
            let mut clusters: Vec<Vec<Vec<f64>>> = vec![Vec::new(); self.k];
            for point in &X_train {
                let mut best_idx = 0;
                let mut min_dist = f64::INFINITY;
                for (idx, cent) in self.centroids.iter().enumerate() {
                    let dist = euclidian_distance(point, cent);
                    if dist < min_dist {
                        min_dist = dist;
                        best_idx = idx;
                    }
                }
                clusters[best_idx].push(point.clone());
            }

            // Step 3: Recalculate centroids
            let mut new_centroids = Vec::with_capacity(self.k);
            for i in 0..self.k {
                if !clusters[i].is_empty() {
                    if let Some(cent) = self.find_cluster_centroid(&clusters[i]) {
                        new_centroids.push(cent);
                    } else {
                        new_centroids.push(self.centroids[i].clone());
                    }
                } else {
                    new_centroids.push(self.centroids[i].clone());
                }
            }

            // Calculate total movement of centroids
            let total_movement: f64 = self.centroids.iter()
                .zip(&new_centroids)
                .map(|(old, new)| euclidian_distance(old, new))
                .sum();

            // Step 4: Check stop condition
            if total_movement < 0.01 {
                // Keep previous centroids (do not update)
                break;
            } else {
                self.centroids = new_centroids;
            }
        }
    }

    fn predict(&self, _X_test: Vec<Vec<f64>>) {
        // Not required for this challenge, but keep as placeholder
    }
}

// Example usage (optional)
fn main() {
    let mut kmeans = KMeans::new(2);
    let X_train = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
        vec![7.0, 8.0],
        vec![9.0, 10.0],
    ];
    kmeans.fit(X_train);
    println!("Centroids: {:?}", kmeans.centroids);
}
Наивный байесовский классификатор 
Наивный байесовский классификатор обучается с учителем — для задач классификации в машинном обучении. Алгоритм наивного байесовского классификатора основан на теореме Байеса, которая объясняет, как рассчитать вероятность события (например, того, что электронное письмо является спамом) на основе предварительных знаний об условиях, которые могут быть связаны с этим событием. Слово «наивный» происходит от предположения, что все признаки (например, слова в электронном письме) независимы друг от друга, что упрощает вычисления. Независимость означает, что одно событие не меняет вероятности другого события.
Подсчет вхождений: Сначала подсчитывается, сколько раз каждая точка данных встречается в разных классах. В обнаружении спама мы будем подсчитывать, сколько раз каждое слово встречалось в спам- или неспам-письмах, в медицинской диагностике мы можем подсчитать, сколько раз (или как долго) встречается каждый симптом. 
Вычисление вероятностей: Для новой точки данных вычисляется вероятность принадлежности к каждому классу (какова вероятность того, что новое письмо является спамом, и какова вероятность того, что оно не является спамом). 
Принятие решения: Наконец, алгоритм объединяет все эти вероятности, используя теорему Байеса, и решает, является ли письмо с большей вероятностью спамом или не спамом. Чем выше вероятность, тем больше вероятность принадлежности к этому классу. 
Преимущество наивного байесовского классификатора заключается в его быстрой и простой реализации. Он хорошо работает с многомерными данными (например, текстом) и может обрабатывать пропущенные значения. Однако предположение о независимости не всегда верно в реальной жизни, что иногда может приводить к менее точным результатам по сравнению с более сложными алгоритмами.
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, stdin};
use csv::ReaderBuilder;

fn main() {
    // чтение пути к файлу и имени целевого столбца из stdin
    let mut input_line = String::new();
    stdin().read_line(&mut input_line).unwrap();
    let dataset_path = input_line.trim();
    let mut target_var = String::new();
    stdin().read_line(&mut target_var).unwrap();
    let target_variable = target_var.trim();

    // открытие CSV
    let file = File::open(dataset_path).expect("Не удалось открыть файл");
    let reader = BufReader::new(file);
    let mut csv_reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(reader);
    let headers = csv_reader.headers().unwrap().clone();

    // чтение всех строк в Vec<BTreeMap<String, String>>
    let mut rows: Vec<BTreeMap<String, String>> = Vec::new();
    for result in csv_reader.records() {
        let record = result.unwrap();
        let mut row = BTreeMap::new();
        for (idx, field) in record.iter().enumerate() {
            let col_name = headers[idx].to_string();
            row.insert(col_name, field.to_string());
        }
        rows.push(row);
    }

    let total_rows = rows.len();

    // уникальные значения целевого столбца
    let mut target_values = Vec::new();
    for row in &rows {
        let val = row.get(target_variable).unwrap();
        if !target_values.contains(val) {
            target_values.push(val.clone());
        }
    }

    // P(target) – вероятность каждого класса
    let mut probability_target = BTreeMap::new();
    for tval in &target_values {
        let count = rows.iter()
            .filter(|row| row.get(target_variable).unwrap() == tval)
            .count();
        probability_target.insert(tval.clone(), count as f64 / total_rows as f64);
    }

    // P(target | feature) для каждого признака
    let mut probability_features: BTreeMap<String, BTreeMap<String, BTreeMap<String, f64>>> = BTreeMap::new();
    for col in headers.iter() {
        let col = col.to_string();
        if col == target_variable { continue; }

        // уникальные значения признака
        let mut distinct_vals = Vec::new();
        for row in &rows {
            let val = row.get(&col).unwrap();
            if !distinct_vals.contains(val) {
                distinct_vals.push(val.clone());
            }
        }

        let mut feature_map = BTreeMap::new();
        for val in &distinct_vals {
            let count_col_val = rows.iter()
                .filter(|row| row.get(&col).unwrap() == val)
                .count();
            let mut target_map = BTreeMap::new();
            for tval in &target_values {
                let total = rows.iter()
                    .filter(|row| row.get(&col).unwrap() == val && row.get(target_variable).unwrap() == tval)
                    .count();
                let prob = if count_col_val > 0 {
                    total as f64 / count_col_val as f64
                } else {
                    0.0
                };
                target_map.insert(tval.clone(), prob);
            }
            feature_map.insert(val.clone(), target_map);
        }
        probability_features.insert(col, feature_map);
    }

    // вывод в формате, похожем на Python
    print_features(&probability_features);
    print_target(&probability_target);
}

// Вспомогательная функция для форматирования числа как в Python (0.0 вместо 0)
fn format_prob(prob: f64) -> String {
    if prob.fract() == 0.0 {
        format!("{:.1}", prob)   // "0.0", "1.0"
    } else {
        // 15 знаков после запятой, как часто бывает в repr() float в Python
        format!("{:.15}", prob).trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn print_features(pf: &BTreeMap<String, BTreeMap<String, BTreeMap<String, f64>>>) {
    print!("{{");
    let mut first_outer = true;
    for (col, inner_map) in pf {
        if !first_outer { print!(", "); }
        first_outer = false;
        print!("'{}': {{", col);
        let mut first_mid = true;
        for (val, target_map) in inner_map {
            if !first_mid { print!(", "); }
            first_mid = false;
            print!("'{}': {{", val);
            let mut first_inner = true;
            for (targ, prob) in target_map {
                if !first_inner { print!(", "); }
                first_inner = false;
                print!("'{}': {}", targ, format_prob(*prob));
            }
            print!("}}");
        }
        print!("}}");
    }
    println!("}}");
}

fn print_target(pt: &BTreeMap<String, f64>) {
    print!("{{");
    let mut first = true;
    for (cls, prob) in pt {
        if !first { print!(", "); }
        first = false;
        print!("'{}': {}", cls, format_prob(*prob));
    }
    println!("}}");
}
Как это работает
    1. Читает путь к CSV и имя целевого столбца из стандартного ввода.
    2. Парсит CSV, сохраняя строки как отображения «имя_столбца → значение».
    3. Вычисляет P(target) – долю каждого класса.
    4. Для каждого нецелевого столбца и каждого его уникального значения вычисляет P(target | feature = значение) как отношение количества строк с данным значением признака и данным целевым классом к общему числу строк с этим значением признака.
    5. Выводит два отсортированных словаря.
Логистическая регрессия
Логистическая регрессия, является линейной моделью, используемой в основном для задач классификации, а не регрессии. Она идеально подходит для ситуаций, когда необходимо предсказать наличие или отсутствие чего-либо, например, является ли электронное письмо спамом. Логистическая регрессия хорошо работает для бинарной классификации, но может быть расширена до многоклассовой классификации с помощью различных методов. Источник: Википедия По своей сути, логистическая регрессия предсказывает вероятность того, что данная входная точка принадлежит к определённому классу. В отличие от линейной регрессии, которая предсказывает непрерывный результат, логистическая регрессия предсказывает бинарный результат (1/0, Да/Нет). Суть логистической регрессии заключается в логистической функции или сигмоидной функции, которая отображает любое действительное число в значение от 0 до 1, что делает её подходящей для оценки вероятностей. Подгонка модели логистической регрессии включает в себя поиск наилучшего набора коэффициентов для уравнения с учетом обучающих данных. Хотя это эффективный и простой метод для бинарной классификации, его производительность может отставать от более сложных моделей для сложных наборов данных.
fn sigmoid(point: &[f64]) -> Vec<f64> {
    point.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

// Пример использования
fn main() {
    let data = vec![-1.0, 0.0, 1.0, 2.0];
    let result = sigmoid(&data);
    println!("{:?}", result);
}
Линейная регрессия 
Для прогнозирования непрерывных переменных - обучается с учителем. Это основной метод оценки взаимосвязей между переменными и прогнозирования. Независимо от того, прогнозируете ли вы цены на жилье, тенденции фондового рынка или температуру, линейная регрессия может дать представление о том, как эти переменные коррелируют.Линейная регрессия пытается смоделировать взаимосвязь между двумя или более переменными путем подгонки линейного уравнения к наблюдаемым данным. Одна переменная считается объясняющей переменной (независимой), а другая — зависимой переменной. Простейшая форма уравнения линейной регрессии с одной зависимой и одной независимой переменной определяется следующим образом: y = α + βx α обозначает точку пересечения с осью y β обозначает наклон.
fn linear_equation(coef: &[f64], x: &[f64]) -> f64 {
    // coef[0] + coef[1]*x[0] + coef[2]*x[1] + ...
    let mut result = coef[0];
    for (coef_i, x_i) in coef[1..].iter().zip(x.iter()) {
        result += coef_i * x_i;
    }
    result
}
Дерево решений
Нелинейная модель обучается с учителем, используемая как для классификации, так и для регрессии. Оно более точно, чем другие алгоритмы, имитирует процесс принятия решений человеком, что делает его интуитивно понятным и мощным инструментом для обработки сложных наборов данных. По своей сути, дерево решений разделяет данные на подмножества, используя ряд простых правил, что сродни задаванию ряда вопросов «да/нет» о характеристиках точек данных. Представьте дерево решений в виде древовидного графа, где: Каждый внутренний узел представляет собой «проверку» атрибута (например, выпадет ли орёл или решка при подбрасывании монеты), Каждая ветвь представляет собой результат проверки, и Каждый листовой узел представляет собой метку класса (решение, принятое после вычисления всех атрибутов). Типы деревьев решений: Деревья классификации: Используются, когда прогнозируемый результат — это класс, к которому принадлежат данные. Деревья регрессии: используются, когда прогнозируемый результат можно считать действительным числом (например, цена дома).
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
enum RuleType {
    Bigger,
    Smaller,
}

fn decide_outcome(
    rules: &HashMap<String, (RuleType, f64)>,
    outcome: &HashMap<String, HashMap<bool, Option<String>>>,
    point: &HashMap<String, f64>,
) -> String {
    let mut current_rule = "rule1".to_string();

    loop {
        // Get rule condition for current rule
        let (rule_type, rule_value) = rules.get(&current_rule).expect("Rule not found");

        let point_value = point.get(&current_rule).expect("Point value not found");

        let condition_met = match rule_type {
            RuleType::Bigger => *point_value > *rule_value,
            RuleType::Smaller => *point_value < *rule_value,
        };

        // Determine next rule from outcome
        let next_rule = outcome
            .get(&current_rule)
            .expect("Outcome for rule not found")
            .get(&condition_met)
            .expect("Outcome branch not found");

        if let Some(next) = next_rule {
            current_rule = next.clone();
        } else {
            return current_rule;
        }
    }
}

fn main() {
    let mut rules = HashMap::new();
    rules.insert("rule1".to_string(), (RuleType::Bigger, 5.0));
    rules.insert("rule2".to_string(), (RuleType::Smaller, 7.0));

    let mut outcome = HashMap::new();
    let mut rule1_outcome = HashMap::new();
    rule1_outcome.insert(true, Some("rule2".to_string()));
    rule1_outcome.insert(false, Some("rule3".to_string()));
    outcome.insert("rule1".to_string(), rule1_outcome);

    let mut rule2_outcome = HashMap::new();
    rule2_outcome.insert(true, Some("rule3".to_string()));
    rule2_outcome.insert(false, Some("rule4".to_string()));
    outcome.insert("rule2".to_string(), rule2_outcome);

    let mut rule3_outcome = HashMap::new();
    rule3_outcome.insert(true, None);
    rule3_outcome.insert(false, None);
    outcome.insert("rule3".to_string(), rule3_outcome);

    let mut rule4_outcome = HashMap::new();
    rule4_outcome.insert(true, None);
    rule4_outcome.insert(false, None);
    outcome.insert("rule4".to_string(), rule4_outcome);

    let mut point = HashMap::new();
    point.insert("rule1".to_string(), 8.0);
    point.insert("rule2".to_string(), 10.0);

    let result = decide_outcome(&rules, &outcome, &point);
    println!("{}", result); // rule4
}
Метод опорных векторов (SVM) 
Мощный алгоритм машинного обучения с учителем, используемый как для классификации, так и для регрессии. Однако чаще всего он применяется в задачах классификации. SVM обеспечивают высокую точность и хорошо работают с многомерным пространством признаков, что делает их подходящими для приложений от распознавания изображений до биоинформатики. Основная цель алгоритма SVM — найти оптимальную гиперплоскость (в двумерном пространстве это линия), которая наилучшим образом разделяет классы в пространстве признаков. Модель SVM представляет примеры в виде точек в пространстве, отображаемых таким образом, что примеры отдельных категорий разделены четким, максимально широким промежутком. Гиперплоскость: это, по сути, граница принятия решений, разделяющая классы. В двумерном пространстве эта гиперплоскость представляет собой линию, но в более высоких измерениях она может быть более сложной. Опорные векторы: это точки данных, ближайшие к гиперплоскости, которые имеют решающее значение для определения положения гиперплоскости. Классификатор SVM сильно зависит от этих точек, отсюда и название «машина опорных векторов». Отступ: это расстояние между двумя линиями на ближайших точках данных разных классов. Его можно рассматривать как отдельную меру между классами. Обеспечение максимально широкого отступа является ключевой задачей при обучении SVM. Типы SVM: Линейный SVM: Когда данные линейно разделимы, мы можем использовать одну прямую линию для разделения классов. В этом сценарии используется линейный SVM. Нелинейный SVM: Когда данные не линейно разделимы, SVM использует так называемый «ядерный трюк» для преобразования входного пространства в многомерное пространство, где для разделения можно использовать гиперплоскость. Ядерный трюк включает в себя преобразование данных в другое измерение, имеющее четкий отступ между классами данных. Это позволяет SVM решать нелинейные задачи. Распространенные ядра включают: Линейное ядро: Используется, когда данные линейно разделимы. Полиномиальное ядро: Преобразует исходное пространство признаков в полиномиальное пространство признаков заданной степени. Гауссово ядро: Полезно для нелинейных данных, которые группируются в кластеры. Оно отображает пространство признаков в многомерное пространство.
fn polynomial_kernel(x: &[f64], y: &[f64], d: i32, c: f64) -> f64 {
    // Скалярное произведение x и y
    let dot: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    // (dot + c)^d
    (dot + c).powi(d)
}

fn main() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];
    let d = 3;
    let c = 1.0;
    let result = polynomial_kernel(&x, &y, d, c);
    println!("{}", result);
}
Простейшая модель
Общая идея
Модель запоминает для каждого уникального значения X (например, "C", "B") самое частое соответствующее значение y. Если для одного X встречается несколько разных y, то выбирается то, которое встречается чаще. Если частоты одинаковые (ничья), то выбирается наименьшее числовое значение y. При предсказании для нового X модель возвращает запомненную моду, а если X не встречался при обучении — возвращает None.
use std::collections::HashMap;

struct SimpleModel {
    mode: HashMap<String, i32>,
}

impl SimpleModel {
    fn new() -> Self {
        SimpleModel {
            mode: HashMap::new(),
        }
    }

    fn fit(&mut self, x_train: &[&str], y_train: &[i32]) {
        // Группируем значения y по каждому x
        let mut groups: HashMap<String, Vec<i32>> = HashMap::new();
        for (x, y) in x_train.iter().zip(y_train.iter()) {
            groups.entry(x.to_string()).or_insert_with(Vec::new).push(*y);
        }

        // Для каждой группы вычисляем моду (при ничьей выбираем наименьшее значение)
        for (x_val, y_list) in groups {
            let mut freq: HashMap<i32, usize> = HashMap::new();
            for &y in &y_list {
                *freq.entry(y).or_insert(0) += 1;
            }
            let max_freq = freq.values().cloned().max().unwrap_or(0);
            let mut candidates: Vec<i32> = freq
                .into_iter()
                .filter(|&(_, f)| f == max_freq)
                .map(|(y, _)| y)
                .collect();
            candidates.sort(); // минимальное значение будет первым
            self.mode.insert(x_val, candidates[0]);
        }
    }

    fn predict(&self, x_test: &[&str]) -> Vec<Option<i32>> {
        x_test.iter()
            .map(|x| self.mode.get(*x).cloned())
            .collect()
    }
}

fn train_and_predict(x_train: &[&str], y_train: &[i32], x_test: &[&str]) -> Vec<Option<i32>> {
    let mut model = SimpleModel::new();
    model.fit(x_train, y_train);
    model.predict(x_test)
}

// Пример использования (можно закомментировать)
fn main() {
    let x_train = ["C", "C", "B"];
    let y_train = [1, 0, 0];
    let x_test = ["C"];
    let result = train_and_predict(&x_train, &y_train, &x_test);
    println!("{:?}", result); // [Some(0)]
}
Структура данных
struct SimpleModel {
    mode: HashMap<String, i32>,
}
    • mode — это словарь (хеш-таблица), где ключом является строка (String) — значение признака X, а значением — i32 — предсказанная метка (мода). В Rust типы нужно указывать явно: HashMap<K, V>.
Конструктор
fn new() -> Self {
    SimpleModel {
        mode: HashMap::new(),
    }
}
Создаёт новый экземпляр модели с пустым словарём.
Метод fit (обучение)
fn fit(&mut self, x_train: &[&str], y_train: &[i32]) {
    let mut groups: HashMap<String, Vec<i32>> = HashMap::new();
    for (x, y) in x_train.iter().zip(y_train.iter()) {
        groups.entry(x.to_string()).or_insert_with(Vec::new).push(*y);
    }
    // ... далее вычисление моды
}
Параметры
    • &mut self — метод изменяет модель (заполняет mode), поэтому принимает изменяемую ссылку.
    • x_train: &[&str] — срез строковых срезов (например, &["C", "C", "B"]). Каждый элемент — это &str (ссылка на строку, обычно статическую или из кучи). Приём &[&str] позволяет передавать массив строк без копирования.
    • y_train: &[i32] — срез целых чисел.
Шаг 1. Группировка значений y по x
let mut groups: HashMap<String, Vec<i32>> = HashMap::new();
for (x, y) in x_train.iter().zip(y_train.iter()) {
    groups.entry(x.to_string()).or_insert_with(Vec::new).push(*y);
}
    • x_train.iter().zip(y_train.iter()) — создаёт итератор по парам (&str, &i32).
    • entry(x.to_string()) — получает запись в словаре по ключу String. x.to_string() создаёт владеющую строку из &str, потому что ключом в HashMap должна быть владеющая строка (иначе данные могут исчезнуть).
    • or_insert_with(Vec::new) — если ключа ещё нет, вставляет пустой вектор Vec<i32> и возвращает изменяемую ссылку на него. Если ключ есть, просто возвращает ссылку на уже существующий вектор.
    • .push(*y) — разыменовывает y (которое &i32) и добавляет значение в вектор.
В итоге для примера x_train = ["C", "C", "B], y_train = [1, 0, 0] получим:
groups = {
    "C" -> vec![1, 0],
    "B" -> vec![0]
}
Шаг 2. Вычисление моды для каждой группы
for (x_val, y_list) in groups {
    let mut freq: HashMap<i32, usize> = HashMap::new();
    for &y in &y_list {
        *freq.entry(y).or_insert(0) += 1;
    }
    let max_freq = freq.values().cloned().max().unwrap_or(0);
    let mut candidates: Vec<i32> = freq
        .into_iter()
        .filter(|&(_, f)| f == max_freq)
        .map(|(y, _)| y)
        .collect();
    candidates.sort();
    self.mode.insert(x_val, candidates[0]);
}
    • Проходим по каждой паре (x_val, y_list) из groups. Обратите внимание: groups потребляется (перемещается) в цикле, так как мы пишем for (x_val, y_list) in groups — это забирает владение элементами (это нормально, они больше не нужны).
    • Внутри считаем частоту каждого y в y_list:
let mut freq: HashMap<i32, usize> = HashMap::new();
for &y in &y_list {
    *freq.entry(y).or_insert(0) += 1;
}
Здесь &y_list — ссылка на вектор, for &y in &y_list итерирует по ссылкам на элементы и разыменовывает их (получается y: i32). Для каждого y увеличиваем счётчик.
    • let max_freq = freq.values().cloned().max().unwrap_or(0);
        ◦ freq.values() — итератор по ссылкам на счётчики.
        ◦ .cloned() — превращает &usize в usize (копирование).
        ◦ .max() — возвращает Option<usize>, максимальное значение.
        ◦ .unwrap_or(0) — если словарь пуст (не должно быть), то принимаем 0.
    • Далее собираем все y, у которых частота равна max_freq:
let mut candidates: Vec<i32> = freq
    .into_iter()
    .filter(|&(_, f)| f == max_freq)
    .map(|(y, _)| y)
    .collect();
        ◦ freq.into_iter() — поглощает словарь и возвращает итератор по парам (y, частота).
        ◦ .filter(...) — оставляет только те, где частота == максимальной.
        ◦ .map(|(y, _)| y) — из пары берём только y.
        ◦ .collect() — собирает в вектор candidates.
    • candidates.sort(); — сортируем вектор по возрастанию.
    • self.mode.insert(x_val, candidates[0]); — сохраняем в словарь модели наименьшее значение из кандидатов (правило разрешения ничьи). Для "C" кандидаты: [1,0] после сортировки [0,1], берём 0.
Вычисление завершено. Для нашего примера self.mode будет: {"C": 0, "B": 0}.
Метод predict
fn predict(&self, x_test: &[&str]) -> Vec<Option<i32>> {
    x_test.iter()
        .map(|x| self.mode.get(*x).cloned())
        .collect()
}
    • Принимает срез тестовых значений X (&[&str]).
    • x_test.iter() — итератор по каждому &str.
    • .map(|x| self.mode.get(*x).cloned()) — для каждого x:
        ◦ self.mode.get(*x) — ищет ключ в словаре. *x разыменовывает &&str до &str, который автоматически преобразуется в &String? В Rust HashMap<String, i32> имеет метод get(&Q) где Q: Borrow<str>. Поскольку &str реализует Borrow<str>, передавать &str можно напрямую. *x имеет тип &str, что подходит. Метод get возвращает Option<&i32> (ссылку на значение, если ключ есть).
        ◦ .cloned() — превращает Option<&i32> в Option<i32>, копируя число (для i32 копирование дешёвое). Если ключа нет, остаётся None.
    • .collect() — собирает итератор в Vec<Option<i32>>.
Таким образом, для x_test = ["C"] результат будет vec![Some(0)], что соответствует ожиданию задачи.
Функция train_and_predict
fn train_and_predict(x_train: &[&str], y_train: &[i32], x_test: &[&str]) -> Vec<Option<i32>> {
    let mut model = SimpleModel::new();
    model.fit(x_train, y_train);
    model.predict(x_test)
}
    • Создаёт модель, обучает её и возвращает предсказания для тестовых данных. Эта функция является обёрткой, требуемой в задании.
Пример в main
fn main() {
    let x_train = ["C", "C", "B"];
    let y_train = [1, 0, 0];
    let x_test = ["C"];
    let result = train_and_predict(&x_train, &y_train, &x_test);
    println!("{:?}", result); // [Some(0)]
}
    • Демонстрация работы. Выведет [Some(0)].
Ключевые особенности Rust, использованные в коде
    • Владение и заимствование — параметры &[&str] позволяют не копировать входные данные.
    • Итераторы и адаптеры — iter(), zip(), map(), collect() — декларативный стиль обработки коллекций.
    • Entry API — эффективная работа со словарём: entry(key).or_insert_with(...).
    • Pattern matching и разыменование — for &y in &y_list автоматически разыменовывает ссылки.
    • Option и обработка отсутствия значения — get() возвращает Option, который мы преобразуем в Option<i32>.
Более сложная модель
Общая идея
Модель запоминает наиболее часто встречающееся значение y (моду) для каждого уникального x. При предсказании для нового x возвращается соответствующая мода (или None, если такой x не встречался в обучении). Если для одного x есть несколько y с одинаковой максимальной частотой — выбирается последний встреченный.
Пояснения
    • fit собирает все значения y для каждого x в вектор, затем для каждого вектора подсчитывает частоты и определяет моду, обновляя кандидата при *count >= max_count. Это гарантирует выбор последнего значения при равных частотах .
    • predict возвращает Vec<Option<i32>>, где отсутствующим ключам соответствует None.
    • train_and_predict создаёт модель, обучает её и возвращает предсказания.
use std::collections::HashMap;

#[derive(Debug)]
struct SimpleModel {
    mode: HashMap<String, i32>,
}

impl SimpleModel {
    fn new() -> Self {
        SimpleModel {
            mode: HashMap::new(),
        }
    }

    fn fit(&mut self, X: Vec<String>, y: Vec<i32>) {
        // Собираем все значения y для каждого уникального X (в порядке поступления)
        let mut buckets: HashMap<String, Vec<i32>> = HashMap::new();
        for (x_val, y_val) in X.iter().zip(y.iter()) {
            buckets.entry(x_val.clone()).or_insert_with(Vec::new).push(*y_val);
        }

        // Для каждого X вычисляем моду (наиболее частое y)
        for (x_val, y_vals) in buckets {
            let mut counts: HashMap<i32, usize> = HashMap::new();
            let mut mode_y = None;
            let mut max_count = 0;

            for y in y_vals {
                let count = counts.entry(y).or_insert(0);
                *count += 1;
                // При равенстве частот обновляем – берём последний встреченный
                if *count >= max_count {
                    max_count = *count;
                    mode_y = Some(y);
                }
            }
            if let Some(mode) = mode_y {
                self.mode.insert(x_val, mode);
            }
        }
    }

    fn predict(&self, X: Vec<String>) -> Vec<Option<i32>> {
        X.iter().map(|x| self.mode.get(x).copied()).collect()
    }
}

fn train_and_predict(X_train: Vec<String>, y_train: Vec<i32>, X_test: Vec<String>) -> Vec<Option<i32>> {
    let mut model = SimpleModel::new();
    model.fit(X_train, y_train);
    model.predict(X_test)
}

fn main() {
    // Пример из условия
    let X_train = vec!["C".to_string(), "C".to_string(), "B".to_string()];
    let y_train = vec![1, 0, 0];
    let X_test = vec!["C".to_string()];

    let predictions = train_and_predict(X_train, y_train, X_test);
    println!("{:?}", predictions); // Выведет [Some(0)]
}
Импорты и объявление структуры
use std::collections::HashMap;

#[derive(Debug)]
struct SimpleModel {
    mode: HashMap<String, i32>,
}
    • HashMap из стандартной библиотеки .
    • #[derive(Debug)] позволяет выводить структуру в отладочных целях (например, через println!("{:?}", model)).
    • SimpleModel содержит поле mode — отображение из строки x в моду (целое число).
Реализация методов
fn new() -> Self
fn new() -> Self {
    SimpleModel {
        mode: HashMap::new(),
    }
}
Конструктор, создающий пустую модель.
Метод fit
fn fit(&mut self, X: Vec<String>, y: Vec<i32>) {
    // 1. Группируем все значения y по x (в порядке поступления)
    let mut buckets: HashMap<String, Vec<i32>> = HashMap::new();
    for (x_val, y_val) in X.iter().zip(y.iter()) {
        buckets.entry(x_val.clone()).or_insert_with(Vec::new).push(*y_val);
    }

    // 2. Для каждого x вычисляем моду
    for (x_val, y_vals) in buckets {
        let mut counts: HashMap<i32, usize> = HashMap::new();
        let mut mode_y = None;
        let mut max_count = 0;

        for y in y_vals {
            let count = counts.entry(y).or_insert(0);
            *count += 1;
            // Если частота >= максимальной — обновляем моду (берём последний при равенстве)
            if *count >= max_count {
                max_count = *count;
                mode_y = Some(y);
            }
        }
        if let Some(mode) = mode_y {
            self.mode.insert(x_val, mode);
        }
    }
}
Шаг 1: группировка значений y по ключам x
    • Создаём пустую хеш-таблицу buckets, где ключ — String (копия x_val), значение — вектор i32.
    • X.iter().zip(y.iter()) — параллельный итератор по X и y. Каждая итерация даёт пару (&String, &i32).
    • buckets.entry(x_val.clone()) — получает «вход» для ключа. Если ключа нет, то or_insert_with(Vec::new) создаёт новый пустой вектор. Затем .push(*y_val) добавляет значение y в этот вектор.
Шаг 2: вычисление моды для каждого x
    • Перебираем каждую пару (x_val, y_vals) из buckets.
    • Внутренняя хеш-таблица counts подсчитывает, сколько раз встретилось каждое y в векторе y_vals.
    • Перебираем y из y_vals (сохраняя порядок, в котором они были добавлены). Для каждого:
        ◦ Увеличиваем счётчик в counts.
        ◦ Если текущий счётчик >= max_count, то обновляем max_count и запоминаем mode_y = Some(y). Важно: условие >=, а не >. Это означает, что при равенстве частот более позднее (последнее) значение затирает предыдущее. 
    • После цикла, если mode_y определена, вставляем в self.mode пару (x_val, mode).
Метод predict
fn predict(&self, X: Vec<String>) -> Vec<Option<i32>> {
    X.iter().map(|x| self.mode.get(x).copied()).collect()
}
    • Принимает вектор X_test (строки).
    • Для каждого элемента x вызывает self.mode.get(x), который возвращает Option<&i32>.
    • copied() превращает Option<&i32> в Option<i32> (копирует значение, если оно есть).
    • Результат собирается в Vec<Option<i32>>. Если ключа нет в self.mode, будет None.
Функция train_and_predict
fn train_and_predict(X_train: Vec<String>, y_train: Vec<i32>, X_test: Vec<String>) -> Vec<Option<i32>> {
    let mut model = SimpleModel::new();
    model.fit(X_train, y_train);
    model.predict(X_test)
}
Просто создаёт модель, обучает её на обучающих данных и возвращает предсказания для тестовых.
Функция main (пример использования)
fn main() {
    let X_train = vec!["C".to_string(), "C".to_string(), "B".to_string()];
    let y_train = vec![1, 0, 0];
    let X_test = vec!["C".to_string()];

    let predictions = train_and_predict(X_train, y_train, X_test);
    println!("{:?}", predictions); // [Some(0)]
}
    • Обучающие данные: для "C" значения [1, 0], для "B" [0].
    • Подсчёт: для "C": 1 встречается 1 раз, 0 встречается 1 раз. Порядок добавления в y_vals: [1, 0]. При проверке:
        ◦ Первый y=1: count=1, max_count=0, условие >= выполняется → mode=1, max_count=1.
        ◦ Второй y=0: count=1, max_count=1, условие >= (1 >= 1) выполняется → mode=0, max_count=1.
    • Итог: мода для "C" = 0. "B" имеет единственное значение 0 → мода 0.
    • Предсказание для "C" → Some(0).
Примеры обучения модели в Burn
Подробнее зесь https://dzen.ru/a/aJoS-fmlBQr37M97 
Наиболее простой вариант обучения модели содержится в модуле Context ML.
Обучение модели строиться везде по разному. Например в Burn обучение модели строится вокруг Autodiff-декоратора, который добавляет автоматическое дифференцирование к любому бэкенду. Это позволяет писать один и тот же код модели, а обучение проводить на CPU или GPU, просто меняя тип бэкенда.
Основные компоненты обучения любой модели:
    • Autodiff<B> — бэкенд с градиентами.
    • Оптимизатор (Adam, SGD и др.).
    • Функция потерь (CrossEntropy, MSE и т.д.).
    • Даталоадеры (Burn поддерживает свои или из burn-dataset).
    • Learner — high-level API для запуска обучения (с метриками, чекпоинтами, логами).
1. Autodiff — бэкенд с градиентами
Простыми словами: Autodiff (автоматическое дифференцирование) — это технология, которая автоматически считает производные (градиенты) функций. Градиенты показывают, как сильно нужно изменить параметры модели, чтобы уменьшить ошибку.
Пример: Представьте, что вы учитесь кататься на велосипеде. Autodiff — это как тренер, который говорит: «Ты слишком сильно наклонился влево, поправь руль вправо на 10 градусов, чтобы не упасть». Здесь «на 10 градусов» — это градиент.
2. Оптимизатор (Adam, SGD и др.)
Простыми словами: Оптимизатор — это алгоритм, который использует градиенты, чтобы обновлять параметры модели. Он решает, насколько сильно и в каком направлении изменить параметры, чтобы модель быстрее и точнее училась.
Примеры:
        ◦ SGD (Стохастический градиентный спуск): Как спускаться с горы маленькими шагами, каждый раз выбирая направление вниз.
        ◦ Adam: Более умный вариант SGD, который подстраивает размер шага для каждого параметра отдельно.
3. Функция потерь (CrossEntropy, MSE и т.д.)
Простыми словами: Функция потерь — это способ измерить, насколько плохо ваша модель работает. Чем меньше значение функции потерь, тем лучше модель.
Примеры:
        ◦ MSE (Среднеквадратичная ошибка): Используется для задач предсказания чисел (например, цена на дом). Считает, насколько сильно предсказания модели отличаются от реальных значений.
        ◦ CrossEntropy: Используется для задач классификации (например, распознавание кошек и собак). Считает, насколько уверенно модель ошиблась.
4. Даталоадеры (Burn поддерживает свои или из burn-dataset)
Простыми словами: Даталоадер — это инструмент, который загружает данные для обучения модели порциями (батчами). Он отвечает за то, чтобы данные подавались в модель быстро и эффективно.
Пример: Представьте, что вы учитесь готовить блюдо по рецепту. Даталоадер — это как помощник, который подаёт вам ингредиенты небольшими порциями, чтобы вы не перегрузились и могли готовить без остановки.
5. Learner — high-level API для запуска обучения (с метриками, чекпоинтами, логами)
Простыми словами: Learner — это готовый инструмент, который упрощает процесс обучения модели. Он автоматически следит за метриками (например, точность), сохраняет промежуточные результаты (чекпоинты) и ведёт логи (журнал обучения).
Пример: Learner — это как автопилот в самолёте: вы задаёте маршрут (параметры обучения), а он сам следит за высотой, скоростью и записывает все данные о полёте.
Полный пример: обучение модели на MNIST
Это адаптированный пример из официальных examples Burn (mnist-training). Предполагаем, что у вас скачаны данные MNIST в папку data/mnist.
use burn::{
    config::Config,
    data::{
        dataset::{Dataset, MnistDataset},
        dataloader::{batcher::Batcher, DataLoaderBuilder},
    },
    module::Module,
    nn::{
        loss::CrossEntropyLoss,
        Linear, LinearConfig, Relu,
    },
    optim::AdamConfig,
    prelude::*,
    tensor::backend::Autodiff,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(images);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }

    // Классификация: добавляем log_softmax для CrossEntropyLoss
    pub fn forward_classification(&self, images: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Loss<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLoss::new(None, &self.devices());
        loss.forward(output, targets)
    }
}

#[derive(Config)]
pub struct ModelConfig {
    #[config(default = 10)]
    num_classes: usize,
    #[config(default = 128)]
    hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(28 * 28, self.hidden_size).init(device),
            activation: Relu::new(),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
        }
    }
}

// Batcher для MNIST
#[derive(Clone)]
pub struct MnistBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MnistItem> for MnistBatcher<B> {
    type Batch = MnistBatch<B>;

    fn batch(&self, items: Vec<MnistItem>) -> Self::Batch {
        let images = items
            .iter()
            .map(|item| item.image.clone())
            .collect::<Vec<_>>()
            .concat()
            .reshape([items.len(), 1, 28, 28])
            .to_device(&self.device);

        let targets = items
            .iter()
            .map(|item| item.label as i64)
            .collect::<Vec<_>>()
            .into_tensor()
            .reshape([items.len(), 1])
            .to_device(&self.device);

        MnistBatch { images: images.squeeze(1), targets }
    }
}

fn main() {
    // Выбор бэкенда
    type Backend = burn::backend::NdArray; // Для CPU
    // type Backend = burn::backend::Wgpu; // Для GPU (нужно указать f32/i32)
    type TrainBackend = Autodiff<Backend>;

    let device = TrainBackend::Device::default();

    // Даталоадеры
    let batcher_train = MnistBatcher::<TrainBackend>::new(device.clone());
    let batcher_valid = MnistBatcher::<Backend>::new(device.clone()); // Без autodiff для валидации

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(32)
        .shuffle(42)
        .build(MnistDataset::train());

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(256)
        .build(MnistDataset::test());

    // Конфиг
    let config = ModelConfig::new();
    let optimizer = AdamConfig::new().init();
    let model = config.init::<TrainBackend>(&device);

    // Learner
    let learner = LearnerBuilder::new("mnist-out")
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .devices(vec![device])
        .num_epochs(10)
        .build(model, optimizer, CrossEntropyLoss::new(None, &[device]));

    let trained_model = learner.learn(dataloader_train, dataloader_valid);

    // Сохранение
    trained_model.save_file("mnist-out/model", &burn::record::FullRecorder::new()).expect("Save failed");
}
Полные примеры всегда доступны в репозитории Burn: https://github.com/tracel-ai/burn/tree/main/examples
Веса модели
Есть важный нюанс: публично доступных предобученных моделей НЕ СУЩЕСТВУЕТ. Если речь идет о узкоспециализированной задаче. Давайте разберём все аспекты:
🔍 Что такое "веса модели" в контексте ML?
Компонент	Аналогия	В проекте
Архитектура модели	Каркас здания (стены, перекрытия)	nn::Sequential с линейными слоями в inference.rs
Веса модели	Отделка, мебель, проводка	Файл rustfix_v2.pt — числовые параметры, обученные на данных
Инференс	Жильцы используют готовое здание	model.forward(&input) — применение весов к новому коду
Веса — это бинарный файл (обычно .pt для PyTorch), содержащий миллионы чисел — результат обучения на датасете.
Проблема	Объяснение
Нет публичных моделей	Никто не публиковал модели специально для конкретной задачи (в отличие от, например, код-генерации)
Специфичность задачи	Модель должна понимать AST Rust, borrow checker, lifetime system — это требует кастомной архитектуры
Юридические ограничения	Датасеты из реального кода часто содержат проприетарный код — их нельзя публиковать свободно
✅ Практические пути получения весов
Вариант 1: Использовать моковую модель (рекомендуется для старта)
# Создайте заглушку-веса (0 байт — модель всё равно не используется в моковом режиме)
touch weights/rustfix_v2.pt

# Или создайте минимальный валидный файл для тестов
python3 << 'EOF'
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 3)
)
torch.save(model.state_dict(), 'weights/rustfix_v2.pt')
print("✅ Created mock weights (1.2 MB)")
EOF
Плюсы:
✅ Мгновенная настройка
✅ Не требует GPU/обучения
✅ Демонстрирует архитектуру проекта
Минусы:
❌ Не даёт реальных предсказаний - только эвристики
Вариант 2: Обучить модель самостоятельно (реалистичный путь)
Шаг 1: Подготовьте датасет
# ml/training/generate_dataset.py
import json
from pathlib import Path
import re

def extract_error_fix_pairs(repo_path):
    """Извлекает пары (ошибка → исправление) из истории коммитов"""
    pairs = []
    
    # Пример: найти коммиты с исправлением "use of moved value"
    pattern = r'let\s+(\w+)\s*=\s*(\w+);.*?println!\(.*?\1\)'
    
    for rs_file in Path(repo_path).rglob("*.rs"):
        content = rs_file.read_text()
        if re.search(pattern, content, re.DOTALL):
            # Найти исправленную версию в истории git
            fixed = content.replace(f"println!(\"{{}}\", {pattern})", 
                                   f"println!(\"{{}}\", {pattern}.clone())")
            pairs.append({
                "file": str(rs_file),
                "original": content,
                "fixed": fixed,
                "error_type": "move_semantics",
                "confidence": 0.95
            })
    
    return pairs

# Соберите 10к+ примеров из открытых репозиториев
repos = [
    "https://github.com/rust-lang/rust",
    "https://github.com/tokio-rs/tokio",
    "https://github.com/serde-rs/serde",
    # ... ещё 50 популярных репозиториев
]

dataset = []
for repo in repos:
    dataset.extend(extract_error_fix_pairs(clone_repo(repo)))

with open("datasets/rust_errors_v2.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

print(f"✅ Собрано {len(dataset)} примеров")
Шаг 2: Обучите модель
# ml/training/train.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

class RustErrorDataset(Dataset):
    def __init__(self, jsonl_path):
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.items = [json.loads(line) for line in open(jsonl_path)]
    
    def __getitem__(self, idx):
        item = self.items[idx]
        # Токенизируем ошибочный код
        tokens = self.tokenizer(item["original"], return_tensors="pt", max_length=512, truncation=True)
        # Метка: класс ошибки (0=move, 1=lifetime, 2=trait)
        label = {"move_semantics": 0, "lifetime": 1, "trait": 2}[item["error_type"]]
        return tokens["input_ids"].squeeze(), torch.tensor(label)

# Архитектура: CodeBERT + классификатор
from transformers import RobertaModel

class RustFixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3)  # 3 класса ошибок
        )
    
    def forward(self, input_ids):
        features = self.encoder(input_ids).last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(features)

# Обучение
model = RustFixModel()
dataset = RustErrorDataset("datasets/rust_errors_v2.jsonl")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
for epoch in range(10):
    for input_ids, labels in loader:
        outputs = model(input_ids)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Сохранение весов
torch.save(model.state_dict(), "weights/rustfix_v2.pt")
print("✅ Модель обучена и сохранена")
Требования:
    • ⏱ Время: 8-24 часа на среднем GPU (RTX 3090)
    • 💾 Данные: 10к+ примеров ошибок/исправлений
    • 💰 Стоимость: ~$15 на Google Colab Pro для обучения
Вариант 3: Интеграция с существующими инструментами (гибридный подход)
Вместо кастомной модели используйте официальные инструменты Rust :
// src/ml/hybrid.rs
pub fn analyze_with_rustc(
    path: &Path,
) -> Result<Vec<AnalysisError>, Exn<AnalysisError>> {
    // Запускаем rustc с выводом в JSON
    let output = std::process::Command::new("rustc")
        .args(&[
            "--crate-type", "lib",
            "--error-format=json",
            "--emit=metadata",
            path.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| Exn::new(AnalysisError::infrastructure(&format!(
            "rustc invocation failed: {}",
            e
        ))))?;

    if !output.status.success() {
        // Парсим диагностические сообщения
        let diagnostics: Vec<Diagnostic> = serde_json::from_slice(&output.stderr)
            .map_err(|e| Exn::new(AnalysisError::infrastructure(&format!(
                "Failed to parse rustc diagnostics: {}",
                e
            ))))?;

        let errors = diagnostics
            .into_iter()
            .filter(|d| d.level == "error")
            .map(|d| {
                AnalysisError {
                    kind: AnalysisKind::CompilerDiagnostic,
                    message: d.message,
                    suggestion: d.suggestions.first().cloned(),
                    line: Some(d.line_start as u32),
                    column: Some(d.column_start as u32),
                    confidence: 1.0, // rustc всегда точен
                }
            })
            .collect();

        return Ok(errors);
    }

    Ok(vec![])
}
Преимущества гибридного подхода:
✅ 100% точность для компиляторных ошибок
✅ Не требует обучения модели
✅ Работает "из коробки" с любым кодом
✅ exn добавляет контекст к диагностике rustc

