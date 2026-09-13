**[Т]{.underline}[е]{.underline}[о]{.underline}[р]{.underline}[и]{.underline}[я]{.underline}**

# [Обучение с учителем]{.underline}

[Обучение с учителем включает в себя обучение модели на наборе данных,
содержащем как входные признаки, так и соответствующие правильные
выходные данные (метки). Цель состоит в том, чтобы предсказать выходные
данные для новых, ранее не встречавшихся входных данных на основе
изученной функции. Существует два основных типа обучения с учителем:
Классификация: цель состоит в том, чтобы предсказать метку класса,
которая представляет собой выбор из предопределенного списка
возможностей. Например, представьте, что вы пытаетесь отсортировать
электронные письма на «спам» или «не спам». Или когда ваш телефон
распознает ваше лицо для разблокировки, он использует классификацию,
чтобы сказать: «Да, это вы!» Регрессия: цель состоит в том, чтобы
предсказать величину. Например, предсказать цену дома на основе
различных характеристик, таких как его размер и местоположение, или
определить, сколько стоит автомобиль, исходя из его марки, модели и
возраста. Это как играть в «Цена правильная». По мере изучения различных
алгоритмов в этом курсе мы будем говорить о том, являются ли они больше
похожими на обучение с учителем или на обучение без
учителя.]{.underline}

# [Обучение без учителя ]{.underline}

[Включает в себя обучение модели на неразмеченном наборе данных. Цель
--- обнаружить скрытые закономерности или структуры в данных. Это как
дать ребенку большую коробку с кубиками LEGO без инструкции. Цель ---
увидеть, какие интересные структуры или закономерности он сам найдет.
Примеры включают задачи кластеризации и уменьшения размерности:
Кластеризация: Цель --- сгруппировать похожие экземпляры в кластеры. Это
как организовать свои игрушки в группы, не объясняя, как это сделать.
Например, магазин может разделить покупателей на разные типы, чтобы
определить наилучший способ рекламы для каждого типа. Или в социальных
сетях это помогает определить, кто в какой группе друзей. Уменьшение
размерности: Метод, который упрощает сложность многомерных данных,
сохраняя при этом их основные характеристики. Представьте, что у вас
есть огромный рюкзак, полный школьных принадлежностей, но вы хотите
сделать его легче. Уменьшение размерности --- это как определить, что
важно, а что нет, чтобы вы носили только то, что вам действительно
нужно. Это помогает справляться с избытком информации --- например,
когда вы пытаетесь упростить очень сложную картинку, чтобы её было легче
понять. В ходе изучения различных алгоритмов в этом курсе мы будем
обсуждать, относятся ли они больше к контролируемому или
неконтролируемому обучению.]{.underline}

## [Алгоритм KNN]{.underline}

[Метод KNN не имеет явной фазы обучения. Он просто сохраняет набор
данных. Только на этапе прогнозирования он вычисляет расстояние между
новой точкой и остальными.]{.underline}

[Алгоритм KNN (K-ближайших соседей) --- это алгоритм обучения с
учителем, используемый как для классификации, так и для регрессии. Он
прост, но мощен и часто используется в качестве базового алгоритма для
более сложных моделей. Алгоритм KNN находит сходство признаков,
используя метрики расстояния для определения ближайших соседей к точке.
Например, он может сортировать вина, сравнивая их с ближайшими соседями,
например, группируя вина со схожими вкусовыми качествами. Это как иметь
гида, который помогает вам найти похожие вина на основе их наиболее
общих характеристик. ]{.underline}

[Идея KNN довольно проста: Для заданной тестовой точки алгоритм
вычисляет расстояние между этой точкой и каждой другой точкой в
​​обучающем наборе данных. Расстояние может быть вычислено различными
способами, например, евклидовым или манхэттенским расстоянием. Поиск
ближайших соседей: Затем алгоритм выбирает k точек в обучающих данных,
которые находятся ближе всего к тестовой точке. k --- это определяемая
пользователем константа, которая может быть любым целым числом.
Предсказание: Для классификации алгоритм присваивает тестовой точке
метку наиболее часто встречающегося класса среди k ближайших соседей.
Для регрессии он присваивает среднее значение меток «k» ближайших
соседей. KNN --- это алгоритм ленивого обучения, то есть он фактически
не обучается модели. Вместо этого он запоминает обучающие примеры и
использует их для предсказания (в следующем уроке будет рассмотрена фаза
обучения). Это делает KNN вычислительно затратным на этапе тестирования.
Несмотря на свою простоту, KNN может показать удивительно хорошие
результаты. Однако он чувствителен к выбору k и типу метрики расстояния,
и плохо работает с многомерными данными из-за «проклятия
размерности».]{.underline}

[`fn distances_option_calculator(X: &[i32], y: &[i32], distance_type: &str) -> f64 ``{`]{.underline}

[`    ``match distance_type ``{`]{.underline}

[`        ``"euclidian" => ``{`]{.underline}

[`            ``let sum_sq: i32 = X.iter().zip(y).map(|(&a, &b)| (a - b).pow(2)).sum();`]{.underline}

[`            ``(sum_sq as f64).sqrt()`]{.underline}

[`        ``}`]{.underline}

[`        ``"manhattan" => ``{`]{.underline}

[`            ``let sum_abs: i32 = X.iter().zip(y).map(|(&a, &b)| (a - b).abs()).sum();`]{.underline}

[`            ``sum_abs as f64`]{.underline}

[`        ``}`]{.underline}

[`        ``"hamming" => ``{`]{.underline}

[`            ``let diff_count: i32 = X.iter().zip(y).filter(|(&a, &b)| a != b).count() as i32;`]{.underline}

[`            ``diff_count as f64`]{.underline}

[`        ``}`]{.underline}

[`        ``_ => panic!("distance_type must be 'euclidian', 'manhattan', or 'hamming'"),`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[`use std::f64;`]{.underline}

[`struct KNN<T> ``{`]{.underline}

[`    ``points: Vec<Vec<f64>>,`]{.underline}

[`    ``labels: Vec<T>,`]{.underline}

[`    ``k: usize,`]{.underline}

[`}`]{.underline}

[`impl<T> KNN<T> ``{`]{.underline}

[`    ``// ``К``о``н``с``т``р``у``к``т``о``р`]{.underline}

[`    ``fn new(k: usize) -> Self ``{`]{.underline}

[`        ``KNN ``{`]{.underline}

[`            ``points: Vec::new(),`]{.underline}

[`            ``labels: Vec::new(),`]{.underline}

[`            ``k,`]{.underline}

[`        ``}`]{.underline}

[`    ``}`]{.underline}

[`    ``// ``Е``в``к``л``и``д``о``в``о`` ``р``а``с``с``т``о``я``н``и``е`` ``м``е``ж``д``у`` ``д``в``у``м``я`` ``т``о``ч``к``а``м``и`` (``в``е``к``т``о``р``а``м``и`` ``о``д``и``н``а``к``о``в``о``й`` ``д``л``и``н``ы``)`]{.underline}

[`    ``fn distance(&self, point_a: &[f64], point_b: &[f64]) -> f64 ``{`]{.underline}

[`        ``let sum_sq: f64 = point_a`]{.underline}

[`            ``.iter()`]{.underline}

[`            ``.zip(point_b)`]{.underline}

[`            ``.map(|(&a, &b)| (a - b).powi(2))`]{.underline}

[`            ``.sum();`]{.underline}

[`        ``sum_sq.sqrt()`]{.underline}

[`    ``}`]{.underline}

[`    ``// ``С``о``х``р``а``н``я``е``т`` ``о``б``у``ч``а``ю``щ``и``е`` ``т``о``ч``к``и`` ``и`` ``м``е``т``к``и`]{.underline}

[`    ``fn fit(&mut self, X_train: Vec<Vec<f64>>, y_train: Vec<T>) ``{`]{.underline}

[`        ``self.points = X_train;`]{.underline}

[`        ``self.labels = y_train;`]{.underline}

[`    ``}`]{.underline}

[`    ``// ``З``а``г``л``у``ш``к``а`` ``д``л``я`` predict (``к``а``к`` ``в`` ``и``с``х``о``д``н``о``м`` ``к``о``д``е``)`]{.underline}

[`    ``fn predict(&self, _X_test: Vec<Vec<f64>>) -> Vec<T> ``{`]{.underline}

[`        ``todo!("``М``е``т``о``д`` predict ``п``о``к``а`` ``н``е`` ``р``е``а``л``и``з``о``в``а``н``")`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[`fn main() ``{`]{.underline}

[`    ``// ``П``р``и``м``е``р`` ``и``с``п``о``л``ь``з``о``в``а``н``и``я`` `]{.underline}

[`    ``let mut knn = KNN::new(3);`]{.underline}

[`    ``// ``О``б``у``ч``а``ю``щ``и``е`` ``т``о``ч``к``и``: (1,2), (2,3), (3,5)`]{.underline}

[`    ``let X_train = vec![`]{.underline}

[`        ``vec![1.0, 2.0],`]{.underline}

[`        ``vec![2.0, 3.0],`]{.underline}

[`        ``vec![3.0, 5.0],`]{.underline}

[`    ``];`]{.underline}

[`    ``// ``М``е``т``к``и``: A, B, A`]{.underline}

[`    ``let y_train = vec!["A", "B", "A"];`]{.underline}

[`    ``// ``В``ы``з``о``в`` fit`]{.underline}

[`    ``knn.fit(X_train, y_train);`]{.underline}

[`    ``println!("Fit method pass");`]{.underline}

[`    ``// ``П``р``о``в``е``р``к``а`` distance ``н``а`` ``д``в``у``х`` ``т``о``ч``к``а``х`]{.underline}

[`    ``let dist = knn.distance(&[1.0, 2.0], &[2.0, 3.0]);`]{.underline}

[`    ``println!("Distance method pass (distance = ``{``}``)", dist);`]{.underline}

[`}`]{.underline}

[`use std::collections::HashMap;`]{.underline}

[`struct KNN ``{`]{.underline}

[`    ``points: Vec<Vec<f64>>,`]{.underline}

[`    ``labels: Vec<String>,`]{.underline}

[`    ``k: usize,`]{.underline}

[`}`]{.underline}

[`impl KNN ``{`]{.underline}

[`    ``fn new(k: usize) -> Self ``{`]{.underline}

[`        ``KNN ``{`]{.underline}

[`            ``points: Vec::new(),`]{.underline}

[`            ``labels: Vec::new(),`]{.underline}

[`            ``k,`]{.underline}

[`        ``}`]{.underline}

[`    ``}`]{.underline}

[`    ``// ``Е``в``к``л``и``д``о``в``о`` ``р``а``с``с``т``о``я``н``и``е`` ``м``е``ж``д``у`` ``д``в``у``м``я`` ``т``о``ч``к``а``м``и`` (``в``е``к``т``о``р``а``м``и`` ``о``д``и``н``а``к``о``в``о``й`` ``д``л``и``н``ы``)`]{.underline}

[`    ``fn distance(&self, point_a: &[f64], point_b: &[f64]) -> f64 ``{`]{.underline}

[`        ``let sum_sq: f64 = point_a`]{.underline}

[`            ``.iter()`]{.underline}

[`            ``.zip(point_b)`]{.underline}

[`            ``.map(|(&a, &b)| (a - b).powi(2))`]{.underline}

[`            ``.sum();`]{.underline}

[`        ``sum_sq.sqrt()`]{.underline}

[`    ``}`]{.underline}

[`    ``// ``З``а``п``о``м``и``н``а``е``т`` ``о``б``у``ч``а``ю``щ``и``е`` ``т``о``ч``к``и`` ``и`` ``м``е``т``к``и`]{.underline}

[`    ``fn fit(&mut self, X_train: Vec<Vec<f64>>, y_train: Vec<String>) ``{`]{.underline}

[`        ``self.points = X_train;`]{.underline}

[`        ``self.labels = y_train;`]{.underline}

[`    ``}`]{.underline}

[`    ``// ``П``р``е``д``с``к``а``з``ы``в``а``е``т`` ``м``е``т``к``и`` ``д``л``я`` ``т``е``с``т``о``в``ы``х`` ``т``о``ч``е``к`]{.underline}

[`    ``fn predict(&self, X_test: Vec<Vec<f64>>) -> Vec<String> ``{`]{.underline}

[`        ``let mut predictions = Vec::new();`]{.underline}

[`        ``for test_point in X_test ``{`]{.underline}

[`            ``// ``В``ы``ч``и``с``л``я``е``м`` ``р``а``с``с``т``о``я``н``и``я`` ``д``о`` ``в``с``е``х`` ``о``б``у``ч``а``ю``щ``и``х`` ``т``о``ч``е``к`` ``с`` ``с``о``х``р``а``н``е``н``и``е``м`` ``и``н``д``е``к``с``о``в`]{.underline}

[`            ``let mut distances: Vec<(usize, f64)> = self.points`]{.underline}

[`                ``.iter()`]{.underline}

[`                ``.enumerate()`]{.underline}

[`                ``.map(|(i, train_point)| (i, self.distance(&test_point, train_point)))`]{.underline}

[`                ``.collect();`]{.underline}

[`            ``// ``С``о``р``т``и``р``у``е``м`` ``п``о`` ``в``о``з``р``а``с``т``а``н``и``ю`` ``р``а``с``с``т``о``я``н``и``я`]{.underline}

[`            ``distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());`]{.underline}

[`            ``// ``Б``е``р``ё``м`` ``и``н``д``е``к``с``ы`` k ``б``л``и``ж``а``й``ш``и``х`` ``с``о``с``е``д``е``й`]{.underline}

[`            ``let k_nearest_indices: Vec<usize> = distances`]{.underline}

[`                ``.iter()`]{.underline}

[`                ``.take(self.k)`]{.underline}

[`                ``.map(|&(idx, _)| idx)`]{.underline}

[`                ``.collect();`]{.underline}

[`            ``// ``П``о``л``у``ч``а``е``м`` ``м``е``т``к``и`` ``э``т``и``х`` ``с``о``с``е``д``е``й`]{.underline}

[`            ``let k_nearest_labels: Vec<&String> = k_nearest_indices`]{.underline}

[`                ``.iter()`]{.underline}

[`                ``.map(|&idx| &self.labels[idx])`]{.underline}

[`                ``.collect();`]{.underline}

[`            ``// ``П``о``д``с``ч``и``т``ы``в``а``е``м`` ``ч``а``с``т``о``т``у`` ``к``а``ж``д``о``й`` ``м``е``т``к``и`]{.underline}

[`            ``let mut freq = HashMap::new();`]{.underline}

[`            ``for label in &k_nearest_labels ``{`]{.underline}

[`                ``*freq.entry(*label).or_insert(0) += 1;`]{.underline}

[`            ``}`]{.underline}

[`            ``// ``Н``а``х``о``д``и``м`` ``м``е``т``к``у`` ``с`` ``м``а``к``с``и``м``а``л``ь``н``о``й`` ``ч``а``с``т``о``т``о``й`` (``п``р``и`` ``р``а``в``н``ы``х`` - ``б``е``р``ё``м`` ``п``е``р``в``у``ю`` ``п``о`` ``п``о``р``я``д``к``у``)`]{.underline}

[`            ``let mut best_label = k_nearest_labels[0];`]{.underline}

[`            ``let mut max_count = 0;`]{.underline}

[`            ``for label in k_nearest_labels ``{`]{.underline}

[`                ``let count = freq[label];`]{.underline}

[`                ``if count > max_count ``{`]{.underline}

[`                    ``max_count = count;`]{.underline}

[`                    ``best_label = label;`]{.underline}

[`                ``}`]{.underline}

[`            ``}`]{.underline}

[`            ``predictions.push(best_label.clone());`]{.underline}

[`        ``}`]{.underline}

[`        ``predictions`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[`// ``П``р``и``м``е``р`` ``и``с``п``о``л``ь``з``о``в``а``н``и``я`` `]{.underline}

[`fn main() ``{`]{.underline}

[`    ``let mut knn = KNN::new(3);`]{.underline}

[`    ``let X_train = vec![`]{.underline}

[`        ``vec![1.0, 5.0],`]{.underline}

[`        ``vec![5.0, 0.0],`]{.underline}

[`        ``vec![3.0, 2.0],`]{.underline}

[`        ``vec![8.0, 1.0],`]{.underline}

[`        ``vec![4.0, 4.0],`]{.underline}

[`        ``vec![7.0, 7.0],`]{.underline}

[`    ``];`]{.underline}

[`    ``let y_train = vec![`]{.underline}

[`        ``"A".to_string(),`]{.underline}

[`        ``"B".to_string(),`]{.underline}

[`        ``"A".to_string(),`]{.underline}

[`        ``"B".to_string(),`]{.underline}

[`        ``"C".to_string(),`]{.underline}

[`        ``"A".to_string(),`]{.underline}

[`    ``];`]{.underline}

[`    ``knn.fit(X_train, y_train);`]{.underline}

[`    ``let X_test = vec![vec![3.0, 3.0]];`]{.underline}

[`    ``let predictions = knn.predict(X_test);`]{.underline}

[`    ``println!("``{``:?``}``", predictions); // ``В``ы``в``е``д``е``т`` ["A"]`]{.underline}

[`}`]{.underline}

## [Введение в алгоритм K-средних ]{.underline}

[Алгоритм K-средних без учителя --- цель разбить n наблюдений на k
кластеров, в каждом из которых каждое наблюдение принадлежит кластеру с
ближайшим средним значением (центроидом), который служит прототипом
кластера. Допустим, у вас есть набор разных фруктов, и вы хотите
рассортировать их по корзинам в зависимости от типа, но у вас нет меток.
Алгоритм K-средних поможет вам сделать это, но с точками данных вместо
фруктов!]{.underline}

[Алгоритм использует итеративную процедуру для разделения набора данных
на k кластеров. ]{.underline}

[Инициализация центроидов: Сначала выберите k точек в качестве начальных
центроидов из набора данных. Эти точки могут быть выбраны случайным
образом или на основе определенной стратегии.]{.underline}

[Назначение кластеров: Для каждой точки в наборе данных найдите
ближайший центроид (используя меры расстояния, такие как евклидово
расстояние) и назначьте точку этому кластеру. ]{.underline}

[Обновление центроидов: После того, как все точки назначены кластерам,
пересчитайте центроиды, взяв среднее значение всех точек в каждом
кластере. Повторение шагов 2 и 3: Вышеописанные шаги повторяются до тех
пор, пока центроиды перестанут существенно изменяться. Это означает, что
алгоритм сошелся, и кластеры стали стабильными.]{.underline}

[`fn euclidian_distance(point_a: &[f64], point_b: &[f64]) -> f64 ``{`]{.underline}

[`    ``point_a.iter()`]{.underline}

[`        ``.zip(point_b)`]{.underline}

[`        ``.map(|(&a, &b)| (a - b).powi(2))`]{.underline}

[`        ``.sum::<f64>()`]{.underline}

[`        ``.sqrt()`]{.underline}

[`}`]{.underline}

[`fn wcss(points: &[Vec<f64>]) -> f64 ``{`]{.underline}

[`    ``if points.is_empty() ``{`]{.underline}

[`        ``return 0.0;`]{.underline}

[`    ``}`]{.underline}

[`    ``let n_points = points.len();`]{.underline}

[`    ``let n_dims = points[0].len();`]{.underline}

[`    ``// Compute centroid (mean of all points)`]{.underline}

[`    ``let mut centroid = vec![0.0; n_dims];`]{.underline}

[`    ``for point in points ``{`]{.underline}

[`        ``for i in 0..n_dims ``{`]{.underline}

[`            ``centroid[i] += point[i];`]{.underline}

[`        ``}`]{.underline}

[`    ``}`]{.underline}

[`    ``for i in 0..n_dims ``{`]{.underline}

[`        ``centroid[i] /= n_points as f64;`]{.underline}

[`    ``}`]{.underline}

[`    ``// Sum Euclidean distances from each point to centroid`]{.underline}

[`    ``let total_distance: f64 = points.iter()`]{.underline}

[`        ``.map(|point| euclidian_distance(point, &centroid))`]{.underline}

[`        ``.sum();`]{.underline}

[`    ``// Return average distance`]{.underline}

[`    ``total_distance / n_points as f64`]{.underline}

[`}`]{.underline}

[`// ``П``р``и``м``е``р`` ``и``с``п``о``л``ь``з``о``в``а``н``и``я`]{.underline}

[`fn main() ``{`]{.underline}

[`    ``let points = vec![`]{.underline}

[`        ``vec![1.0, 2.0, 3.0],`]{.underline}

[`        ``vec![4.0, 5.0, 6.0],`]{.underline}

[`    ``];`]{.underline}

[`    ``let result = wcss(&points);`]{.underline}

[`    ``println!("WCSS: ``{``}``", result);`]{.underline}

[`}`]{.underline}

[`fn euclidian_distance(point_a: &[f64], point_b: &[f64]) -> f64 ``{`]{.underline}

[`    ``point_a.iter()`]{.underline}

[`        ``.zip(point_b)`]{.underline}

[`        ``.map(|(&a, &b)| (a - b).powi(2))`]{.underline}

[`        ``.sum::<f64>()`]{.underline}

[`        ``.sqrt()`]{.underline}

[`}`]{.underline}

[`struct KMeans ``{`]{.underline}

[`    ``k: usize,`]{.underline}

[`    ``centroids: Vec<Vec<f64>>,`]{.underline}

[`}`]{.underline}

[`impl KMeans ``{`]{.underline}

[`    ``fn new(k: usize) -> Self ``{`]{.underline}

[`        ``KMeans ``{`]{.underline}

[`            ``k,`]{.underline}

[`            ``centroids: Vec::new(),`]{.underline}

[`        ``}`]{.underline}

[`    ``}`]{.underline}

[`    ``fn find_cluster_centroid(&self, cluster_points: &[Vec<f64>]) -> Option<Vec<f64>> ``{`]{.underline}

[`        ``if cluster_points.is_empty() ``{`]{.underline}

[`            ``return None;`]{.underline}

[`        ``}`]{.underline}

[`        ``let n_dims = cluster_points[0].len();`]{.underline}

[`        ``let n_points = cluster_points.len() as f64;`]{.underline}

[`        ``let mut centroid = vec![0.0; n_dims];`]{.underline}

[`        ``for point in cluster_points ``{`]{.underline}

[`            ``for i in 0..n_dims ``{`]{.underline}

[`                ``centroid[i] += point[i];`]{.underline}

[`            ``}`]{.underline}

[`        ``}`]{.underline}

[`        ``for i in 0..n_dims ``{`]{.underline}

[`            ``centroid[i] /= n_points;`]{.underline}

[`        ``}`]{.underline}

[`        ``Some(centroid)`]{.underline}

[`    ``}`]{.underline}

[`    ``fn fit(&mut self, X_train: Vec<Vec<f64>>) ``{`]{.underline}

[`        ``// Step 1: Initialize centroids using 3rd, 5th, 7th, ... points (indices 2,4,6,...)`]{.underline}

[`        ``let mut indices = Vec::new();`]{.underline}

[`        ``for i in 0..self.k ``{`]{.underline}

[`            ``indices.push(2 + 2 * i);`]{.underline}

[`        ``}`]{.underline}

[`        ``self.centroids = indices.iter()`]{.underline}

[`            ``.map(|&idx| X_train[idx].clone())`]{.underline}

[`            ``.collect();`]{.underline}

[`        ``// Iterate until convergence`]{.underline}

[`        ``loop ``{`]{.underline}

[`            ``// Step 2: Assign each point to nearest centroid`]{.underline}

[`            ``let mut clusters: Vec<Vec<Vec<f64>>> = vec![Vec::new(); self.k];`]{.underline}

[`            ``for point in &X_train ``{`]{.underline}

[`                ``let mut best_idx = 0;`]{.underline}

[`                ``let mut min_dist = f64::INFINITY;`]{.underline}

[`                ``for (idx, cent) in self.centroids.iter().enumerate() ``{`]{.underline}

[`                    ``let dist = euclidian_distance(point, cent);`]{.underline}

[`                    ``if dist < min_dist ``{`]{.underline}

[`                        ``min_dist = dist;`]{.underline}

[`                        ``best_idx = idx;`]{.underline}

[`                    ``}`]{.underline}

[`                ``}`]{.underline}

[`                ``clusters[best_idx].push(point.clone());`]{.underline}

[`            ``}`]{.underline}

[`            ``// Step 3: Recalculate centroids`]{.underline}

[`            ``let mut new_centroids = Vec::with_capacity(self.k);`]{.underline}

[`            ``for i in 0..self.k ``{`]{.underline}

[`                ``if !clusters[i].is_empty() ``{`]{.underline}

[`                    ``if let Some(cent) = self.find_cluster_centroid(&clusters[i]) ``{`]{.underline}

[`                        ``new_centroids.push(cent);`]{.underline}

[`                    ``}`` else ``{`]{.underline}

[`                        ``new_centroids.push(self.centroids[i].clone());`]{.underline}

[`                    ``}`]{.underline}

[`                ``}`` else ``{`]{.underline}

[`                    ``new_centroids.push(self.centroids[i].clone());`]{.underline}

[`                ``}`]{.underline}

[`            ``}`]{.underline}

[`            ``// Calculate total movement of centroids`]{.underline}

[`            ``let total_movement: f64 = self.centroids.iter()`]{.underline}

[`                ``.zip(&new_centroids)`]{.underline}

[`                ``.map(|(old, new)| euclidian_distance(old, new))`]{.underline}

[`                ``.sum();`]{.underline}

[`            ``// Step 4: Check stop condition`]{.underline}

[`            ``if total_movement < 0.01 ``{`]{.underline}

[`                ``// Keep previous centroids (do not update)`]{.underline}

[`                ``break;`]{.underline}

[`            ``}`` else ``{`]{.underline}

[`                ``self.centroids = new_centroids;`]{.underline}

[`            ``}`]{.underline}

[`        ``}`]{.underline}

[`    ``}`]{.underline}

[`    ``fn predict(&self, _X_test: Vec<Vec<f64>>) ``{`]{.underline}

[`        ``// Not required for this challenge, but keep as placeholder`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[`// Example usage (optional)`]{.underline}

[`fn main() ``{`]{.underline}

[`    ``let mut kmeans = KMeans::new(2);`]{.underline}

[`    ``let X_train = vec![`]{.underline}

[`        ``vec![1.0, 2.0],`]{.underline}

[`        ``vec![3.0, 4.0],`]{.underline}

[`        ``vec![5.0, 6.0],`]{.underline}

[`        ``vec![7.0, 8.0],`]{.underline}

[`        ``vec![9.0, 10.0],`]{.underline}

[`    ``];`]{.underline}

[`    ``kmeans.fit(X_train);`]{.underline}

[`    ``println!("Centroids: ``{``:?``}``", kmeans.centroids);`]{.underline}

[`}`]{.underline}

## [Наивный байесовский классификатор ]{.underline}

[Наивный байесовский классификатор обучается с учителем --- для задач
классификации в машинном обучении. Алгоритм наивного байесовского
классификатора основан на теореме Байеса, которая объясняет, как
рассчитать вероятность события (например, того, что электронное письмо
является спамом) на основе предварительных знаний об условиях, которые
могут быть связаны с этим событием. Слово «наивный» происходит от
предположения, что все признаки (например, слова в электронном письме)
независимы друг от друга, что упрощает вычисления. Независимость
означает, что одно событие не меняет вероятности другого
события.]{.underline}

[Подсчет вхождений: Сначала подсчитывается, сколько раз каждая точка
данных встречается в разных классах. В обнаружении спама мы будем
подсчитывать, сколько раз каждое слово встречалось в спам- или
неспам-письмах, в медицинской диагностике мы можем подсчитать, сколько
раз (или как долго) встречается каждый симптом. ]{.underline}

[Вычисление вероятностей: Для новой точки данных вычисляется вероятность
принадлежности к каждому классу (какова вероятность того, что новое
письмо является спамом, и какова вероятность того, что оно не является
спамом). ]{.underline}

[Принятие решения: Наконец, алгоритм объединяет все эти вероятности,
используя теорему Байеса, и решает, является ли письмо с большей
вероятностью спамом или не спамом. Чем выше вероятность, тем больше
вероятность принадлежности к этому классу. ]{.underline}

[Преимущество наивного байесовского классификатора заключается в его
быстрой и простой реализации. Он хорошо работает с многомерными данными
(например, текстом) и может обрабатывать пропущенные значения. Однако
предположение о независимости не всегда верно в реальной жизни, что
иногда может приводить к менее точным результатам по сравнению с более
сложными алгоритмами.]{.underline}

[`use std::collections::BTreeMap;`]{.underline}

[`use std::fs::File;`]{.underline}

[`use std::io::``{``BufReader, stdin``}``;`]{.underline}

[`use csv::ReaderBuilder;`]{.underline}

[`fn main() ``{`]{.underline}

[`    ``// ``ч``т``е``н``и``е`` ``п``у``т``и`` ``к`` ``ф``а``й``л``у`` ``и`` ``и``м``е``н``и`` ``ц``е``л``е``в``о``г``о`` ``с``т``о``л``б``ц``а`` ``и``з`` stdin`]{.underline}

[`    ``let mut input_line = String::new();`]{.underline}

[`    ``stdin().read_line(&mut input_line).unwrap();`]{.underline}

[`    ``let dataset_path = input_line.trim();`]{.underline}

[`    ``let mut target_var = String::new();`]{.underline}

[`    ``stdin().read_line(&mut target_var).unwrap();`]{.underline}

[`    ``let target_variable = target_var.trim();`]{.underline}

[`    ``// ``о``т``к``р``ы``т``и``е`` CSV`]{.underline}

[`    ``let file = File::open(dataset_path).expect("``Н``е`` ``у``д``а``л``о``с``ь`` ``о``т``к``р``ы``т``ь`` ``ф``а``й``л``");`]{.underline}

[`    ``let reader = BufReader::new(file);`]{.underline}

[`    ``let mut csv_reader = ReaderBuilder::new()`]{.underline}

[`        ``.has_headers(true)`]{.underline}

[`        ``.from_reader(reader);`]{.underline}

[`    ``let headers = csv_reader.headers().unwrap().clone();`]{.underline}

[`    ``// ``ч``т``е``н``и``е`` ``в``с``е``х`` ``с``т``р``о``к`` ``в`` Vec<BTreeMap<String, String>>`]{.underline}

[`    ``let mut rows: Vec<BTreeMap<String, String>> = Vec::new();`]{.underline}

[`    ``for result in csv_reader.records() ``{`]{.underline}

[`        ``let record = result.unwrap();`]{.underline}

[`        ``let mut row = BTreeMap::new();`]{.underline}

[`        ``for (idx, field) in record.iter().enumerate() ``{`]{.underline}

[`            ``let col_name = headers[idx].to_string();`]{.underline}

[`            ``row.insert(col_name, field.to_string());`]{.underline}

[`        ``}`]{.underline}

[`        ``rows.push(row);`]{.underline}

[`    ``}`]{.underline}

[`    ``let total_rows = rows.len();`]{.underline}

[`    ``// ``у``н``и``к``а``л``ь``н``ы``е`` ``з``н``а``ч``е``н``и``я`` ``ц``е``л``е``в``о``г``о`` ``с``т``о``л``б``ц``а`]{.underline}

[`    ``let mut target_values = Vec::new();`]{.underline}

[`    ``for row in &rows ``{`]{.underline}

[`        ``let val = row.get(target_variable).unwrap();`]{.underline}

[`        ``if !target_values.contains(val) ``{`]{.underline}

[`            ``target_values.push(val.clone());`]{.underline}

[`        ``}`]{.underline}

[`    ``}`]{.underline}

[`    ``// P(target) ``–`` ``в``е``р``о``я``т``н``о``с``т``ь`` ``к``а``ж``д``о``г``о`` ``к``л``а``с``с``а`]{.underline}

[`    ``let mut probability_target = BTreeMap::new();`]{.underline}

[`    ``for tval in &target_values ``{`]{.underline}

[`        ``let count = rows.iter()`]{.underline}

[`            ``.filter(|row| row.get(target_variable).unwrap() == tval)`]{.underline}

[`            ``.count();`]{.underline}

[`        ``probability_target.insert(tval.clone(), count as f64 / total_rows as f64);`]{.underline}

[`    ``}`]{.underline}

[`    ``// P(target | feature) ``д``л``я`` ``к``а``ж``д``о``г``о`` ``п``р``и``з``н``а``к``а`]{.underline}

[`    ``let mut probability_features: BTreeMap<String, BTreeMap<String, BTreeMap<String, f64>>> = BTreeMap::new();`]{.underline}

[`    ``for col in headers.iter() ``{`]{.underline}

[`        ``let col = col.to_string();`]{.underline}

[`        ``if col == target_variable ``{`` continue; ``}`]{.underline}

[`        ``// ``у``н``и``к``а``л``ь``н``ы``е`` ``з``н``а``ч``е``н``и``я`` ``п``р``и``з``н``а``к``а`]{.underline}

[`        ``let mut distinct_vals = Vec::new();`]{.underline}

[`        ``for row in &rows ``{`]{.underline}

[`            ``let val = row.get(&col).unwrap();`]{.underline}

[`            ``if !distinct_vals.contains(val) ``{`]{.underline}

[`                ``distinct_vals.push(val.clone());`]{.underline}

[`            ``}`]{.underline}

[`        ``}`]{.underline}

[`        ``let mut feature_map = BTreeMap::new();`]{.underline}

[`        ``for val in &distinct_vals ``{`]{.underline}

[`            ``let count_col_val = rows.iter()`]{.underline}

[`                ``.filter(|row| row.get(&col).unwrap() == val)`]{.underline}

[`                ``.count();`]{.underline}

[`            ``let mut target_map = BTreeMap::new();`]{.underline}

[`            ``for tval in &target_values ``{`]{.underline}

[`                ``let total = rows.iter()`]{.underline}

[`                    ``.filter(|row| row.get(&col).unwrap() == val && row.get(target_variable).unwrap() == tval)`]{.underline}

[`                    ``.count();`]{.underline}

[`                ``let prob = if count_col_val > 0 ``{`]{.underline}

[`                    ``total as f64 / count_col_val as f64`]{.underline}

[`                ``}`` else ``{`]{.underline}

[`                    ``0.0`]{.underline}

[`                ``}``;`]{.underline}

[`                ``target_map.insert(tval.clone(), prob);`]{.underline}

[`            ``}`]{.underline}

[`            ``feature_map.insert(val.clone(), target_map);`]{.underline}

[`        ``}`]{.underline}

[`        ``probability_features.insert(col, feature_map);`]{.underline}

[`    ``}`]{.underline}

[`    ``// ``в``ы``в``о``д`` ``в`` ``ф``о``р``м``а``т``е``, ``п``о``х``о``ж``е``м`` ``н``а`` Python`]{.underline}

[`    ``print_features(&probability_features);`]{.underline}

[`    ``print_target(&probability_target);`]{.underline}

[`}`]{.underline}

[`// ``В``с``п``о``м``о``г``а``т``е``л``ь``н``а``я`` ``ф``у``н``к``ц``и``я`` ``д``л``я`` ``ф``о``р``м``а``т``и``р``о``в``а``н``и``я`` ``ч``и``с``л``а`` ``к``а``к`` ``в`` Python (0.0 ``в``м``е``с``т``о`` 0)`]{.underline}

[`fn format_prob(prob: f64) -> String ``{`]{.underline}

[`    ``if prob.fract() == 0.0 ``{`]{.underline}

[`        ``format!("``{``:.1``}``", prob)   // "0.0", "1.0"`]{.underline}

[`    ``}`` else ``{`]{.underline}

[`        ``// 15 ``з``н``а``к``о``в`` ``п``о``с``л``е`` ``з``а``п``я``т``о``й``, ``к``а``к`` ``ч``а``с``т``о`` ``б``ы``в``а``е``т`` ``в`` repr() float ``в`` Python`]{.underline}

[`        ``format!("``{``:.15``}``", prob).trim_end_matches('0').trim_end_matches('.').to_string()`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[`fn print_features(pf: &BTreeMap<String, BTreeMap<String, BTreeMap<String, f64>>>) ``{`]{.underline}

[`    ``print!("``{``{``");`]{.underline}

[`    ``let mut first_outer = true;`]{.underline}

[`    ``for (col, inner_map) in pf ``{`]{.underline}

[`        ``if !first_outer ``{`` print!(", "); ``}`]{.underline}

[`        ``first_outer = false;`]{.underline}

[`        ``print!("'``{``}``': ``{``{``", col);`]{.underline}

[`        ``let mut first_mid = true;`]{.underline}

[`        ``for (val, target_map) in inner_map ``{`]{.underline}

[`            ``if !first_mid ``{`` print!(", "); ``}`]{.underline}

[`            ``first_mid = false;`]{.underline}

[`            ``print!("'``{``}``': ``{``{``", val);`]{.underline}

[`            ``let mut first_inner = true;`]{.underline}

[`            ``for (targ, prob) in target_map ``{`]{.underline}

[`                ``if !first_inner ``{`` print!(", "); ``}`]{.underline}

[`                ``first_inner = false;`]{.underline}

[`                ``print!("'``{``}``': ``{``}``", targ, format_prob(*prob));`]{.underline}

[`            ``}`]{.underline}

[`            ``print!("``}``}``");`]{.underline}

[`        ``}`]{.underline}

[`        ``print!("``}``}``");`]{.underline}

[`    ``}`]{.underline}

[`    ``println!("``}``}``");`]{.underline}

[`}`]{.underline}

[`fn print_target(pt: &BTreeMap<String, f64>) ``{`]{.underline}

[`    ``print!("``{``{``");`]{.underline}

[`    ``let mut first = true;`]{.underline}

[`    ``for (cls, prob) in pt ``{`]{.underline}

[`        ``if !first ``{`` print!(", "); ``}`]{.underline}

[`        ``first = false;`]{.underline}

[`        ``print!("'``{``}``': ``{``}``", cls, format_prob(*prob));`]{.underline}

[`    ``}`]{.underline}

[`    ``println!("``}``}``");`]{.underline}

[`}`]{.underline}

[Как это работает]{.underline}

1.  [Читает путь к CSV и имя целевого столбца из стандартного
    ввода.]{.underline}

2.  [Парсит CSV, сохраняя строки как отображения «имя\_столбца →
    значение».]{.underline}

3.  [Вычисляет `P(target)` -- долю каждого класса.]{.underline}

4.  [Для каждого нецелевого столбца и каждого его уникального значения
    вычисляет `P(target | feature = ``з``н``а``ч``е``н``и``е``)` как
    отношение количества строк с данным значением признака и данным
    целевым классом к общему числу строк с этим значением
    признака.]{.underline}

5.  [Выводит два отсортированных словаря.]{.underline}

## [Логистическая регрессия]{.underline}

[Логистическая регрессия, является линейной моделью, используемой в
основном для задач классификации, а не регрессии. Она идеально подходит
для ситуаций, когда необходимо предсказать наличие или отсутствие
чего-либо, например, является ли электронное письмо спамом.
Логистическая регрессия хорошо работает для бинарной классификации, но
может быть расширена до многоклассовой классификации с помощью различных
методов. Источник: Википедия По своей сути, логистическая регрессия
предсказывает вероятность того, что данная входная точка принадлежит к
определённому классу. В отличие от линейной регрессии, которая
предсказывает непрерывный результат, логистическая регрессия
предсказывает бинарный результат (1/0, Да/Нет). Суть логистической
регрессии заключается в логистической функции или сигмоидной функции,
которая отображает любое действительное число в значение от 0 до 1, что
делает её подходящей для оценки вероятностей. Подгонка модели
логистической регрессии включает в себя поиск наилучшего набора
коэффициентов для уравнения с учетом обучающих данных. Хотя это
эффективный и простой метод для бинарной классификации, его
производительность может отставать от более сложных моделей для сложных
наборов данных.]{.underline}

[`fn sigmoid(point: &[f64]) -> Vec<f64> ``{`]{.underline}

[`    ``point.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()`]{.underline}

[`}`]{.underline}

[`// ``П``р``и``м``е``р`` ``и``с``п``о``л``ь``з``о``в``а``н``и``я`]{.underline}

[`fn main() ``{`]{.underline}

[`    ``let data = vec![-1.0, 0.0, 1.0, 2.0];`]{.underline}

[`    ``let result = sigmoid(&data);`]{.underline}

[`    ``println!("``{``:?``}``", result);`]{.underline}

[`}`]{.underline}

## [Линейная регрессия ]{.underline}

[Для прогнозирования непрерывных переменных - обучается с учителем. Это
основной метод оценки взаимосвязей между переменными и прогнозирования.
Независимо от того, прогнозируете ли вы цены на жилье, тенденции
фондового рынка или температуру, линейная регрессия может дать
представление о том, как эти переменные коррелируют.Линейная регрессия
пытается смоделировать взаимосвязь между двумя или более переменными
путем подгонки линейного уравнения к наблюдаемым данным. Одна переменная
считается объясняющей переменной (независимой), а другая --- зависимой
переменной. Простейшая форма уравнения линейной регрессии с одной
зависимой и одной независимой переменной определяется следующим образом:
y = α + βx α обозначает точку пересечения с осью y β обозначает
наклон.]{.underline}

[`fn linear_equation(coef: &[f64], x: &[f64]) -> f64 ``{`]{.underline}

[`    ``// coef[0] + coef[1]*x[0] + coef[2]*x[1] + ...`]{.underline}

[`    ``let mut result = coef[0];`]{.underline}

[`    ``for (coef_i, x_i) in coef[1..].iter().zip(x.iter()) ``{`]{.underline}

[`        ``result += coef_i * x_i;`]{.underline}

[`    ``}`]{.underline}

[`    ``result`]{.underline}

[`}`]{.underline}

## [Дерево решений]{.underline}

[Нелинейная модель обучается с учителем, используемая как для
классификации, так и для регрессии. Оно более точно, чем другие
алгоритмы, имитирует процесс принятия решений человеком, что делает его
интуитивно понятным и мощным инструментом для обработки сложных наборов
данных. По своей сути, дерево решений разделяет данные на подмножества,
используя ряд простых правил, что сродни задаванию ряда вопросов
«да/нет» о характеристиках точек данных. Представьте дерево решений в
виде древовидного графа, где: Каждый внутренний узел представляет собой
«проверку» атрибута (например, выпадет ли орёл или решка при
подбрасывании монеты), Каждая ветвь представляет собой результат
проверки, и Каждый листовой узел представляет собой метку класса
(решение, принятое после вычисления всех атрибутов). Типы деревьев
решений: Деревья классификации: Используются, когда прогнозируемый
результат --- это класс, к которому принадлежат данные. Деревья
регрессии: используются, когда прогнозируемый результат можно считать
действительным числом (например, цена дома).]{.underline}

[`use std::collections::HashMap;`]{.underline}

[`#[derive(Debug, PartialEq)]`]{.underline}

[`enum RuleType ``{`]{.underline}

[`    ``Bigger,`]{.underline}

[`    ``Smaller,`]{.underline}

[`}`]{.underline}

[`fn decide_outcome(`]{.underline}

[`    ``rules: &HashMap<String, (RuleType, f64)>,`]{.underline}

[`    ``outcome: &HashMap<String, HashMap<bool, Option<String>>>,`]{.underline}

[`    ``point: &HashMap<String, f64>,`]{.underline}

[`) -> String ``{`]{.underline}

[`    ``let mut current_rule = "rule1".to_string();`]{.underline}

[`    ``loop ``{`]{.underline}

[`        ``// Get rule condition for current rule`]{.underline}

[`        ``let (rule_type, rule_value) = rules.get(&current_rule).expect("Rule not found");`]{.underline}

[`        ``let point_value = point.get(&current_rule).expect("Point value not found");`]{.underline}

[`        ``let condition_met = match rule_type ``{`]{.underline}

[`            ``RuleType::Bigger => *point_value > *rule_value,`]{.underline}

[`            ``RuleType::Smaller => *point_value < *rule_value,`]{.underline}

[`        ``}``;`]{.underline}

[`        ``// Determine next rule from outcome`]{.underline}

[`        ``let next_rule = outcome`]{.underline}

[`            ``.get(&current_rule)`]{.underline}

[`            ``.expect("Outcome for rule not found")`]{.underline}

[`            ``.get(&condition_met)`]{.underline}

[`            ``.expect("Outcome branch not found");`]{.underline}

[`        ``if let Some(next) = next_rule ``{`]{.underline}

[`            ``current_rule = next.clone();`]{.underline}

[`        ``}`` else ``{`]{.underline}

[`            ``return current_rule;`]{.underline}

[`        ``}`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[`fn main() ``{`]{.underline}

[`    ``let mut rules = HashMap::new();`]{.underline}

[`    ``rules.insert("rule1".to_string(), (RuleType::Bigger, 5.0));`]{.underline}

[`    ``rules.insert("rule2".to_string(), (RuleType::Smaller, 7.0));`]{.underline}

[`    ``let mut outcome = HashMap::new();`]{.underline}

[`    ``let mut rule1_outcome = HashMap::new();`]{.underline}

[`    ``rule1_outcome.insert(true, Some("rule2".to_string()));`]{.underline}

[`    ``rule1_outcome.insert(false, Some("rule3".to_string()));`]{.underline}

[`    ``outcome.insert("rule1".to_string(), rule1_outcome);`]{.underline}

[`    ``let mut rule2_outcome = HashMap::new();`]{.underline}

[`    ``rule2_outcome.insert(true, Some("rule3".to_string()));`]{.underline}

[`    ``rule2_outcome.insert(false, Some("rule4".to_string()));`]{.underline}

[`    ``outcome.insert("rule2".to_string(), rule2_outcome);`]{.underline}

[`    ``let mut rule3_outcome = HashMap::new();`]{.underline}

[`    ``rule3_outcome.insert(true, None);`]{.underline}

[`    ``rule3_outcome.insert(false, None);`]{.underline}

[`    ``outcome.insert("rule3".to_string(), rule3_outcome);`]{.underline}

[`    ``let mut rule4_outcome = HashMap::new();`]{.underline}

[`    ``rule4_outcome.insert(true, None);`]{.underline}

[`    ``rule4_outcome.insert(false, None);`]{.underline}

[`    ``outcome.insert("rule4".to_string(), rule4_outcome);`]{.underline}

[`    ``let mut point = HashMap::new();`]{.underline}

[`    ``point.insert("rule1".to_string(), 8.0);`]{.underline}

[`    ``point.insert("rule2".to_string(), 10.0);`]{.underline}

[`    ``let result = decide_outcome(&rules, &outcome, &point);`]{.underline}

[`    ``println!("``{``}``", result); // rule4`]{.underline}

[`}`]{.underline}

## [Метод опорных векторов (SVM) ]{.underline}

[Мощный алгоритм машинного обучения с учителем, используемый как для
классификации, так и для регрессии. Однако чаще всего он применяется в
задачах классификации. SVM обеспечивают высокую точность и хорошо
работают с многомерным пространством признаков, что делает их
подходящими для приложений от распознавания изображений до
биоинформатики. Основная цель алгоритма SVM --- найти оптимальную
гиперплоскость (в двумерном пространстве это линия), которая наилучшим
образом разделяет классы в пространстве признаков. Модель SVM
представляет примеры в виде точек в пространстве, отображаемых таким
образом, что примеры отдельных категорий разделены четким, максимально
широким промежутком. Гиперплоскость: это, по сути, граница принятия
решений, разделяющая классы. В двумерном пространстве эта гиперплоскость
представляет собой линию, но в более высоких измерениях она может быть
более сложной. Опорные векторы: это точки данных, ближайшие к
гиперплоскости, которые имеют решающее значение для определения
положения гиперплоскости. Классификатор SVM сильно зависит от этих
точек, отсюда и название «машина опорных векторов». Отступ: это
расстояние между двумя линиями на ближайших точках данных разных
классов. Его можно рассматривать как отдельную меру между классами.
Обеспечение максимально широкого отступа является ключевой задачей при
обучении SVM. Типы SVM: Линейный SVM: Когда данные линейно разделимы, мы
можем использовать одну прямую линию для разделения классов. В этом
сценарии используется линейный SVM. Нелинейный SVM: Когда данные не
линейно разделимы, SVM использует так называемый «ядерный трюк» для
преобразования входного пространства в многомерное пространство, где для
разделения можно использовать гиперплоскость. Ядерный трюк включает в
себя преобразование данных в другое измерение, имеющее четкий отступ
между классами данных. Это позволяет SVM решать нелинейные задачи.
Распространенные ядра включают: Линейное ядро: Используется, когда
данные линейно разделимы. Полиномиальное ядро: Преобразует исходное
пространство признаков в полиномиальное пространство признаков заданной
степени. Гауссово ядро: Полезно для нелинейных данных, которые
группируются в кластеры. Оно отображает пространство признаков в
многомерное пространство.]{.underline}

[`fn polynomial_kernel(x: &[f64], y: &[f64], d: i32, c: f64) -> f64 ``{`]{.underline}

[`    ``// ``С``к``а``л``я``р``н``о``е`` ``п``р``о``и``з``в``е``д``е``н``и``е`` x ``и`` y`]{.underline}

[`    ``let dot: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();`]{.underline}

[`    ``// (dot + c)^d`]{.underline}

[`    ``(dot + c).powi(d)`]{.underline}

[`}`]{.underline}

[`fn main() ``{`]{.underline}

[`    ``let x = vec![1.0, 2.0, 3.0];`]{.underline}

[`    ``let y = vec![4.0, 5.0, 6.0];`]{.underline}

[`    ``let d = 3;`]{.underline}

[`    ``let c = 1.0;`]{.underline}

[`    ``let result = polynomial_kernel(&x, &y, d, c);`]{.underline}

[`    ``println!("``{``}``", result);`]{.underline}

[`}`]{.underline}

**[П]{.underline}[р]{.underline}[о]{.underline}[с]{.underline}[т]{.underline}[е]{.underline}[й]{.underline}[ш]{.underline}[а]{.underline}[я]{.underline}[
]{.underline}[м]{.underline}[о]{.underline}[д]{.underline}[е]{.underline}[л]{.underline}[ь]{.underline}**

## [Общая идея]{.underline}

[Модель запоминает для каждого уникального значения `X` (например,
`"C"`, `"B"`) самое частое соответствующее значение `y`. Если для одного
`X` встречается несколько разных `y`, то выбирается то, которое
встречается чаще. Если частоты одинаковые (ничья), то выбирается
наименьшее числовое значение `y`. При предсказании для нового `X` модель
возвращает запомненную моду, а если `X` не встречался при обучении ---
возвращает `None`.]{.underline}

[`use std::collections::HashMap;`]{.underline}

[`struct SimpleModel ``{`]{.underline}

[`    ``mode: HashMap<String, i32>,`]{.underline}

[`}`]{.underline}

[`impl SimpleModel ``{`]{.underline}

[`    ``fn new() -> Self ``{`]{.underline}

[`        ``SimpleModel ``{`]{.underline}

[`            ``mode: HashMap::new(),`]{.underline}

[`        ``}`]{.underline}

[`    ``}`]{.underline}

[`    ``fn fit(&mut self, x_train: &[&str], y_train: &[i32]) ``{`]{.underline}

[`        ``// ``Г``р``у``п``п``и``р``у``е``м`` ``з``н``а``ч``е``н``и``я`` y ``п``о`` ``к``а``ж``д``о``м``у`` x`]{.underline}

[`        ``let mut groups: HashMap<String, Vec<i32>> = HashMap::new();`]{.underline}

[`        ``for (x, y) in x_train.iter().zip(y_train.iter()) ``{`]{.underline}

[`            ``groups.entry(x.to_string()).or_insert_with(Vec::new).push(*y);`]{.underline}

[`        ``}`]{.underline}

[`        ``// ``Д``л``я`` ``к``а``ж``д``о``й`` ``г``р``у``п``п``ы`` ``в``ы``ч``и``с``л``я``е``м`` ``м``о``д``у`` (``п``р``и`` ``н``и``ч``ь``е``й`` ``в``ы``б``и``р``а``е``м`` ``н``а``и``м``е``н``ь``ш``е``е`` ``з``н``а``ч``е``н``и``е``)`]{.underline}

[`        ``for (x_val, y_list) in groups ``{`]{.underline}

[`            ``let mut freq: HashMap<i32, usize> = HashMap::new();`]{.underline}

[`            ``for &y in &y_list ``{`]{.underline}

[`                ``*freq.entry(y).or_insert(0) += 1;`]{.underline}

[`            ``}`]{.underline}

[`            ``let max_freq = freq.values().cloned().max().unwrap_or(0);`]{.underline}

[`            ``let mut candidates: Vec<i32> = freq`]{.underline}

[`                ``.into_iter()`]{.underline}

[`                ``.filter(|&(_, f)| f == max_freq)`]{.underline}

[`                ``.map(|(y, _)| y)`]{.underline}

[`                ``.collect();`]{.underline}

[`            ``candidates.sort(); // ``м``и``н``и``м``а``л``ь``н``о``е`` ``з``н``а``ч``е``н``и``е`` ``б``у``д``е``т`` ``п``е``р``в``ы``м`]{.underline}

[`            ``self.mode.insert(x_val, candidates[0]);`]{.underline}

[`        ``}`]{.underline}

[`    ``}`]{.underline}

[`    ``fn predict(&self, x_test: &[&str]) -> Vec<Option<i32>> ``{`]{.underline}

[`        ``x_test.iter()`]{.underline}

[`            ``.map(|x| self.mode.get(*x).cloned())`]{.underline}

[`            ``.collect()`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[`fn train_and_predict(x_train: &[&str], y_train: &[i32], x_test: &[&str]) -> Vec<Option<i32>> ``{`]{.underline}

[`    ``let mut model = SimpleModel::new();`]{.underline}

[`    ``model.fit(x_train, y_train);`]{.underline}

[`    ``model.predict(x_test)`]{.underline}

[`}`]{.underline}

[`// ``П``р``и``м``е``р`` ``и``с``п``о``л``ь``з``о``в``а``н``и``я`` (``м``о``ж``н``о`` ``з``а``к``о``м``м``е``н``т``и``р``о``в``а``т``ь``)`]{.underline}

[`fn main() ``{`]{.underline}

[`    ``let x_train = ["C", "C", "B"];`]{.underline}

[`    ``let y_train = [1, 0, 0];`]{.underline}

[`    ``let x_test = ["C"];`]{.underline}

[`    ``let result = train_and_predict(&x_train, &y_train, &x_test);`]{.underline}

[`    ``println!("``{``:?``}``", result); // [Some(0)]`]{.underline}

[`}`]{.underline}

## [Структура данных]{.underline}

[`struct SimpleModel ``{`]{.underline}

[`    ``mode: HashMap<String, i32>,`]{.underline}

[`}`]{.underline}

-   [`mode` --- это словарь (хеш-таблица), где ключом является строка
    (`String`) --- значение признака `X`, а значением --- `i32` ---
    предсказанная метка (мода). В Rust типы нужно указывать явно:
    `HashMap<K, V>`.]{.underline}

## [Конструктор]{.underline}

[`fn new() -> Self ``{`]{.underline}

[`    ``SimpleModel ``{`]{.underline}

[`        ``mode: HashMap::new(),`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[Создаёт новый экземпляр модели с пустым словарём.]{.underline}

## [Метод `fit` (обучение)]{.underline}

[`fn fit(&mut self, x_train: &[&str], y_train: &[i32]) ``{`]{.underline}

[`    ``let mut groups: HashMap<String, Vec<i32>> = HashMap::new();`]{.underline}

[`    ``for (x, y) in x_train.iter().zip(y_train.iter()) ``{`]{.underline}

[`        ``groups.entry(x.to_string()).or_insert_with(Vec::new).push(*y);`]{.underline}

[`    ``}`]{.underline}

[`    ``// ... ``д``а``л``е``е`` ``в``ы``ч``и``с``л``е``н``и``е`` ``м``о``д``ы`]{.underline}

[`}`]{.underline}

### [Параметры]{.underline}

-   [`&mut self` --- метод изменяет модель (заполняет `mode`), поэтому
    принимает изменяемую ссылку.]{.underline}

-   [`x_train: &[&str]` --- срез строковых срезов (например,
    `&["C", "C", "B"]`). Каждый элемент --- это `&str` (ссылка на
    строку, обычно статическую или из кучи). Приём `&[&str]` позволяет
    передавать массив строк без копирования.]{.underline}

-   [`y_train: &[i32]` --- срез целых чисел.]{.underline}

### [Шаг 1. Группировка значений `y` по `x`]{.underline}

[`let mut groups: HashMap<String, Vec<i32>> = HashMap::new();`]{.underline}

[`for (x, y) in x_train.iter().zip(y_train.iter()) ``{`]{.underline}

[`    ``groups.entry(x.to_string()).or_insert_with(Vec::new).push(*y);`]{.underline}

[`}`]{.underline}

-   [`x_train.iter().zip(y_train.iter())` --- создаёт итератор по парам
    `(&str, &i32)`.]{.underline}

-   [`entry(x.to_string())` --- получает запись в словаре по ключу
    `String`. `x.to_string()` создаёт владеющую строку из `&str`, потому
    что ключом в `HashMap` должна быть владеющая строка (иначе данные
    могут исчезнуть).]{.underline}

-   [`or_insert_with(Vec::new)` --- если ключа ещё нет, вставляет пустой
    вектор `Vec<i32>` и возвращает изменяемую ссылку на него. Если ключ
    есть, просто возвращает ссылку на уже существующий
    вектор.]{.underline}

-   [`.push(*y)` --- разыменовывает `y` (которое `&i32`) и добавляет
    значение в вектор.]{.underline}

[В итоге для примера `x_train = ["C", "C", "B]`, `y_train = [1, 0, 0]`
получим:]{.underline}

[`groups = ``{`]{.underline}

[`    ``"C" -> vec![1, 0],`]{.underline}

[`    ``"B" -> vec![0]`]{.underline}

[`}`]{.underline}

### [Шаг 2. Вычисление моды для каждой группы]{.underline}

[`for (x_val, y_list) in groups ``{`]{.underline}

[`    ``let mut freq: HashMap<i32, usize> = HashMap::new();`]{.underline}

[`    ``for &y in &y_list ``{`]{.underline}

[`        ``*freq.entry(y).or_insert(0) += 1;`]{.underline}

[`    ``}`]{.underline}

[`    ``let max_freq = freq.values().cloned().max().unwrap_or(0);`]{.underline}

[`    ``let mut candidates: Vec<i32> = freq`]{.underline}

[`        ``.into_iter()`]{.underline}

[`        ``.filter(|&(_, f)| f == max_freq)`]{.underline}

[`        ``.map(|(y, _)| y)`]{.underline}

[`        ``.collect();`]{.underline}

[`    ``candidates.sort();`]{.underline}

[`    ``self.mode.insert(x_val, candidates[0]);`]{.underline}

[`}`]{.underline}

-   [Проходим по каждой паре `(x_val, y_list)` из `groups`. Обратите
    внимание: `groups` потребляется (перемещается) в цикле, так как мы
    пишем `for (x_val, y_list) in groups` --- это забирает владение
    элементами (это нормально, они больше не нужны).]{.underline}

-   [Внутри считаем частоту каждого `y` в `y_list`:]{.underline}

[`let mut freq: HashMap<i32, usize> = HashMap::new();`]{.underline}

[`for &y in &y_list ``{`]{.underline}

[`    ``*freq.entry(y).or_insert(0) += 1;`]{.underline}

[`}`]{.underline}

[Здесь `&y_list` --- ссылка на вектор, `for &y in &y_list` итерирует по
ссылкам на элементы и разыменовывает их (получается `y: i32`). Для
каждого `y` увеличиваем счётчик.]{.underline}

-   [`let max_freq = freq.values().cloned().max().unwrap_or(0);`]{.underline}

    -   [`freq.values()` --- итератор по ссылкам на
        счётчики.]{.underline}

    -   [`.cloned()` --- превращает `&usize` в `usize`
        (копирование).]{.underline}

    -   [`.max()` --- возвращает `Option<usize>`, максимальное
        значение.]{.underline}

    -   [`.unwrap_or(0)` --- если словарь пуст (не должно быть), то
        принимаем 0.]{.underline}

```{=html}
<!-- -->
```
-   [Далее собираем все `y`, у которых частота равна
    `max_freq`:]{.underline}

[`let mut candidates: Vec<i32> = freq`]{.underline}

[`    ``.into_iter()`]{.underline}

[`    ``.filter(|&(_, f)| f == max_freq)`]{.underline}

[`    ``.map(|(y, _)| y)`]{.underline}

[`    ``.collect();`]{.underline}

-   [`freq.into_iter()` --- поглощает словарь и возвращает итератор по
    парам `(y, ``ч``а``с``т``о``т``а``)`.]{.underline}

-   [`.filter(...)` --- оставляет только те, где частота ==
    максимальной.]{.underline}

-   [`.map(|(y, _)| y)` --- из пары берём только `y`.]{.underline}

-   [`.collect()` --- собирает в вектор `candidates`.]{.underline}

```{=html}
<!-- -->
```
-   [`candidates.sort();` --- сортируем вектор по
    возрастанию.]{.underline}

-   [`self.mode.insert(x_val, candidates[0]);` --- сохраняем в словарь
    модели наименьшее значение из кандидатов (правило разрешения ничьи).
    Для `"C"` кандидаты: `[1,0]` после сортировки `[0,1]`, берём
    `0`.]{.underline}

[Вычисление завершено. Для нашего примера `self.mode` будет:
`{``"C": 0, "B": 0``}`.]{.underline}

## [Метод `predict`]{.underline}

[`fn predict(&self, x_test: &[&str]) -> Vec<Option<i32>> ``{`]{.underline}

[`    ``x_test.iter()`]{.underline}

[`        ``.map(|x| self.mode.get(*x).cloned())`]{.underline}

[`        ``.collect()`]{.underline}

[`}`]{.underline}

-   [Принимает срез тестовых значений `X` (`&[&str]`).]{.underline}

-   [`x_test.iter()` --- итератор по каждому `&str`.]{.underline}

-   [`.map(|x| self.mode.get(*x).cloned())` --- для каждого
    `x`:]{.underline}

    -   [`self.mode.get(*x)` --- ищет ключ в словаре. `*x`
        разыменовывает `&&str` до `&str`, который автоматически
        преобразуется в `&String`? В Rust `HashMap<String, i32>` имеет
        метод `get(&Q)` где `Q: Borrow<str>`. Поскольку `&str` реализует
        `Borrow<str>`, передавать `&str` можно напрямую. `*x` имеет тип
        `&str`, что подходит. Метод `get` возвращает `Option<&i32>`
        (ссылку на значение, если ключ есть).]{.underline}

    -   [`.cloned()` --- превращает `Option<&i32>` в `Option<i32>`,
        копируя число (для `i32` копирование дешёвое). Если ключа нет,
        остаётся `None`.]{.underline}

```{=html}
<!-- -->
```
-   [`.collect()` --- собирает итератор в
    `Vec<Option<i32>>`.]{.underline}

[Таким образом, для `x_test = ["C"]` результат будет `vec![Some(0)]`,
что соответствует ожиданию задачи.]{.underline}

## [Функция `train_and_predict`]{.underline}

[`fn train_and_predict(x_train: &[&str], y_train: &[i32], x_test: &[&str]) -> Vec<Option<i32>> ``{`]{.underline}

[`    ``let mut model = SimpleModel::new();`]{.underline}

[`    ``model.fit(x_train, y_train);`]{.underline}

[`    ``model.predict(x_test)`]{.underline}

[`}`]{.underline}

-   [Создаёт модель, обучает её и возвращает предсказания для тестовых
    данных. Эта функция является обёрткой, требуемой в
    задании.]{.underline}

## [Пример в `main`]{.underline}

[`fn main() ``{`]{.underline}

[`    ``let x_train = ["C", "C", "B"];`]{.underline}

[`    ``let y_train = [1, 0, 0];`]{.underline}

[`    ``let x_test = ["C"];`]{.underline}

[`    ``let result = train_and_predict(&x_train, &y_train, &x_test);`]{.underline}

[`    ``println!("``{``:?``}``", result); // [Some(0)]`]{.underline}

[`}`]{.underline}

-   [Демонстрация работы. Выведет `[Some(0)]`.]{.underline}

## [Ключевые особенности Rust, использованные в коде]{.underline}

-   **[В]{.underline}[л]{.underline}[а]{.underline}[д]{.underline}[е]{.underline}[н]{.underline}[и]{.underline}[е]{.underline}[
    ]{.underline}[и]{.underline}[
    ]{.underline}[з]{.underline}[а]{.underline}[и]{.underline}[м]{.underline}[с]{.underline}[т]{.underline}[в]{.underline}[о]{.underline}[в]{.underline}[а]{.underline}[н]{.underline}[и]{.underline}[е]{.underline}**[
    --- параметры `&[&str]` позволяют не копировать входные
    данные.]{.underline}

-   **[И]{.underline}[т]{.underline}[е]{.underline}[р]{.underline}[а]{.underline}[т]{.underline}[о]{.underline}[р]{.underline}[ы]{.underline}[
    ]{.underline}[и]{.underline}[
    ]{.underline}[а]{.underline}[д]{.underline}[а]{.underline}[п]{.underline}[т]{.underline}[е]{.underline}[р]{.underline}[ы]{.underline}**[
    --- `iter()`, `zip()`, `map()`, `collect()` --- декларативный стиль
    обработки коллекций.]{.underline}

-   **[Entry API]{.underline}**[ --- эффективная работа со словарём:
    `entry(key).or_insert_with(...)`.]{.underline}

-   **[Pattern matching ]{.underline}[и]{.underline}[
    ]{.underline}[р]{.underline}[а]{.underline}[з]{.underline}[ы]{.underline}[м]{.underline}[е]{.underline}[н]{.underline}[о]{.underline}[в]{.underline}[а]{.underline}[н]{.underline}[и]{.underline}[е]{.underline}**[
    --- `for &y in &y_list` автоматически разыменовывает
    ссылки.]{.underline}

-   **[Option ]{.underline}[и]{.underline}[
    ]{.underline}[о]{.underline}[б]{.underline}[р]{.underline}[а]{.underline}[б]{.underline}[о]{.underline}[т]{.underline}[к]{.underline}[а]{.underline}[
    ]{.underline}[о]{.underline}[т]{.underline}[с]{.underline}[у]{.underline}[т]{.underline}[с]{.underline}[т]{.underline}[в]{.underline}[и]{.underline}[я]{.underline}[
    ]{.underline}[з]{.underline}[н]{.underline}[а]{.underline}[ч]{.underline}[е]{.underline}[н]{.underline}[и]{.underline}[я]{.underline}**[
    --- `get()` возвращает `Option`, который мы преобразуем в
    `Option<i32>`.]{.underline}

**[Б]{.underline}[о]{.underline}[л]{.underline}[е]{.underline}[е]{.underline}[
]{.underline}[с]{.underline}[л]{.underline}[о]{.underline}[ж]{.underline}[н]{.underline}[а]{.underline}[я]{.underline}[
]{.underline}[м]{.underline}[о]{.underline}[д]{.underline}[е]{.underline}[л]{.underline}[ь]{.underline}**

## [Общая идея]{.underline}

[Модель запоминает наиболее часто встречающееся значение `y` (моду) для
каждого уникального `x`. При предсказании для нового `x` возвращается
соответствующая мода (или `None`, если такой `x` не встречался в
обучении). Если для одного `x` есть несколько `y` с одинаковой
максимальной частотой --- выбирается
]{.underline}**[п]{.underline}[о]{.underline}[с]{.underline}[л]{.underline}[е]{.underline}[д]{.underline}[н]{.underline}[и]{.underline}[й]{.underline}**[
встреченный.]{.underline}

### [Пояснения]{.underline}

-   [`fit` собирает все значения `y` для каждого `x` в вектор, затем для
    каждого вектора подсчитывает частоты и определяет моду, обновляя
    кандидата при `*count >= max_count`. Это гарантирует выбор
    ]{.underline}**[п]{.underline}[о]{.underline}[с]{.underline}[л]{.underline}[е]{.underline}[д]{.underline}[н]{.underline}[е]{.underline}[г]{.underline}[о]{.underline}**[
    значения при равных частотах .]{.underline}

-   [`predict` возвращает `Vec<Option<i32>>`, где отсутствующим ключам
    соответствует `None`.]{.underline}

-   [`train_and_predict` создаёт модель, обучает её и возвращает
    предсказания.]{.underline}

[`use std::collections::HashMap;`]{.underline}

[`#[derive(Debug)]`]{.underline}

[`struct SimpleModel ``{`]{.underline}

[`    ``mode: HashMap<String, i32>,`]{.underline}

[`}`]{.underline}

[`impl SimpleModel ``{`]{.underline}

[`    ``fn new() -> Self ``{`]{.underline}

[`        ``SimpleModel ``{`]{.underline}

[`            ``mode: HashMap::new(),`]{.underline}

[`        ``}`]{.underline}

[`    ``}`]{.underline}

[`    ``fn fit(&mut self, X: Vec<String>, y: Vec<i32>) ``{`]{.underline}

[`        ``// ``С``о``б``и``р``а``е``м`` ``в``с``е`` ``з``н``а``ч``е``н``и``я`` y ``д``л``я`` ``к``а``ж``д``о``г``о`` ``у``н``и``к``а``л``ь``н``о``г``о`` X (``в`` ``п``о``р``я``д``к``е`` ``п``о``с``т``у``п``л``е``н``и``я``)`]{.underline}

[`        ``let mut buckets: HashMap<String, Vec<i32>> = HashMap::new();`]{.underline}

[`        ``for (x_val, y_val) in X.iter().zip(y.iter()) ``{`]{.underline}

[`            ``buckets.entry(x_val.clone()).or_insert_with(Vec::new).push(*y_val);`]{.underline}

[`        ``}`]{.underline}

[`        ``// ``Д``л``я`` ``к``а``ж``д``о``г``о`` X ``в``ы``ч``и``с``л``я``е``м`` ``м``о``д``у`` (``н``а``и``б``о``л``е``е`` ``ч``а``с``т``о``е`` y)`]{.underline}

[`        ``for (x_val, y_vals) in buckets ``{`]{.underline}

[`            ``let mut counts: HashMap<i32, usize> = HashMap::new();`]{.underline}

[`            ``let mut mode_y = None;`]{.underline}

[`            ``let mut max_count = 0;`]{.underline}

[`            ``for y in y_vals ``{`]{.underline}

[`                ``let count = counts.entry(y).or_insert(0);`]{.underline}

[`                ``*count += 1;`]{.underline}

[`                ``// ``П``р``и`` ``р``а``в``е``н``с``т``в``е`` ``ч``а``с``т``о``т`` ``о``б``н``о``в``л``я``е``м`` ``–`` ``б``е``р``ё``м`` ``п``о``с``л``е``д``н``и``й`` ``в``с``т``р``е``ч``е``н``н``ы``й`]{.underline}

[`                ``if *count >= max_count ``{`]{.underline}

[`                    ``max_count = *count;`]{.underline}

[`                    ``mode_y = Some(y);`]{.underline}

[`                ``}`]{.underline}

[`            ``}`]{.underline}

[`            ``if let Some(mode) = mode_y ``{`]{.underline}

[`                ``self.mode.insert(x_val, mode);`]{.underline}

[`            ``}`]{.underline}

[`        ``}`]{.underline}

[`    ``}`]{.underline}

[`    ``fn predict(&self, X: Vec<String>) -> Vec<Option<i32>> ``{`]{.underline}

[`        ``X.iter().map(|x| self.mode.get(x).copied()).collect()`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[`fn train_and_predict(X_train: Vec<String>, y_train: Vec<i32>, X_test: Vec<String>) -> Vec<Option<i32>> ``{`]{.underline}

[`    ``let mut model = SimpleModel::new();`]{.underline}

[`    ``model.fit(X_train, y_train);`]{.underline}

[`    ``model.predict(X_test)`]{.underline}

[`}`]{.underline}

[`fn main() ``{`]{.underline}

[`    ``// ``П``р``и``м``е``р`` ``и``з`` ``у``с``л``о``в``и``я`]{.underline}

[`    ``let X_train = vec!["C".to_string(), "C".to_string(), "B".to_string()];`]{.underline}

[`    ``let y_train = vec![1, 0, 0];`]{.underline}

[`    ``let X_test = vec!["C".to_string()];`]{.underline}

[`    ``let predictions = train_and_predict(X_train, y_train, X_test);`]{.underline}

[`    ``println!("``{``:?``}``", predictions); // ``В``ы``в``е``д``е``т`` [Some(0)]`]{.underline}

[`}`]{.underline}

## [Импорты и объявление структуры]{.underline}

[`use std::collections::HashMap;`]{.underline}

[`#[derive(Debug)]`]{.underline}

[`struct SimpleModel ``{`]{.underline}

[`    ``mode: HashMap<String, i32>,`]{.underline}

[`}`]{.underline}

-   [`HashMap` из стандартной библиотеки .]{.underline}

-   [`#[derive(Debug)]` позволяет выводить структуру в отладочных целях
    (например, через `println!("``{``:?``}``", model)`).]{.underline}

-   [`SimpleModel` содержит поле `mode` --- отображение из строки `x` в
    моду (целое число).]{.underline}

## [Реализация методов]{.underline}

### [`fn new() -> Self`]{.underline}

[`fn new() -> Self ``{`]{.underline}

[`    ``SimpleModel ``{`]{.underline}

[`        ``mode: HashMap::new(),`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[Конструктор, создающий пустую модель.]{.underline}

### [Метод `fit`]{.underline}

[`fn fit(&mut self, X: Vec<String>, y: Vec<i32>) ``{`]{.underline}

[`    ``// 1. ``Г``р``у``п``п``и``р``у``е``м`` ``в``с``е`` ``з``н``а``ч``е``н``и``я`` y ``п``о`` x (``в`` ``п``о``р``я``д``к``е`` ``п``о``с``т``у``п``л``е``н``и``я``)`]{.underline}

[`    ``let mut buckets: HashMap<String, Vec<i32>> = HashMap::new();`]{.underline}

[`    ``for (x_val, y_val) in X.iter().zip(y.iter()) ``{`]{.underline}

[`        ``buckets.entry(x_val.clone()).or_insert_with(Vec::new).push(*y_val);`]{.underline}

[`    ``}`]{.underline}

[`    ``// 2. ``Д``л``я`` ``к``а``ж``д``о``г``о`` x ``в``ы``ч``и``с``л``я``е``м`` ``м``о``д``у`]{.underline}

[`    ``for (x_val, y_vals) in buckets ``{`]{.underline}

[`        ``let mut counts: HashMap<i32, usize> = HashMap::new();`]{.underline}

[`        ``let mut mode_y = None;`]{.underline}

[`        ``let mut max_count = 0;`]{.underline}

[`        ``for y in y_vals ``{`]{.underline}

[`            ``let count = counts.entry(y).or_insert(0);`]{.underline}

[`            ``*count += 1;`]{.underline}

[`            ``// ``Е``с``л``и`` ``ч``а``с``т``о``т``а`` >= ``м``а``к``с``и``м``а``л``ь``н``о``й`` ``—`` ``о``б``н``о``в``л``я``е``м`` ``м``о``д``у`` (``б``е``р``ё``м`` ``п``о``с``л``е``д``н``и``й`` ``п``р``и`` ``р``а``в``е``н``с``т``в``е``)`]{.underline}

[`            ``if *count >= max_count ``{`]{.underline}

[`                ``max_count = *count;`]{.underline}

[`                ``mode_y = Some(y);`]{.underline}

[`            ``}`]{.underline}

[`        ``}`]{.underline}

[`        ``if let Some(mode) = mode_y ``{`]{.underline}

[`            ``self.mode.insert(x_val, mode);`]{.underline}

[`        ``}`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

#### [Шаг 1: группировка значений `y` по ключам `x`]{.underline}

-   [Создаём пустую хеш-таблицу `buckets`, где ключ --- `String` (копия
    `x_val`), значение --- вектор `i32`.]{.underline}

-   [`X.iter().zip(y.iter())` --- параллельный итератор по `X` и `y`.
    Каждая итерация даёт пару `(&String, &i32)`.]{.underline}

-   [`buckets.entry(x_val.clone())` --- получает «вход» для ключа. Если
    ключа нет, то `or_insert_with(Vec::new)` создаёт новый пустой
    вектор. Затем `.push(*y_val)` добавляет значение `y` в этот
    вектор.]{.underline}

#### [Шаг 2: вычисление моды для каждого `x`]{.underline}

-   [Перебираем каждую пару `(x_val, y_vals)` из `buckets`.]{.underline}

-   [Внутренняя хеш-таблица `counts` подсчитывает, сколько раз
    встретилось каждое `y` в векторе `y_vals`.]{.underline}

-   [Перебираем `y` из `y_vals` (сохраняя порядок, в котором они были
    добавлены). Для каждого:]{.underline}

    -   [Увеличиваем счётчик в `counts`.]{.underline}

    -   [Если текущий счётчик `>= max_count`, то обновляем `max_count` и
        запоминаем `mode_y = Some(y)`.
        ]{.underline}**[В]{.underline}[а]{.underline}[ж]{.underline}[н]{.underline}[о]{.underline}**[:
        условие `>=`, а не `>`. Это означает, что при равенстве частот
        более позднее (последнее) значение затирает предыдущее.
        ]{.underline}

```{=html}
<!-- -->
```
-   [После цикла, если `mode_y` определена, вставляем в `self.mode` пару
    `(x_val, mode)`.]{.underline}

### [Метод `predict`]{.underline}

[`fn predict(&self, X: Vec<String>) -> Vec<Option<i32>> ``{`]{.underline}

[`    ``X.iter().map(|x| self.mode.get(x).copied()).collect()`]{.underline}

[`}`]{.underline}

-   [Принимает вектор `X_test` (строки).]{.underline}

-   [Для каждого элемента `x` вызывает `self.mode.get(x)`, который
    возвращает `Option<&i32>`.]{.underline}

-   [`copied()` превращает `Option<&i32>` в `Option<i32>` (копирует
    значение, если оно есть).]{.underline}

-   [Результат собирается в `Vec<Option<i32>>`. Если ключа нет в
    `self.mode`, будет `None`.]{.underline}

### [Функция `train_and_predict`]{.underline}

[`fn train_and_predict(X_train: Vec<String>, y_train: Vec<i32>, X_test: Vec<String>) -> Vec<Option<i32>> ``{`]{.underline}

[`    ``let mut model = SimpleModel::new();`]{.underline}

[`    ``model.fit(X_train, y_train);`]{.underline}

[`    ``model.predict(X_test)`]{.underline}

[`}`]{.underline}

[Просто создаёт модель, обучает её на обучающих данных и возвращает
предсказания для тестовых.]{.underline}

### [Функция `main` (пример использования)]{.underline}

[`fn main() ``{`]{.underline}

[`    ``let X_train = vec!["C".to_string(), "C".to_string(), "B".to_string()];`]{.underline}

[`    ``let y_train = vec![1, 0, 0];`]{.underline}

[`    ``let X_test = vec!["C".to_string()];`]{.underline}

[`    ``let predictions = train_and_predict(X_train, y_train, X_test);`]{.underline}

[`    ``println!("``{``:?``}``", predictions); // [Some(0)]`]{.underline}

[`}`]{.underline}

-   [Обучающие данные: для `"C"` значения `[1, 0]`, для `"B"`
    `[0]`.]{.underline}

-   [Подсчёт: для `"C"`: `1` встречается 1 раз, `0` встречается 1 раз.
    Порядок добавления в `y_vals`: `[1, 0]`. При проверке:]{.underline}

    -   [Первый `y=1`: `count=1`, `max_count=0`, условие `>=`
        выполняется → `mode=1`, `max_count=1`.]{.underline}

    -   [Второй `y=0`: `count=1`, `max_count=1`, условие `>=` (1 \>= 1)
        выполняется → `mode=0`, `max_count=1`.]{.underline}

```{=html}
<!-- -->
```
-   [Итог: мода для `"C"` = `0`. `"B"` имеет единственное значение `0` →
    мода `0`.]{.underline}

-   [Предсказание для `"C"` → `Some(0)`.]{.underline}

**[П]{.underline}[р]{.underline}[и]{.underline}[м]{.underline}[е]{.underline}[р]{.underline}[ы]{.underline}[
]{.underline}[о]{.underline}[б]{.underline}[у]{.underline}[ч]{.underline}[е]{.underline}[н]{.underline}[и]{.underline}[я]{.underline}[
]{.underline}[м]{.underline}[о]{.underline}[д]{.underline}[е]{.underline}[л]{.underline}[и]{.underline}[
]{.underline}[в]{.underline}[ Burn]{.underline}**

[Наиболее простой вариант обучения модели содержится в модуле Context
ML.]{.underline}

[Обучение модели строиться везде по разному. Например в Burn обучение
модели строится вокруг
]{.underline}**[Autodiff]{.underline}**[-декоратора, который добавляет
автоматическое дифференцирование к любому бэкенду. Это позволяет писать
один и тот же код модели, а обучение проводить на CPU или GPU, просто
меняя тип бэкенда.]{.underline}

[Основные компоненты обучения любой модели:]{.underline}

-   [Autodiff\<B\> --- бэкенд с градиентами.]{.underline}

-   [Оптимизатор (Adam, SGD и др.).]{.underline}

-   [Функция потерь (CrossEntropy, MSE и т.д.).]{.underline}

-   [Даталоадеры (Burn поддерживает свои или из
    burn-dataset).]{.underline}

-   [Learner --- high-level API для запуска обучения (с метриками,
    чекпоинтами, логами).]{.underline}

[1. ]{.underline}**[Autodiff ]{.underline}[---]{.underline}[
]{.underline}[б]{.underline}[э]{.underline}[к]{.underline}[е]{.underline}[н]{.underline}[д]{.underline}[
]{.underline}[с]{.underline}[
]{.underline}[г]{.underline}[р]{.underline}[а]{.underline}[д]{.underline}[и]{.underline}[е]{.underline}[н]{.underline}[т]{.underline}[а]{.underline}[м]{.underline}[и]{.underline}**

**[П]{.underline}[р]{.underline}[о]{.underline}[с]{.underline}[т]{.underline}[ы]{.underline}[м]{.underline}[и]{.underline}[
]{.underline}[с]{.underline}[л]{.underline}[о]{.underline}[в]{.underline}[а]{.underline}[м]{.underline}[и]{.underline}[:]{.underline}**[
Autodiff (автоматическое дифференцирование) --- это технология, которая
автоматически считает производные (градиенты) функций. Градиенты
показывают, как сильно нужно изменить параметры модели, чтобы уменьшить
ошибку.]{.underline}

**[П]{.underline}[р]{.underline}[и]{.underline}[м]{.underline}[е]{.underline}[р]{.underline}[:]{.underline}**[
Представьте, что вы учитесь кататься на велосипеде. Autodiff --- это как
тренер, который говорит: «Ты слишком сильно наклонился влево, поправь
руль вправо на 10 градусов, чтобы не упасть». Здесь «на 10 градусов» ---
это градиент.]{.underline}

[2.
]{.underline}**[О]{.underline}[п]{.underline}[т]{.underline}[и]{.underline}[м]{.underline}[и]{.underline}[з]{.underline}[а]{.underline}[т]{.underline}[о]{.underline}[р]{.underline}[
(Adam, SGD ]{.underline}[и]{.underline}[
]{.underline}[д]{.underline}[р]{.underline}[.)]{.underline}**

**[П]{.underline}[р]{.underline}[о]{.underline}[с]{.underline}[т]{.underline}[ы]{.underline}[м]{.underline}[и]{.underline}[
]{.underline}[с]{.underline}[л]{.underline}[о]{.underline}[в]{.underline}[а]{.underline}[м]{.underline}[и]{.underline}[:]{.underline}**[
Оптимизатор --- это алгоритм, который использует градиенты, чтобы
обновлять параметры модели. Он решает, насколько сильно и в каком
направлении изменить параметры, чтобы модель быстрее и точнее
училась.]{.underline}

**[П]{.underline}[р]{.underline}[и]{.underline}[м]{.underline}[е]{.underline}[р]{.underline}[ы]{.underline}[:]{.underline}**

-   **[SGD
    (]{.underline}[С]{.underline}[т]{.underline}[о]{.underline}[х]{.underline}[а]{.underline}[с]{.underline}[т]{.underline}[и]{.underline}[ч]{.underline}[е]{.underline}[с]{.underline}[к]{.underline}[и]{.underline}[й]{.underline}[
    ]{.underline}[г]{.underline}[р]{.underline}[а]{.underline}[д]{.underline}[и]{.underline}[е]{.underline}[н]{.underline}[т]{.underline}[н]{.underline}[ы]{.underline}[й]{.underline}[
    ]{.underline}[с]{.underline}[п]{.underline}[у]{.underline}[с]{.underline}[к]{.underline}[):]{.underline}**[
    Как спускаться с горы маленькими шагами, каждый раз выбирая
    направление вниз.]{.underline}

-   **[Adam:]{.underline}**[ Более умный вариант SGD, который
    подстраивает размер шага для каждого параметра
    отдельно.]{.underline}

[3.
]{.underline}**[Ф]{.underline}[у]{.underline}[н]{.underline}[к]{.underline}[ц]{.underline}[и]{.underline}[я]{.underline}[
]{.underline}[п]{.underline}[о]{.underline}[т]{.underline}[е]{.underline}[р]{.underline}[ь]{.underline}[
(CrossEntropy, MSE ]{.underline}[и]{.underline}[
]{.underline}[т]{.underline}[.]{.underline}[д]{.underline}[.)]{.underline}**

**[П]{.underline}[р]{.underline}[о]{.underline}[с]{.underline}[т]{.underline}[ы]{.underline}[м]{.underline}[и]{.underline}[
]{.underline}[с]{.underline}[л]{.underline}[о]{.underline}[в]{.underline}[а]{.underline}[м]{.underline}[и]{.underline}[:]{.underline}**[
Функция потерь --- это способ измерить, насколько плохо ваша модель
работает. Чем меньше значение функции потерь, тем лучше
модель.]{.underline}

**[П]{.underline}[р]{.underline}[и]{.underline}[м]{.underline}[е]{.underline}[р]{.underline}[ы]{.underline}[:]{.underline}**

-   **[MSE
    (]{.underline}[С]{.underline}[р]{.underline}[е]{.underline}[д]{.underline}[н]{.underline}[е]{.underline}[к]{.underline}[в]{.underline}[а]{.underline}[д]{.underline}[р]{.underline}[а]{.underline}[т]{.underline}[и]{.underline}[ч]{.underline}[н]{.underline}[а]{.underline}[я]{.underline}[
    ]{.underline}[о]{.underline}[ш]{.underline}[и]{.underline}[б]{.underline}[к]{.underline}[а]{.underline}[):]{.underline}**[
    Используется для задач предсказания чисел (например, цена на дом).
    Считает, насколько сильно предсказания модели отличаются от реальных
    значений.]{.underline}

-   **[CrossEntropy:]{.underline}**[ Используется для задач
    классификации (например, распознавание кошек и собак). Считает,
    насколько уверенно модель ошиблась.]{.underline}

[4.
]{.underline}**[Д]{.underline}[а]{.underline}[т]{.underline}[а]{.underline}[л]{.underline}[о]{.underline}[а]{.underline}[д]{.underline}[е]{.underline}[р]{.underline}[ы]{.underline}[
(Burn
]{.underline}[п]{.underline}[о]{.underline}[д]{.underline}[д]{.underline}[е]{.underline}[р]{.underline}[ж]{.underline}[и]{.underline}[в]{.underline}[а]{.underline}[е]{.underline}[т]{.underline}[
]{.underline}[с]{.underline}[в]{.underline}[о]{.underline}[и]{.underline}[
]{.underline}[и]{.underline}[л]{.underline}[и]{.underline}[
]{.underline}[и]{.underline}[з]{.underline}[
burn-dataset)]{.underline}**

**[П]{.underline}[р]{.underline}[о]{.underline}[с]{.underline}[т]{.underline}[ы]{.underline}[м]{.underline}[и]{.underline}[
]{.underline}[с]{.underline}[л]{.underline}[о]{.underline}[в]{.underline}[а]{.underline}[м]{.underline}[и]{.underline}[:]{.underline}**[
Даталоадер --- это инструмент, который загружает данные для обучения
модели порциями (батчами). Он отвечает за то, чтобы данные подавались в
модель быстро и эффективно.]{.underline}

**[П]{.underline}[р]{.underline}[и]{.underline}[м]{.underline}[е]{.underline}[р]{.underline}[:]{.underline}**[
Представьте, что вы учитесь готовить блюдо по рецепту. Даталоадер ---
это как помощник, который подаёт вам ингредиенты небольшими порциями,
чтобы вы не перегрузились и могли готовить без остановки.]{.underline}

[5. ]{.underline}**[Learner ]{.underline}[---]{.underline}[ high-level
API ]{.underline}[д]{.underline}[л]{.underline}[я]{.underline}[
]{.underline}[з]{.underline}[а]{.underline}[п]{.underline}[у]{.underline}[с]{.underline}[к]{.underline}[а]{.underline}[
]{.underline}[о]{.underline}[б]{.underline}[у]{.underline}[ч]{.underline}[е]{.underline}[н]{.underline}[и]{.underline}[я]{.underline}[
(]{.underline}[с]{.underline}[
]{.underline}[м]{.underline}[е]{.underline}[т]{.underline}[р]{.underline}[и]{.underline}[к]{.underline}[а]{.underline}[м]{.underline}[и]{.underline}[,
]{.underline}[ч]{.underline}[е]{.underline}[к]{.underline}[п]{.underline}[о]{.underline}[и]{.underline}[н]{.underline}[т]{.underline}[а]{.underline}[м]{.underline}[и]{.underline}[,
]{.underline}[л]{.underline}[о]{.underline}[г]{.underline}[а]{.underline}[м]{.underline}[и]{.underline}[)]{.underline}**

**[П]{.underline}[р]{.underline}[о]{.underline}[с]{.underline}[т]{.underline}[ы]{.underline}[м]{.underline}[и]{.underline}[
]{.underline}[с]{.underline}[л]{.underline}[о]{.underline}[в]{.underline}[а]{.underline}[м]{.underline}[и]{.underline}[:]{.underline}**[
Learner --- это готовый инструмент, который упрощает процесс обучения
модели. Он автоматически следит за метриками (например, точность),
сохраняет промежуточные результаты (чекпоинты) и ведёт логи (журнал
обучения).]{.underline}

**[П]{.underline}[р]{.underline}[и]{.underline}[м]{.underline}[е]{.underline}[р]{.underline}[:]{.underline}**[
Learner --- это как автопилот в самолёте: вы задаёте маршрут (параметры
обучения), а он сам следит за высотой, скоростью и записывает все данные
о полёте.]{.underline}

#### [Полный пример: обучение модели на MNIST]{.underline}

[Это адаптированный пример из официальных examples Burn
(mnist-training). Предполагаем, что у вас скачаны данные MNIST в папку
data/mnist.]{.underline}

[`use burn::``{`]{.underline}

[`    ``config::Config,`]{.underline}

[`    ``data::``{`]{.underline}

[`        ``dataset::``{``Dataset, MnistDataset``}``,`]{.underline}

[`        ``dataloader::``{``batcher::Batcher, DataLoaderBuilder``}``,`]{.underline}

[`    ``}``,`]{.underline}

[`    ``module::Module,`]{.underline}

[`    ``nn::``{`]{.underline}

[`        ``loss::CrossEntropyLoss,`]{.underline}

[`        ``Linear, LinearConfig, Relu,`]{.underline}

[`    ``}``,`]{.underline}

[`    ``optim::AdamConfig,`]{.underline}

[`    ``prelude::*,`]{.underline}

[`    ``tensor::backend::Autodiff,`]{.underline}

[`    ``train::``{`]{.underline}

[`        ``metric::``{``AccuracyMetric, LossMetric``}``,`]{.underline}

[`        ``LearnerBuilder,`]{.underline}

[`    ``}``,`]{.underline}

[`}``;`]{.underline}

[`#[derive(Module, Debug)]`]{.underline}

[`pub struct Model<B: Backend> ``{`]{.underline}

[`    ``linear1: Linear<B>,`]{.underline}

[`    ``linear2: Linear<B>,`]{.underline}

[`    ``activation: Relu,`]{.underline}

[`}`]{.underline}

[`impl<B: Backend> Model<B> ``{`]{.underline}

[`    ``pub fn forward(&self, images: Tensor<B, 2>) -> Tensor<B, 2> ``{`]{.underline}

[`        ``let x = self.linear1.forward(images);`]{.underline}

[`        ``let x = self.activation.forward(x);`]{.underline}

[`        ``self.linear2.forward(x)`]{.underline}

[`    ``}`]{.underline}

[`    ``// ``К``л``а``с``с``и``ф``и``к``а``ц``и``я``: ``д``о``б``а``в``л``я``е``м`` log_softmax ``д``л``я`` CrossEntropyLoss`]{.underline}

[`    ``pub fn forward_classification(&self, images: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Loss<B> ``{`]{.underline}

[`        ``let output = self.forward(images);`]{.underline}

[`        ``let loss = CrossEntropyLoss::new(None, &self.devices());`]{.underline}

[`        ``loss.forward(output, targets)`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[`#[derive(Config)]`]{.underline}

[`pub struct ModelConfig ``{`]{.underline}

[`    ``#[config(default = 10)]`]{.underline}

[`    ``num_classes: usize,`]{.underline}

[`    ``#[config(default = 128)]`]{.underline}

[`    ``hidden_size: usize,`]{.underline}

[`}`]{.underline}

[`impl ModelConfig ``{`]{.underline}

[`    ``pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> ``{`]{.underline}

[`        ``Model ``{`]{.underline}

[`            ``linear1: LinearConfig::new(28 * 28, self.hidden_size).init(device),`]{.underline}

[`            ``activation: Relu::new(),`]{.underline}

[`            ``linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),`]{.underline}

[`        ``}`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[`// Batcher ``д``л``я`` MNIST`]{.underline}

[`#[derive(Clone)]`]{.underline}

[`pub struct MnistBatcher<B: Backend> ``{`]{.underline}

[`    ``device: B::Device,`]{.underline}

[`}`]{.underline}

[`impl<B: Backend> MnistBatcher<B> ``{`]{.underline}

[`    ``pub fn new(device: B::Device) -> Self ``{`]{.underline}

[`        ``Self ``{`` device ``}`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[`#[derive(Clone, Debug)]`]{.underline}

[`pub struct MnistBatch<B: Backend> ``{`]{.underline}

[`    ``pub images: Tensor<B, 3>,`]{.underline}

[`    ``pub targets: Tensor<B, 1, Int>,`]{.underline}

[`}`]{.underline}

[`impl<B: Backend> Batcher<MnistItem> for MnistBatcher<B> ``{`]{.underline}

[`    ``type Batch = MnistBatch<B>;`]{.underline}

[`    ``fn batch(&self, items: Vec<MnistItem>) -> Self::Batch ``{`]{.underline}

[`        ``let images = items`]{.underline}

[`            ``.iter()`]{.underline}

[`            ``.map(|item| item.image.clone())`]{.underline}

[`            ``.collect::<Vec<_>>()`]{.underline}

[`            ``.concat()`]{.underline}

[`            ``.reshape([items.len(), 1, 28, 28])`]{.underline}

[`            ``.to_device(&self.device);`]{.underline}

[`        ``let targets = items`]{.underline}

[`            ``.iter()`]{.underline}

[`            ``.map(|item| item.label as i64)`]{.underline}

[`            ``.collect::<Vec<_>>()`]{.underline}

[`            ``.into_tensor()`]{.underline}

[`            ``.reshape([items.len(), 1])`]{.underline}

[`            ``.to_device(&self.device);`]{.underline}

[`        ``MnistBatch ``{`` images: images.squeeze(1), targets ``}`]{.underline}

[`    ``}`]{.underline}

[`}`]{.underline}

[`fn main() ``{`]{.underline}

[`    ``// ``В``ы``б``о``р`` ``б``э``к``е``н``д``а`]{.underline}

[`    ``type Backend = burn::backend::NdArray; // ``Д``л``я`` CPU`]{.underline}

[`    ``// type Backend = burn::backend::Wgpu; // ``Д``л``я`` GPU (``н``у``ж``н``о`` ``у``к``а``з``а``т``ь`` f32/i32)`]{.underline}

[`    ``type TrainBackend = Autodiff<Backend>;`]{.underline}

[`    ``let device = TrainBackend::Device::default();`]{.underline}

[`    ``// ``Д``а``т``а``л``о``а``д``е``р``ы`]{.underline}

[`    ``let batcher_train = MnistBatcher::<TrainBackend>::new(device.clone());`]{.underline}

[`    ``let batcher_valid = MnistBatcher::<Backend>::new(device.clone()); // ``Б``е``з`` autodiff ``д``л``я`` ``в``а``л``и``д``а``ц``и``и`]{.underline}

[`    ``let dataloader_train = DataLoaderBuilder::new(batcher_train)`]{.underline}

[`        ``.batch_size(32)`]{.underline}

[`        ``.shuffle(42)`]{.underline}

[`        ``.build(MnistDataset::train());`]{.underline}

[`    ``let dataloader_valid = DataLoaderBuilder::new(batcher_valid)`]{.underline}

[`        ``.batch_size(256)`]{.underline}

[`        ``.build(MnistDataset::test());`]{.underline}

[`    ``// ``К``о``н``ф``и``г`]{.underline}

[`    ``let config = ModelConfig::new();`]{.underline}

[`    ``let optimizer = AdamConfig::new().init();`]{.underline}

[`    ``let model = config.init::<TrainBackend>(&device);`]{.underline}

[`    ``// Learner`]{.underline}

[`    ``let learner = LearnerBuilder::new("mnist-out")`]{.underline}

[`        ``.metric_train_numeric(AccuracyMetric::new())`]{.underline}

[`        ``.metric_valid_numeric(AccuracyMetric::new())`]{.underline}

[`        ``.metric_train(LossMetric::new())`]{.underline}

[`        ``.metric_valid(LossMetric::new())`]{.underline}

[`        ``.devices(vec![device])`]{.underline}

[`        ``.num_epochs(10)`]{.underline}

[`        ``.build(model, optimizer, CrossEntropyLoss::new(None, &[device]));`]{.underline}

[`    ``let trained_model = learner.learn(dataloader_train, dataloader_valid);`]{.underline}

[`    ``// ``С``о``х``р``а``н``е``н``и``е`]{.underline}

[`    ``trained_model.save_file("mnist-out/model", &burn::record::FullRecorder::new()).expect("Save failed");`]{.underline}

[`}`]{.underline}

[Полные примеры всегда доступны в репозитории Burn: ]{.underline}

**[В]{.underline}[е]{.underline}[с]{.underline}[а]{.underline}[
]{.underline}[м]{.underline}[о]{.underline}[д]{.underline}[е]{.underline}[л]{.underline}[и]{.underline}**

[Есть
]{.underline}**[в]{.underline}[а]{.underline}[ж]{.underline}[н]{.underline}[ы]{.underline}[й]{.underline}[
]{.underline}[н]{.underline}[ю]{.underline}[а]{.underline}[н]{.underline}[с]{.underline}**[:
]{.underline}**[п]{.underline}[у]{.underline}[б]{.underline}[л]{.underline}[и]{.underline}[ч]{.underline}[н]{.underline}[о]{.underline}[
]{.underline}[д]{.underline}[о]{.underline}[с]{.underline}[т]{.underline}[у]{.underline}[п]{.underline}[н]{.underline}[ы]{.underline}[х]{.underline}[
]{.underline}[п]{.underline}[р]{.underline}[е]{.underline}[д]{.underline}[о]{.underline}[б]{.underline}[у]{.underline}[ч]{.underline}[е]{.underline}[н]{.underline}[н]{.underline}[ы]{.underline}[х]{.underline}[
]{.underline}[м]{.underline}[о]{.underline}[д]{.underline}[е]{.underline}[л]{.underline}[е]{.underline}[й]{.underline}[
]{.underline}[Н]{.underline}[Е]{.underline}[
]{.underline}[С]{.underline}[У]{.underline}[Щ]{.underline}[Е]{.underline}[С]{.underline}[Т]{.underline}[В]{.underline}[У]{.underline}[Е]{.underline}[Т]{.underline}**[.
Если речь идет о узкоспециализированной задаче. Давайте разберём все
аспекты:]{.underline}

### [�� Что такое \"веса модели\" в контексте ML?]{.underline}

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ ------------------------------------------------------------------------------------------------------------------------------ ---------------------------------------------------------------------------------------------------------------------------------------------
  **[К]{.underline}[о]{.underline}[м]{.underline}[п]{.underline}[о]{.underline}[н]{.underline}[е]{.underline}[н]{.underline}[т]{.underline}**                                                                                                                                          **[А]{.underline}[н]{.underline}[а]{.underline}[л]{.underline}[о]{.underline}[г]{.underline}[и]{.underline}[я]{.underline}**   **[В]{.underline}[ ]{.underline}[п]{.underline}[р]{.underline}[о]{.underline}[е]{.underline}[к]{.underline}[т]{.underline}[е]{.underline}**
  **[А]{.underline}[р]{.underline}[х]{.underline}[и]{.underline}[т]{.underline}[е]{.underline}[к]{.underline}[т]{.underline}[у]{.underline}[р]{.underline}[а]{.underline}[ ]{.underline}[м]{.underline}[о]{.underline}[д]{.underline}[е]{.underline}[л]{.underline}[и]{.underline}**   [Каркас здания (стены, перекрытия)]{.underline}                                                                                [`nn::Sequential` с линейными слоями в `inference.rs`]{.underline}
  **[В]{.underline}[е]{.underline}[с]{.underline}[а]{.underline}[ ]{.underline}[м]{.underline}[о]{.underline}[д]{.underline}[е]{.underline}[л]{.underline}[и]{.underline}**                                                                                                            [Отделка, мебель, проводка]{.underline}                                                                                        [Файл `rustfix_v2.pt` --- числовые параметры, обученные на данных]{.underline}
  **[И]{.underline}[н]{.underline}[ф]{.underline}[е]{.underline}[р]{.underline}[е]{.underline}[н]{.underline}[с]{.underline}**                                                                                                                                                         [Жильцы используют готовое здание]{.underline}                                                                                 [`model.forward(&input)` --- применение весов к новому коду]{.underline}
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ ------------------------------------------------------------------------------------------------------------------------------ ---------------------------------------------------------------------------------------------------------------------------------------------

[Веса --- это
]{.underline}**[б]{.underline}[и]{.underline}[н]{.underline}[а]{.underline}[р]{.underline}[н]{.underline}[ы]{.underline}[й]{.underline}[
]{.underline}[ф]{.underline}[а]{.underline}[й]{.underline}[л]{.underline}**[
(обычно `.pt` для PyTorch), содержащий миллионы чисел --- результат
обучения на датасете.]{.underline}

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------
  **[П]{.underline}[р]{.underline}[о]{.underline}[б]{.underline}[л]{.underline}[е]{.underline}[м]{.underline}[а]{.underline}**                                                                                                                                                                                                                                    **[О]{.underline}[б]{.underline}[ъ]{.underline}[я]{.underline}[с]{.underline}[н]{.underline}[е]{.underline}[н]{.underline}[и]{.underline}[е]{.underline}**
  **[Н]{.underline}[е]{.underline}[т]{.underline}[ ]{.underline}[п]{.underline}[у]{.underline}[б]{.underline}[л]{.underline}[и]{.underline}[ч]{.underline}[н]{.underline}[ы]{.underline}[х]{.underline}[ ]{.underline}[м]{.underline}[о]{.underline}[д]{.underline}[е]{.underline}[л]{.underline}[е]{.underline}[й]{.underline}**                                 [Никто не публиковал модели специально для конкретной задачи (в отличие от, например, код-генерации)]{.underline}
  **[С]{.underline}[п]{.underline}[е]{.underline}[ц]{.underline}[и]{.underline}[ф]{.underline}[и]{.underline}[ч]{.underline}[н]{.underline}[о]{.underline}[с]{.underline}[т]{.underline}[ь]{.underline}[ ]{.underline}[з]{.underline}[а]{.underline}[д]{.underline}[а]{.underline}[ч]{.underline}[и]{.underline}**                                                [Модель должна понимать AST Rust, borrow checker, lifetime system --- это требует кастомной архитектуры]{.underline}
  **[Ю]{.underline}[р]{.underline}[и]{.underline}[д]{.underline}[и]{.underline}[ч]{.underline}[е]{.underline}[с]{.underline}[к]{.underline}[и]{.underline}[е]{.underline}[ ]{.underline}[о]{.underline}[г]{.underline}[р]{.underline}[а]{.underline}[н]{.underline}[и]{.underline}[ч]{.underline}[е]{.underline}[н]{.underline}[и]{.underline}[я]{.underline}**   [Датасеты из реального кода часто содержат проприетарный код --- их нельзя публиковать свободно]{.underline}
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------

### [✅ Практические пути получения весов]{.underline}

#### [Вариант 1: Использовать моковую модель (рекомендуется для старта)]{.underline}

[`# ``С``о``з``д``а``й``т``е`` ``з``а``г``л``у``ш``к``у``-``в``е``с``а`` (0 ``б``а``й``т`` ``—`` ``м``о``д``е``л``ь`` ``в``с``ё`` ``р``а``в``н``о`` ``н``е`` ``и``с``п``о``л``ь``з``у``е``т``с``я`` ``в`` ``м``о``к``о``в``о``м`` ``р``е``ж``и``м``е``)`]{.underline}

[`touch weights/rustfix_v2.pt`]{.underline}

[`# ``И``л``и`` ``с``о``з``д``а``й``т``е`` ``м``и``н``и``м``а``л``ь``н``ы``й`` ``в``а``л``и``д``н``ы``й`` ``ф``а``й``л`` ``д``л``я`` ``т``е``с``т``о``в`]{.underline}

[`python3 << 'EOF'`]{.underline}

[`import torch`]{.underline}

[`import torch.nn as nn`]{.underline}

[`model = nn.Sequential(`]{.underline}

[`    ``nn.Linear(512, 256),`]{.underline}

[`    ``nn.ReLU(),`]{.underline}

[`    ``nn.Linear(256, 128),`]{.underline}

[`    ``nn.ReLU(),`]{.underline}

[`    ``nn.Linear(128, 3)`]{.underline}

[`)`]{.underline}

[`torch.save(model.state_dict(), 'weights/rustfix_v2.pt')`]{.underline}

[`print("``✅`` Created mock weights (1.2 MB)")`]{.underline}

[`EOF`]{.underline}

**[П]{.underline}[л]{.underline}[ю]{.underline}[с]{.underline}[ы]{.underline}**[:]{.underline}\
[✅ Мгновенная настройка]{.underline}\
[✅ Не требует GPU/обучения]{.underline}\
[✅ Демонстрирует архитектуру проекта]{.underline}

**[М]{.underline}[и]{.underline}[н]{.underline}[у]{.underline}[с]{.underline}[ы]{.underline}**[:]{.underline}\
[❌ Не даёт реальных предсказаний - только эвристики]{.underline}

#### [Вариант 2: Обучить модель самостоятельно (реалистичный путь)]{.underline}

##### [Шаг 1: Подготовьте датасет]{.underline}

[`# ml/training/generate_dataset.py`]{.underline}

[`import json`]{.underline}

[`from pathlib import Path`]{.underline}

[`import re`]{.underline}

[`def extract_error_fix_pairs(repo_path):`]{.underline}

[`    ``"""``И``з``в``л``е``к``а``е``т`` ``п``а``р``ы`` (``о``ш``и``б``к``а`` ``→`` ``и``с``п``р``а``в``л``е``н``и``е``) ``и``з`` ``и``с``т``о``р``и``и`` ``к``о``м``м``и``т``о``в``"""`]{.underline}

[`    ``pairs = []`]{.underline}

[`    ``# ``П``р``и``м``е``р``: ``н``а``й``т``и`` ``к``о``м``м``и``т``ы`` ``с`` ``и``с``п``р``а``в``л``е``н``и``е``м`` "use of moved value"`]{.underline}

[`    ``pattern = r'let``\``s+(``\``w+)``\``s*=``\``s*(``\``w+);.*?println!``\``(.*?``\``1``\``)'`]{.underline}

[`    ``for rs_file in Path(repo_path).rglob("*.rs"):`]{.underline}

[`        ``content = rs_file.read_text()`]{.underline}

[`        ``if re.search(pattern, content, re.DOTALL):`]{.underline}

[`            ``# ``Н``а``й``т``и`` ``и``с``п``р``а``в``л``е``н``н``у``ю`` ``в``е``р``с``и``ю`` ``в`` ``и``с``т``о``р``и``и`` git`]{.underline}

[`            ``fixed = content.replace(f"println!(``\``"``{``{``}``}``\``", ``{``pattern``}``)", `]{.underline}

[`                                   ``f"println!(``\``"``{``{``}``}``\``", ``{``pattern``}``.clone())")`]{.underline}

[`            ``pairs.append(``{`]{.underline}

[`                ``"file": str(rs_file),`]{.underline}

[`                ``"original": content,`]{.underline}

[`                ``"fixed": fixed,`]{.underline}

[`                ``"error_type": "move_semantics",`]{.underline}

[`                ``"confidence": 0.95`]{.underline}

[`            ``}``)`]{.underline}

[`    ``return pairs`]{.underline}

[`# ``С``о``б``е``р``и``т``е`` 10``к``+ ``п``р``и``м``е``р``о``в`` ``и``з`` ``о``т``к``р``ы``т``ы``х`` ``р``е``п``о``з``и``т``о``р``и``е``в`]{.underline}

[`repos = [`]{.underline}

[`    ``"https://github.com/rust-lang/rust",`]{.underline}

[`    ``"https://github.com/tokio-rs/tokio",`]{.underline}

[`    ``"https://github.com/serde-rs/serde",`]{.underline}

[`    ``# ... ``е``щ``ё`` 50 ``п``о``п``у``л``я``р``н``ы``х`` ``р``е``п``о``з``и``т``о``р``и``е``в`]{.underline}

[`]`]{.underline}

[`dataset = []`]{.underline}

[`for repo in repos:`]{.underline}

[`    ``dataset.extend(extract_error_fix_pairs(clone_repo(repo)))`]{.underline}

[`with open("datasets/rust_errors_v2.jsonl", "w") as f:`]{.underline}

[`    ``for item in dataset:`]{.underline}

[`        ``f.write(json.dumps(item) + "``\``n")`]{.underline}

[`print(f"``✅`` ``С``о``б``р``а``н``о`` ``{``len(dataset)``}`` ``п``р``и``м``е``р``о``в``")`]{.underline}

##### [Шаг 2: Обучите модель]{.underline}

[`# ml/training/train.py`]{.underline}

[`import torch`]{.underline}

[`from torch.utils.data import Dataset, DataLoader`]{.underline}

[`from transformers import RobertaTokenizer`]{.underline}

[`class RustErrorDataset(Dataset):`]{.underline}

[`    ``def __init__(self, jsonl_path):`]{.underline}

[`        ``self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")`]{.underline}

[`        ``self.items = [json.loads(line) for line in open(jsonl_path)]`]{.underline}

[`    ``def __getitem__(self, idx):`]{.underline}

[`        ``item = self.items[idx]`]{.underline}

[`        ``# ``Т``о``к``е``н``и``з``и``р``у``е``м`` ``о``ш``и``б``о``ч``н``ы``й`` ``к``о``д`]{.underline}

[`        ``tokens = self.tokenizer(item["original"], return_tensors="pt", max_length=512, truncation=True)`]{.underline}

[`        ``# ``М``е``т``к``а``: ``к``л``а``с``с`` ``о``ш``и``б``к``и`` (0=move, 1=lifetime, 2=trait)`]{.underline}

[`        ``label = ``{``"move_semantics": 0, "lifetime": 1, "trait": 2``}``[item["error_type"]]`]{.underline}

[`        ``return tokens["input_ids"].squeeze(), torch.tensor(label)`]{.underline}

[`# ``А``р``х``и``т``е``к``т``у``р``а``: CodeBERT + ``к``л``а``с``с``и``ф``и``к``а``т``о``р`]{.underline}

[`from transformers import RobertaModel`]{.underline}

[`class RustFixModel(torch.nn.Module):`]{.underline}

[`    ``def __init__(self):`]{.underline}

[`        ``super().__init__()`]{.underline}

[`        ``self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")`]{.underline}

[`        ``self.classifier = torch.nn.Sequential(`]{.underline}

[`            ``torch.nn.Linear(768, 256),`]{.underline}

[`            ``torch.nn.ReLU(),`]{.underline}

[`            ``torch.nn.Linear(256, 3)  # 3 ``к``л``а``с``с``а`` ``о``ш``и``б``о``к`]{.underline}

[`        ``)`]{.underline}

[`    ``def forward(self, input_ids):`]{.underline}

[`        ``features = self.encoder(input_ids).last_hidden_state[:, 0, :]  # [CLS] token`]{.underline}

[`        ``return self.classifier(features)`]{.underline}

[`# ``О``б``у``ч``е``н``и``е`]{.underline}

[`model = RustFixModel()`]{.underline}

[`dataset = RustErrorDataset("datasets/rust_errors_v2.jsonl")`]{.underline}

[`loader = DataLoader(dataset, batch_size=16, shuffle=True)`]{.underline}

[`optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)`]{.underline}

[`for epoch in range(10):`]{.underline}

[`    ``for input_ids, labels in loader:`]{.underline}

[`        ``outputs = model(input_ids)`]{.underline}

[`        ``loss = torch.nn.functional.cross_entropy(outputs, labels)`]{.underline}

[`        ``loss.backward()`]{.underline}

[`        ``optimizer.step()`]{.underline}

[`        ``optimizer.zero_grad()`]{.underline}

[`# ``С``о``х``р``а``н``е``н``и``е`` ``в``е``с``о``в`]{.underline}

[`torch.save(model.state_dict(), "weights/rustfix_v2.pt")`]{.underline}

[`print("``✅`` ``М``о``д``е``л``ь`` ``о``б``у``ч``е``н``а`` ``и`` ``с``о``х``р``а``н``е``н``а``")`]{.underline}

**[Т]{.underline}[р]{.underline}[е]{.underline}[б]{.underline}[о]{.underline}[в]{.underline}[а]{.underline}[н]{.underline}[и]{.underline}[я]{.underline}**[:]{.underline}

-   [⏱ Время: 8-24 часа на среднем GPU (RTX 3090)]{.underline}

-   [�� Данные: 10к+ примеров ошибок/исправлений]{.underline}

-   [�� Стоимость: \~\$15 на Google Colab Pro для обучения]{.underline}

#### [Вариант 3: Интеграция с существующими инструментами (гибридный подход)]{.underline}

[Вместо кастомной модели используйте
]{.underline}**[о]{.underline}[ф]{.underline}[и]{.underline}[ц]{.underline}[и]{.underline}[а]{.underline}[л]{.underline}[ь]{.underline}[н]{.underline}[ы]{.underline}[е]{.underline}[
]{.underline}[и]{.underline}[н]{.underline}[с]{.underline}[т]{.underline}[р]{.underline}[у]{.underline}[м]{.underline}[е]{.underline}[н]{.underline}[т]{.underline}[ы]{.underline}[
Rust]{.underline}**[ :]{.underline}

[`// src/ml/hybrid.rs`]{.underline}

[`pub fn analyze_with_rustc(`]{.underline}

[`    ``path: &Path,`]{.underline}

[`) -> Result<Vec<AnalysisError>, Exn<AnalysisError>> ``{`]{.underline}

[`    ``// ``З``а``п``у``с``к``а``е``м`` rustc ``с`` ``в``ы``в``о``д``о``м`` ``в`` JSON`]{.underline}

[`    ``let output = std::process::Command::new("rustc")`]{.underline}

[`        ``.args(&[`]{.underline}

[`            ``"--crate-type", "lib",`]{.underline}

[`            ``"--error-format=json",`]{.underline}

[`            ``"--emit=metadata",`]{.underline}

[`            ``path.to_str().unwrap(),`]{.underline}

[`        ``])`]{.underline}

[`        ``.output()`]{.underline}

[`        ``.map_err(|e| Exn::new(AnalysisError::infrastructure(&format!(`]{.underline}

[`            ``"rustc invocation failed: ``{``}``",`]{.underline}

[`            ``e`]{.underline}

[`        ``))))?;`]{.underline}

[`    ``if !output.status.success() ``{`]{.underline}

[`        ``// ``П``а``р``с``и``м`` ``д``и``а``г``н``о``с``т``и``ч``е``с``к``и``е`` ``с``о``о``б``щ``е``н``и``я`]{.underline}

[`        ``let diagnostics: Vec<Diagnostic> = serde_json::from_slice(&output.stderr)`]{.underline}

[`            ``.map_err(|e| Exn::new(AnalysisError::infrastructure(&format!(`]{.underline}

[`                ``"Failed to parse rustc diagnostics: ``{``}``",`]{.underline}

[`                ``e`]{.underline}

[`            ``))))?;`]{.underline}

[`        ``let errors = diagnostics`]{.underline}

[`            ``.into_iter()`]{.underline}

[`            ``.filter(|d| d.level == "error")`]{.underline}

[`            ``.map(|d| ``{`]{.underline}

[`                ``AnalysisError ``{`]{.underline}

[`                    ``kind: AnalysisKind::CompilerDiagnostic,`]{.underline}

[`                    ``message: d.message,`]{.underline}

[`                    ``suggestion: d.suggestions.first().cloned(),`]{.underline}

[`                    ``line: Some(d.line_start as u32),`]{.underline}

[`                    ``column: Some(d.column_start as u32),`]{.underline}

[`                    ``confidence: 1.0, // rustc ``в``с``е``г``д``а`` ``т``о``ч``е``н`]{.underline}

[`                ``}`]{.underline}

[`            ``}``)`]{.underline}

[`            ``.collect();`]{.underline}

[`        ``return Ok(errors);`]{.underline}

[`    ``}`]{.underline}

[`    ``Ok(vec![])`]{.underline}

[`}`]{.underline}

**[П]{.underline}[р]{.underline}[е]{.underline}[и]{.underline}[м]{.underline}[у]{.underline}[щ]{.underline}[е]{.underline}[с]{.underline}[т]{.underline}[в]{.underline}[а]{.underline}[
]{.underline}[г]{.underline}[и]{.underline}[б]{.underline}[р]{.underline}[и]{.underline}[д]{.underline}[н]{.underline}[о]{.underline}[г]{.underline}[о]{.underline}[
]{.underline}[п]{.underline}[о]{.underline}[д]{.underline}[х]{.underline}[о]{.underline}[д]{.underline}[а]{.underline}**[:]{.underline}\
[✅ 100% точность для компиляторных ошибок]{.underline}\
[✅ Не требует обучения модели]{.underline}\
[✅ Работает \"из коробки\" с любым кодом]{.underline}\
[✅ `exn` добавляет контекст к диагностике rustc]{.underline}
