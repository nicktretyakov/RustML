**Single Responsibility Principle (SRP)**

Разделяйте этапы пайплайна: отдельный модуль для загрузки данных, отдельный — для предобработки, отдельный — для обучения модели, отдельный — для валидации/инференса. Это упрощает тестирование и отладку (например, если меняется источник данных, меняется только один модуль).

    Плохой пример (одна структура делает всё):

    struct BadMlPipeline {

        // данные и модель внутри

    }

    impl BadMlPipeline {

        fn load_data(&self, path: &str) { /* загрузка */ }

        fn preprocess(&mut self, data: Array2<f64>) { /* предобработка */ }

        fn train(&mut self, x: Array2<f64>, y: Array1<f64>) { /* обучение */ }

        fn predict(&self, x: Array2<f64>) -> Array1<f64> { /* предсказание */ }

    }

    Хороший пример (каждая структура — одна ответственность):

    use ndarray::{Array1, Array2};

    struct DataLoader;

    impl DataLoader {

        fn load(&self, path: &str) -> Array2<f64> {

            // только загрузка данных

            unimplemented!()

        }

    }

    struct Preprocessor;

    impl Preprocessor {

        fn transform(&self, data: Array2<f64>) -> Array2<f64> {

            // только предобработка

            unimplemented!()

        }

    }

    struct Trainer;

    impl Trainer {

        fn fit(&self, x: Array2<f64>, y: Array1<f64>) -> Box<dyn Model> {

            // только обучение

            unimplemented!()

        }

    }

    trait Model {

        fn predict(&self, x: Array2<f64>) -> Array1<f64>;

    }

**Open-Closed Principle (OCP)**

Базовый трейт модели или пайплайна должен позволять добавлять новые модели/трансформации через наследование или композицию, не меняя существующий код. Например, базовый трейт Trainer с методом fit(), который можно расширять для новых алгоритмов (XGBoost, NeuralNet), не трогая основной код.

    Плохой пример (чтобы добавить новую модель, меняем существующий код):

    enum ModelType { Linear, Tree }

    struct BadTrainer;

    impl BadTrainer {

        fn train(&self, model_type: ModelType, x: Array2<f64>, y: Array1<f64>) -> Box<dyn Model> {

            match model_type {

                ModelType::Linear => {

                    // обучить линейную регрессию

                }

                ModelType::Tree => {

                    // обучить дерево (добавили бы ещё один матч)

                }

            }

            unimplemented!()

        }

    }

    Хороший пример (расширение через новые реализации трейта, без изменения базового кода):

    trait ModelFactory {

        fn create(&self) -> Box<dyn Model>;

        fn train(&self, x: Array2<f64>, y: Array1<f64>) -> Box<dyn Model> {

            let mut model = self.create();

            // общая логика обучения (например, градиентный спуск)

            // ...

            model

        }

    }

    struct LinearRegressionFactory;

    impl ModelFactory for LinearRegressionFactory {

        fn create(&self) -> Box<dyn Model> {

            Box::new(LinearRegression::default())

        }

    }

    // Добавляем новую модель — просто новая структура

    struct DecisionTreeFactory;

    impl ModelFactory for DecisionTreeFactory {

        fn create(&self) -> Box<dyn Model> {

            Box::new(DecisionTree::default())

        }

    }

**Liskov Substitution Principle (LSP)**

Если у вас есть базовый трейт Model с методом predict(), то любой наследник (например, LinearRegressionModel или RandomForestModel) должен возвращать предсказания в том же формате и с тем же поведением (не бросать неожиданные исключения, не менять тип вывода).

    Плохой пример (подтип меняет контракт — возвращает другой тип):

    trait BaseModel {

        fn predict(&self, x: Array2<f64>) -> Array1<f64>; // всегда ndarray

    }

    struct BadNeuralNet;

    impl BaseModel for BadNeuralNet {

        fn predict(&self, x: Array2<f64>) -> Array1<f64> {

            // возвращаем Vec<f64> вместо Array1 — нарушение!

            unimplemented!()

        }

    }

    Хороший пример (все реализации сохраняют одинаковый контракт):

    trait Model {

        fn predict(&self, x: &Array2<f64>) -> Array1<f64>;

    }

    struct LinearRegression { weights: Array1<f64> }

    impl Model for LinearRegression {

        fn predict(&self, x: &Array2<f64>) -> Array1<f64> {

            x.dot(&self.weights)

        }

    }

    struct TorchModel { /* обёртка над tch-rs */ }

    impl Model for TorchModel {

        fn predict(&self, x: &Array2<f64>) -> Array1<f64> {

            // конвертируем в тензор, предсказываем, возвращаем Array1<f64>

            unimplemented!()

        }

    }

**Interface Segregation Principle (ISP)**

Вместо одного огромного трейта MLPipeline с методами для всего (load, preprocess, train, evaluate, deploy) создавайте отдельные трейты: DataLoader, Preprocessor, Trainer, Evaluator. Это особенно полезно в больших командах, где разные люди работают с разными частями.

    Плохой пример («жирный» трейт заставляет реализовывать ненужное):

    trait BigMlTrait {

        fn load_data(&self);

        fn preprocess(&self);

        fn train(&self);

        fn evaluate(&self);

        fn deploy(&self);

    }

    // Маленький компонент вынужден реализовывать всё

    struct SimplePreprocessor;

    impl BigMlTrait for SimplePreprocessor {

        fn load_data(&self) { unimplemented!() }

        // ...

    }

    Хороший пример (мелкие специализированные трейты):

    trait DataLoaderTrait {

        fn load(&self, path: &str) -> Array2<f64>;

    }

    trait PreprocessorTrait {

        fn transform(&self, data: Array2<f64>) -> Array2<f64>;

    }

    trait TrainerTrait {

        fn fit(&self, x: Array2<f64>, y: Array1<f64>) -> Box<dyn Model>;

    }

    trait EvaluatorTrait {

        fn evaluate(&self, model: &dyn Model, x: Array2<f64>, y: Array1<f64>) -> f64;

    }

    // Теперь каждый компонент реализует только нужное

    struct MyPreprocessor;

    impl PreprocessorTrait for MyPreprocessor {

        fn transform(&self, data: Array2<f64>) -> Array2<f64> {

            unimplemented!()

        }

    }

**Dependency Inversion Principle (DIP)**

Вместо жёсткой привязки к конкретной библиотеке (например, sklearn.LinearRegression) используйте абстракции и dependency injection. Например, передавайте модель или даталоадер как параметр в конструктор пайплайна (часто реализуется через конфиги или фреймворки вроде ZenML, Metaflow, MLflow).

    Плохой пример (высокоуровневый модуль зависит от конкретной реализации):

    struct LinearRegression; // конкретная модель

    struct BadPipeline {

        model: LinearRegression, // жёстко привязаны к одной модели

    }

    Хороший пример (зависимость от абстракции + инъекция через трейт):

    trait Model: Send + Sync {

        fn predict(&self, x: &Array2<f64>) -> Array1<f64>;

    }

    struct MlPipeline<M: Model> {

        model: M, // generic над трейтом

    }

    impl<M: Model> MlPipeline<M> {

        fn new(model: M) -> Self {

            Self { model }

        }

        fn run(&self, x: &Array2<f64>) -> Array1<f64> {

            self.model.predict(x)

        }

    }

    // Использование:

    fn main() {

        let linear = LinearRegression { weights: Array1::zeros(10) };

        let pipeline = MlPipeline::new(linear);

        // Можно передать любую другую реализацию Model

        // let tree = DecisionTree { ... };

        // let pipeline = MlPipeline::new(tree);

    }
