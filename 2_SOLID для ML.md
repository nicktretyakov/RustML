**Single Responsibility Principle (SRP)**

**`Ч``т``о`` ``э``т``о`` ``о``з``н``а``ч``а``е``т`` ``в`` ML:`**` ``Р``а``з``д``е``л``я``й``т``е`` ``э``т``а``п``ы`` ``п``а``й``п``л``а``й``н``а``: ``о``т``д``е``л``ь``н``ы``й`` ``м``о``д``у``л``ь`` ``д``л``я`` ``з``а``г``р``у``з``к``и`` ``д``а``н``н``ы``х``, ``о``т``д``е``л``ь``н``ы``й`` ``—`` ``д``л``я`` ``п``р``е``д``о``б``р``а``б``о``т``к``и``, ``о``т``д``е``л``ь``н``ы``й`` ``—`` ``д``л``я`` ``о``б``у``ч``е``н``и``я`` ``м``о``д``е``л``и``, ``о``т``д``е``л``ь``н``ы``й`` ``—`` ``д``л``я`` ``в``а``л``и``д``а``ц``и``и``/``и``н``ф``е``р``е``н``с``а``. ``Э``т``о`` ``у``п``р``о``щ``а``е``т`` ``т``е``с``т``и``р``о``в``а``н``и``е`` ``и`` ``о``т``л``а``д``к``у`` (``н``а``п``р``и``м``е``р``, ``е``с``л``и`` ``м``е``н``я``е``т``с``я`` ``и``с``т``о``ч``н``и``к`` ``д``а``н``н``ы``х``, ``м``е``н``я``е``т``с``я`` ``т``о``л``ь``к``о`` ``о``д``и``н`` ``м``о``д``у``л``ь``).`

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

**`Ч``т``о`` ``э``т``о`` ``о``з``н``а``ч``а``е``т`` ``в`` ML:`**` ``Б``а``з``о``в``ы``й`` ``т``р``е``й``т`` ``м``о``д``е``л``и`` ``и``л``и`` ``п``а``й``п``л``а``й``н``а`` ``д``о``л``ж``е``н`` ``п``о``з``в``о``л``я``т``ь`` ``д``о``б``а``в``л``я``т``ь`` ``н``о``в``ы``е`` ``м``о``д``е``л``и``/``т``р``а``н``с``ф``о``р``м``а``ц``и``и`` ``ч``е``р``е``з`` ``н``а``с``л``е``д``о``в``а``н``и``е`` ``и``л``и`` ``к``о``м``п``о``з``и``ц``и``ю``, ``н``е`` ``м``е``н``я``я`` ``с``у``щ``е``с``т``в``у``ю``щ``и``й`` ``к``о``д``. ``Н``а``п``р``и``м``е``р``, ``б``а``з``о``в``ы``й`` ``т``р``е``й``т`` Trainer ``с`` ``м``е``т``о``д``о``м`` fit(), ``к``о``т``о``р``ы``й`` ``м``о``ж``н``о`` ``р``а``с``ш``и``р``я``т``ь`` ``д``л``я`` ``н``о``в``ы``х`` ``а``л``г``о``р``и``т``м``о``в`` (XGBoost, NeuralNet), ``н``е`` ``т``р``о``г``а``я`` ``о``с``н``о``в``н``о``й`` ``к``о``д``.`

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

**`Ч``т``о`` ``э``т``о`` ``о``з``н``а``ч``а``е``т`` ``в`` ML:`**` ``Е``с``л``и`` ``у`` ``в``а``с`` ``е``с``т``ь`` ``б``а``з``о``в``ы``й`` ``т``р``е``й``т`` Model ``с`` ``м``е``т``о``д``о``м`` predict(), ``т``о`` ``л``ю``б``о``й`` ``н``а``с``л``е``д``н``и``к`` (``н``а``п``р``и``м``е``р``, LinearRegressionModel ``и``л``и`` RandomForestModel) ``д``о``л``ж``е``н`` ``в``о``з``в``р``а``щ``а``т``ь`` ``п``р``е``д``с``к``а``з``а``н``и``я`` ``в`` ``т``о``м`` ``ж``е`` ``ф``о``р``м``а``т``е`` ``и`` ``с`` ``т``е``м`` ``ж``е`` ``п``о``в``е``д``е``н``и``е``м`` (``н``е`` ``б``р``о``с``а``т``ь`` ``н``е``о``ж``и``д``а``н``н``ы``е`` ``и``с``к``л``ю``ч``е``н``и``я``, ``н``е`` ``м``е``н``я``т``ь`` ``т``и``п`` ``в``ы``в``о``д``а``).`

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

**`Ч``т``о`` ``э``т``о`` ``о``з``н``а``ч``а``е``т`` ``в`` ML:`**` ``В``м``е``с``т``о`` ``о``д``н``о``г``о`` ``о``г``р``о``м``н``о``г``о`` ``т``р``е``й``т``а`` MLPipeline ``с`` ``м``е``т``о``д``а``м``и`` ``д``л``я`` ``в``с``е``г``о`` (load, preprocess, train, evaluate, deploy) ``с``о``з``д``а``в``а``й``т``е`` ``о``т``д``е``л``ь``н``ы``е`` ``т``р``е``й``т``ы``: DataLoader, Preprocessor, Trainer, Evaluator. ``Э``т``о`` ``о``с``о``б``е``н``н``о`` ``п``о``л``е``з``н``о`` ``в`` ``б``о``л``ь``ш``и``х`` ``к``о``м``а``н``д``а``х``, ``г``д``е`` ``р``а``з``н``ы``е`` ``л``ю``д``и`` ``р``а``б``о``т``а``ю``т`` ``с`` ``р``а``з``н``ы``м``и`` ``ч``а``с``т``я``м``и``.`

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

**`Ч``т``о`` ``э``т``о`` ``о``з``н``а``ч``а``е``т`` ``в`` ML:`**` ``В``м``е``с``т``о`` ``ж``ё``с``т``к``о``й`` ``п``р``и``в``я``з``к``и`` ``к`` ``к``о``н``к``р``е``т``н``о``й`` ``б``и``б``л``и``о``т``е``к``е`` (``н``а``п``р``и``м``е``р``, sklearn.LinearRegression) ``и``с``п``о``л``ь``з``у``й``т``е`` ``а``б``с``т``р``а``к``ц``и``и`` ``и`` dependency injection. ``Н``а``п``р``и``м``е``р``, ``п``е``р``е``д``а``в``а``й``т``е`` ``м``о``д``е``л``ь`` ``и``л``и`` ``д``а``т``а``л``о``а``д``е``р`` ``к``а``к`` ``п``а``р``а``м``е``т``р`` ``в`` ``к``о``н``с``т``р``у``к``т``о``р`` ``п``а``й``п``л``а``й``н``а`` (``ч``а``с``т``о`` ``р``е``а``л``и``з``у``е``т``с``я`` ``ч``е``р``е``з`` ``к``о``н``ф``и``г``и`` ``и``л``и`` ``ф``р``е``й``м``в``о``р``к``и`` ``в``р``о``д``е`` ZenML, Metaflow, MLflow).`

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
