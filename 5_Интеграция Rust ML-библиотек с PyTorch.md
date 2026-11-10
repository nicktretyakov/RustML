# Способы интеграции

В экосистеме Rust для ML (особенно Burn) интеграция с PyTorch реализуется несколькими способами. Актуальный подход — импорт предобученных весов из PyTorch, а не прямой runtime-зависимости от libtorch (хотя она возможна). Это обеспечивает переносимость: модель из PyTorch запускается на native Rust-бэкендах Burn (NdArray, WGPU и т.д.) без overhead.
Ранее популярный backend burn-tch (на базе tch-rs и libtorch) всё ещё существует как отдельный crate, но в основной документации Burn акцент сместился на импорт весов и native execution. Прямой LibTorch-backend реже используется из-за зависимости от тяжёлой C++ библиотеки.

**1. Основной способ: Импорт весов из PyTorch в Burn (рекомендуется)**
Используйте crate burn-import для загрузки .pth файлов (стандартный формат PyTorch). Это конвертирует веса в Burn-формат (record), совместимый с любым бэкендом.

**Зависимости в Cargo.toml:**

```
[dependencies]
burn = { version = "0.13.0", features = ["ndarray"] }  # или "wgpu" для GPU
burn-import = "0.13.0"
```


**Пример кода: Загрузка предобученной ResNet-18 из PyTorch**
(Адаптировано из официального блога Burn.
```
use burn::{
    prelude::*,
    module::Module,
    config::Config,
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn::record::FullPrecisionSettings;

// Пример простой модели (полный ResNet в examples Burn)
#[derive(Module, Debug)]
pub struct SimpleModel<B: Backend> {
    // ... слои
}

#[derive(Config)]
pub struct SimpleModelConfig {
    // ... параметры
}

impl SimpleModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SimpleModel<B> {
        // ... инициализация
    }
}

fn main() {
    type Backend = burn::backend::NdArray;  // Или Wgpu<f32, i32> для GPU
    // type Backend = burn::backend::Wgpu;

    let device = Backend::Device::default();

    // Конфиг и пустая модель
    let config = SimpleModelConfig::new();  // ваши параметры
    let mut model: SimpleModel<Backend> = config.init(&device);

    // Загрузка весов из PyTorch .pth
    let load_args = LoadArgs::new("resnet18-f37072fd.pth".to_string())
        .with_key_remap("(.+)\\.downsample\\.0\\.(.+)", "$1.downsample.conv.$2")  // remap для совместимости имён
        .with_key_remap("(.+)\\.downsample\\.1\\.(.+)", "$1.downsample.bn.$2")
        .with_key_remap("(layer[1-4])\\.([0-9]+)\\.(.+)", "$1.blocks.$2.$3");  // если структура отличается

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(load_args, &device)
        .expect("Не удалось загрузить веса PyTorch");

    // Применяем веса к модели
    model = model.load_record(record);

    // Теперь model готова к inference на выбранном бэкенде
    println!("Модель с весами PyTorch загружена: {model:?}");
}

```

**Преимущества:**

    • Нет runtime-зависимости от PyTorch.
    • Работает на WGPU (Vulkan/Metal/DirectX) для кросс-платформенного GPU.
    • Легко для HF-моделей (экспортируйте в .pth).

Для Hugging Face моделей часто используют safetensors (совместим с PyTorch): Burn поддерживает загрузку через BinFileRecorder или custom код.

**2. Прямые bindings: tch-rs + libtorch**
Если нужна максимальная совместимость (CUDA, прямой jit-load TorchScript моделей), используйте tch-rs — Rust-bindings к C++ libtorch.

**Зависимости:**

```
[dependencies]
tch = "0.15.0"  # или актуальная
```

**Установка libtorch:** Скачайте с pytorch.org (CUDA/CPU версия) и укажите переменные окружения (LIBTORCH, LD_LIBRARY_PATH).
**Пример: Загрузка и inference TorchScript модели**


```
use tch::{Tensor, Device, jit::CModule, TchResult};

fn main() -> TchResult<()> {
    // Устройство (CUDA если доступно)
    let device = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    // Загрузка модели (экспортирована в TorchScript: torch.jit.trace/script в Python)
    let mut model = CModule::load_on_device("model.pt", device)?;

    // Входные данные
    let input = Tensor::randn(&[1, 3, 224, 224], (tch::Kind::Float, device));

    // Inference (no_grad)
    let output = tch::no_grad(|| model.forward_is(&[input]));

    println!("Output shape: {:?}", output.size());

    Ok(())
}
```

**Преимущества:** 

Полный доступ к PyTorch kernels (CUDA, оптимизации).

Недостатки: Тяжёлая зависимость, сложная настройка.

**3. Другие варианты**

    • **Candle** (альтернатива Burn): Отлично загружает safetensors/GGUF от PyTorch/HF моделей (LLaMA, Whisper). Простой API, нет autodiff по умолчанию.

    • **ONNX/ Tract:** Экспорт из PyTorch в ONNX, затем runtime в Rust (tract crate).

Для большинства случаев (inference предобученных моделей) рекомендую **Burn + burn-import или Candle + safetensors** —чистый Rust, высокая производительность.
