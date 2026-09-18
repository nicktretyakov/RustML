// =============================================
// + Model serialization (save / load)
// =============================================

use cat_rl_solid::{
    CatEnv, CatPolicyNetwork, Device, DType, ModelArchitecture, ModelConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Cat RL Solid + Сериализация и запись моделей ===\n");

    // ------------------------------------------------------------------
    // 1. Создаём и сохраняем
    // ------------------------------------------------------------------
    println!("--- 1. Создание и save ---");
    let config = ModelConfig {
        architecture: ModelArchitecture::SimpleMLP,
        dtype: DType::Float32,
        device: Device::Cpu,
        hyperparams: cat_rl_solid::Hyperparams::default(),
    };
    let mut nn = CatPolicyNetwork::from_config(config);

    let env = CatEnv::random();
    let before = nn.forward_multi_task(&env.observation())?;
    println!(
        "  До сохранения: policy=[{:.4}, {:.4}]",
        before.policy[0], before.policy[1]
    );

    nn.save("cat_model.model")?;

    // ------------------------------------------------------------------
    // 2. Загружаем в новую сеть
    // ------------------------------------------------------------------
    println!("\n--- 2. load ---");
    let mut cfg2 = ModelConfig::default();
    cfg2.architecture = ModelArchitecture::SimpleMLP;
    cfg2.dtype = DType::Float32;
    let mut nn2 = CatPolicyNetwork::from_config(cfg2);

    nn2.load("cat_model.model")?;

    let after = nn2.forward_multi_task(&env.observation())?;
    println!(
        "  После загрузки: policy=[{:.4}, {:.4}]",
        after.policy[0], after.policy[1]
    );

    // ------------------------------------------------------------------
    // 3. Проверка совпадения
    // ------------------------------------------------------------------
    println!("\n--- 3. Проверка ---");
    let diff = (before.policy[0] - after.policy[0]).abs()
        + (before.policy[1] - after.policy[1]).abs();
    if diff < 1e-4 {
        println!("  Веса совпадают ✅ (diff={:.6})", diff);
    } else {
        println!("  Веса отличаются (diff={:.6})", diff);
    }

    // ------------------------------------------------------------------
    // 4. Ошибка на несуществующем файле
    // ------------------------------------------------------------------
    println!("\n--- 4. Обработка ошибок ---");
    match nn2.load("no_such_model.model") {
        Ok(_) => println!("  Неожиданный успех"),
        Err(e) => println!("  Поймано: {}", e),
    }

    println!("\n🎉 Сериализация моделей успешно добавлена!");
    println!("   Файл: cat_model.model");
    Ok(())
}
