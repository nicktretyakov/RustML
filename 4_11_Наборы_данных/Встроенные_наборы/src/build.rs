//! Build script: находит breast_cancer.csv в нескольких местах и кладёт копию
//! в `OUT_DIR/breast_cancer.csv`, чтобы `dataset.rs` мог сделать
//! `include_str!(concat!(env!("OUT_DIR"), "/breast_cancer.csv"))` независимо от
//! того, где именно лежит CSV в дереве проекта.
//!
//! Поиск ведётся по порядку (первый найденный файл используется):
//!   1. `CARGO_MANIFEST_DIR/data/breast_cancer.csv`   (наш layout в download/)
//!   2. `CARGO_MANIFEST_DIR/breast_cancer.csv`       (CSV рядом с Cargo.toml)
//!   3. `CARGO_MANIFEST_DIR/src/../breast_cancer.csv` (рядом со src/, как у пользователя)
//!   4. `/home/z/my-project/data/breast_cancer.csv`  (sandbox-путь)
//!   5. `/home/z/my-project/src_rust/data/breast_cancer.csv`
//!   6. `/home/z/my-project/download/breast_cancer_app/breast_cancer.csv`
//!   7. `/home/z/my-project/download/breast_cancer_app/data/breast_cancer.csv`
//!   8. `/home/z/my-project/download/breast_cancer_app/src/breast_cancer.csv`
//!
//! Если ни один путь не подходит — build падает с понятным сообщением.
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    // Перекомпилируем build script, если CSV в кандидатах изменится.
    println!("cargo:rerun-if-changed=data/breast_cancer.csv");
    println!("cargo:rerun-if-changed=breast_cancer.csv");

    let manifest = env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR должен быть задан cargo");
    let manifest = PathBuf::from(manifest);

    let candidates: Vec<PathBuf> = vec![
        manifest.join("data").join("breast_cancer.csv"),
        manifest.join("breast_cancer.csv"),
        manifest.join("src").join("..").join("breast_cancer.csv"),
        PathBuf::from("/home/z/my-project/data/breast_cancer.csv"),
        PathBuf::from("/home/z/my-project/src_rust/data/breast_cancer.csv"),
        PathBuf::from("/home/z/my-project/download/breast_cancer_app/breast_cancer.csv"),
        PathBuf::from("/home/z/my-project/download/breast_cancer_app/data/breast_cancer.csv"),
        PathBuf::from("/home/z/my-project/download/breast_cancer_app/src/breast_cancer.csv"),
    ];

    let src = candidates
        .iter()
        .find(|p| p.exists())
        .cloned()
        .unwrap_or_else(|| {
            let list: Vec<String> = candidates
                .iter()
                .map(|p| format!("  - {}", p.display()))
                .collect();
            panic!(
                "build.rs: не найден breast_cancer.csv ни в одном из мест:\n{}\n\
                 Положите breast_cancer.csv рядом с Cargo.toml или в data/ и пересоберите.",
                list.join("\n")
            )
        });

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR должен быть задан cargo"));
    let dst = out_dir.join("breast_cancer.csv");
    fs::copy(&src, &dst).unwrap_or_else(|e| {
        panic!(
            "build.rs: не удалось скопировать {} в {}: {}",
            src.display(),
            dst.display(),
            e
        )
    });

    eprintln!(
        "build.rs: embedded breast_cancer.csv <- {}",
        src.display()
    );
}
