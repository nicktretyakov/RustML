//! Build script: bake the CPython shared-library directory into the binary's
//! rpath so `libpython3.X.so.1.0` is found at runtime without `LD_LIBRARY_PATH`.
//!
//! PyO3 links the interpreter's `libpython` dynamically. On systems where that
//! library lives outside the standard linker search path (pyenv, uv, conda, a
//! from-source build, ...), `cargo run` fails with:
//!
//!     error while loading shared libraries: libpython3.12.so.1.0:
//!     cannot open shared object file: No such file or directory
//!
//! Adding `-Wl,-rpath,<LIBDIR>` to the final link teaches the dynamic linker
//! where to find it, portably and without per-run environment variables.
use std::process::Command;

fn main() {
    // Prefer the same interpreter PyO3 builds against, then fall back through
    // common names so the right `libpython3.12.so.1.0` directory is located.
    let candidates: Vec<String> = [
        std::env::var("PYO3_PYTHON").ok().filter(|s| !s.is_empty()),
        Some("python3.12".into()),
        Some("python3".into()),
        Some("python".into()),
    ]
    .into_iter()
    .flatten()
    .collect();

    for py in &candidates {
        let out = Command::new(py)
            .args([
                "-c",
                "import sysconfig; v=sysconfig.get_config_var('LIBDIR'); print(v if v else '')",
            ])
            .output();
        if let Ok(out) = out {
            if out.status.success() {
                let libdir = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !libdir.is_empty() {
                    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libdir);
                    return;
                }
            }
        }
    }

    println!(
        "cargo:warning=digits_rust: could not detect a Python LIBDIR to bake into rpath. \
         Set PYO3_PYTHON to the interpreter that has sklearn/numpy/matplotlib, or set \
         LD_LIBRARY_PATH at run time."
    );
}
