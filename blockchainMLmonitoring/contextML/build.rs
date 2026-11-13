use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Compile the C file
    cc::Build::new()
        .file("src/cuda_mock.c")
        .compile("cuda_mock");
    
    // Tell cargo to link the library
    println!("cargo:rustc-link-lib=static=cuda_mock");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
}
