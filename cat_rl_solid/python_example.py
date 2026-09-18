#!/usr/bin/env python3
"""
Пример вызова Rust-библиотеки cat_rl_solid из Python через ctypes (FFI).

Перед запуском соберите библиотеку:
    cargo build --release

Затем:
    python3 python_example.py
"""

import ctypes
import os
import sys
from pathlib import Path

# -------------------------------------------------
# Поиск динамической библиотеки
# -------------------------------------------------
def find_library() -> str:
    root = Path(__file__).parent
    candidates = [
        root / "target" / "release" / "libcat_rl_solid.so",      # Linux
        root / "target" / "release" / "libcat_rl_solid.dylib",   # macOS
        root / "target" / "release" / "cat_rl_solid.dll",        # Windows
        root / "target" / "debug" / "libcat_rl_solid.so",
        root / "target" / "debug" / "libcat_rl_solid.dylib",
        root / "target" / "debug" / "cat_rl_solid.dll",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        "Не найдена библиотека libcat_rl_solid. "
        "Сначала выполните: cargo build --release"
    )


# -------------------------------------------------
# Загрузка и настройка сигнатур
# -------------------------------------------------
lib_path = find_library()
print(f"Загружаем библиотеку: {lib_path}")
lib = ctypes.CDLL(lib_path)

# cat_new(name: *const c_char, lives: u8) -> *mut FfiCat
lib.cat_new.argtypes = [ctypes.c_char_p, ctypes.c_uint8]
lib.cat_new.restype = ctypes.c_void_p

# cat_meow(cat: *mut FfiCat)
lib.cat_meow.argtypes = [ctypes.c_void_p]
lib.cat_meow.restype = None

# cat_get_lives(cat: *const FfiCat) -> u8
lib.cat_get_lives.argtypes = [ctypes.c_void_p]
lib.cat_get_lives.restype = ctypes.c_uint8

# cat_feed(cat: *mut FfiCat) -> f32
lib.cat_feed.argtypes = [ctypes.c_void_p]
lib.cat_feed.restype = ctypes.c_float

# cat_play(cat: *mut FfiCat) -> f32
lib.cat_play.argtypes = [ctypes.c_void_p]
lib.cat_play.restype = ctypes.c_float

# cat_free(cat: *mut FfiCat)
lib.cat_free.argtypes = [ctypes.c_void_p]
lib.cat_free.restype = None


# -------------------------------------------------
# Демонстрация
# -------------------------------------------------
def main():
    print("=== Вызов Rust FFI из Python ===\n")

    name = b"Murka"
    lives = 9

    # 1. Создаём кота
    cat_ptr = lib.cat_new(name, lives)
    if not cat_ptr:
        print("Ошибка: не удалось создать кота (проверьте имя и lives 1..9)")
        sys.exit(1)

    print(f"Кот создан, указатель: {hex(cat_ptr)}")

    # 2. Мяукаем
    print("Вызов cat_meow:")
    lib.cat_meow(cat_ptr)

    # 3. Смотрим жизни
    current_lives = lib.cat_get_lives(cat_ptr)
    print(f"Текущие жизни: {current_lives}")

    # 4. Кормим и играем
    print("\nКормим кота (feed):")
    reward = lib.cat_feed(cat_ptr)
    print(f"  reward = {reward:.1f}, lives = {lib.cat_get_lives(cat_ptr)}")

    print("Играем с котом (play):")
    reward = lib.cat_play(cat_ptr)
    print(f"  reward = {reward:.1f}, lives = {lib.cat_get_lives(cat_ptr)}")

    # 5. Ещё раз мяукаем
    print("\nСнова meow:")
    lib.cat_meow(cat_ptr)

    # 6. Обязательно освобождаем память
    lib.cat_free(cat_ptr)
    print("\nПамять освобождена (cat_free). Готово!")


if __name__ == "__main__":
    main()
