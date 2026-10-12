# **FFI (Foreign Function Interface) – вызов из Python**

Далее необходимо создать динамические библиотеки Rust - например для вызова их из Python.

Rust экспортирует функции с сигнатурой C, Python загружает .so / .dylib / .dll через ctypes и вызывает эти символы.

1. Сборка динамической библиотеки


cdylib C-совместимая shared library для Python/C

rlib обычная Rust-библиотека для main и тестов

После:
cargo build --release
появляется:

Linux: target/release/libcat_rl_solid.so
macOS: target/release/libcat_rl_solid.dylib
Windows: target/release/cat_rl_solid.dll

Именно этот файл открывает Python.

2. C-совместимая структура на стороне Rust

Безопасный Cat (с String) нельзя отдавать в C/Python как есть. 

Для FFI — отдельный тип:


```
#[repr(C)]
pub struct FfiCat {
    name: *mut c_char,  // C-строка, не Rust String
    lives: u8,
}
```

#[repr(C)]layout как в C (без Rust-оптимизаций порядка полей)

*mut c_char указатель на null-terminated строку

u8 совместим с ctypes.c_uint8

Внутри Python этот объект — просто непрозрачный указатель (void*), поля напрямую не читают.

3. Экспортируемые функции


Все с:
```
#[no_mangle]
pub extern "C" fn ...

```


#[no_mangle]имя символа не «уродуется» компилятором (cat_new, а не _ZN...)

extern "C" calling convention C (как ожидает ctypes)

Создание


```
pub extern "C" fn cat_new(name: *const c_char, lives: u8) -> *mut FfiCat
```


Что происходит:

Проверка name != null.

CStr::from_ptr → Rust String (unsafe, но локализован).

Валидация: имя не пустое, lives ∈ [1, 9].

Имя снова в CString::into_raw() — память принадлежит C-стороне.

Box::into_raw(Box::new(cat)) — объект в куче, наружу отдаётся сырой указатель.

При ошибке → null (Python проверяет if not cat_ptr).

Использование

```
Rustcat_meow(cat)       // печать "Murka says meow! (lives: N)"
cat_get_lives(cat)  // -> u8
cat_feed(cat)       // -> f32 reward, меняет lives
cat_play(cat)       // -> f32 reward, меняет lives

```

Везде:

проверка на null;

unsafe { &*cat } или &mut *cat;

для имени снова CStr::from_ptr.


Освобождение

```
pub extern "C" fn cat_free(cat: *mut FfiCat) {
    if cat.is_null() { return; }
    unsafe {
        let cat = Box::from_raw(cat);           // вернуть владение Box
        if !cat.name.is_null() {
            let _ = CString::from_raw(cat.name); // освободить C-строку
        }
    } // drop Box
}

```

Без cat_free — утечка: и структура, и строка имени.

4. Сторона Python (python_example.py)

Поиск .so

```
candidates = [
    root / "target" / "release" / "libcat_rl_solid.so",
    root / "target" / "debug" / "libcat_rl_solid.so",
    # + dylib / dll
]
lib = ctypes.CDLL(lib_path)
```


CDLL загружает shared library в процесс Python.

Объявление сигнатур (обязательно)

Без argtypes / restype ctypes угадывает типы неверно → краши / мусор.


```
lib.cat_new.argtypes = [ctypes.c_char_p, ctypes.c_uint8]
lib.cat_new.restype  = ctypes.c_void_p

lib.cat_meow.argtypes = [ctypes.c_void_p]
lib.cat_meow.restype  = None

lib.cat_get_lives.restype = ctypes.c_uint8
lib.cat_feed.restype      = ctypes.c_float
lib.cat_play.restype      = ctypes.c_float
lib.cat_free.argtypes     = [ctypes.c_void_p]
```


Соответствие:

*const c_char c_char_p

u8 c_uint8

*mut FfiCat c_void_p

f32 c_float
() None

Типичный сценарий
```
name = b"Murka"          # bytes → C char*
lives = 9

cat_ptr = lib.cat_new(name, lives)
if not cat_ptr:
    sys.exit(1)

lib.cat_meow(cat_ptr)
lives = lib.cat_get_lives(cat_ptr)
reward = lib.cat_feed(cat_ptr)
reward = lib.cat_play(cat_ptr)

lib.cat_free(cat_ptr)    # обязательно

```
	
5. Схема вызова

Python                          Rust (libcat_rl_solid.so)
──────                          ─────────────────────────

ctypes.CDLL(...)
        ──────────────────────►  загрузка .so в память
        

lib.cat_new(b"Murka", 9)
        ── C ABI ─────────────►  cat_new:
                                   CStr → String
                                   валидация
                                   Box → *mut FfiCat
                                   
        ◄── указатель ─────────
        

lib.cat_meow(cat_ptr)
        ── C ABI ─────────────►  cat_meow: println!
        

lib.cat_feed(cat_ptr)

        ── C ABI ─────────────►  lives += 1, return 1.0
        ◄── float ─────────────
        

lib.cat_free(cat_ptr)

        ── C ABI ─────────────►  Box::from_raw + free name
        

6. Владение памятью (самое важное)


Правила:

Не вызывать cat_free дважды (double free).

Не использовать cat_ptr после cat_free (use-after-free).

Python не делает del по полям структуры — она opaque.


Надёжный паттерн в Python — обёртка:


```
class Cat:
    def __init__(self, name: bytes, lives: int):
        self._ptr = lib.cat_new(name, lives)
        if not self._ptr:
            raise ValueError("cat_new failed")
    def meow(self):
        lib.cat_meow(self._ptr)
    def __del__(self):
        if self._ptr:
            lib.cat_free(self._ptr)
            self._ptr = None
```

(В примере проекта — явный cat_free в конце main.)

7. Что не экспортировано в FFI

Сейчас через Python доступен только кот и простые действия, не весь RL-стек:

Есть в FFI
cat_new / cat_free 
cat_meow 
cat_feed / cat_play
cat_get_lives

Чтобы вызывать нейросеть из Python, нужно добавить отдельные extern "C" функции (например policy_forward(state, out_probs)), сериализацию буферов float* и т.д. Архитектура уже готова: тот же cdylib.


