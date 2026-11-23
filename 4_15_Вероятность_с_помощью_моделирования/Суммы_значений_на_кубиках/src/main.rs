//! Бросок 2d4 (два четырёхгранных кубика), подсчёт урона.
//!
//! Архитектура (SOLID):
//!   * SRP — каждый модуль делает ровно одну вещь (SeedSequence, Pcg64,
//!           UniformInt, Dice, Estimator, Renderer).
//!   * OCP / LSP / ISP — маленькие трейты `WordSource`, `UniformInt`,
//!           `Dice`, `DamageEstimator`.
//!   * DIP — `Simulator<D: Dice>` не знает, какой именно ГПСЧ используется.
//!
//! «ML» — `FrequencyEstimator`: онлайн MLE категориального распределения.

use rayon::prelude::*;
use std::io::{self, BufRead};

// ======================================================================
// 1. SeedSequence — порт numpy.random.SeedSequence (bit-exact)
// ======================================================================
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

#[inline]
fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^= v >> XSHIFT;
    v
}

#[inline]
fn mix32(x: u32) -> u32 {
    let mut r = (x ^ (x >> XSHIFT)).wrapping_mul(MIX_MULT_L);
    r = (r ^ (r >> XSHIFT)).wrapping_mul(MIX_MULT_R);
    r ^ (r >> XSHIFT)
}

pub struct SeedSequence {
    pool: [u32; 4],
}

impl SeedSequence {
    pub fn new(entropy: u64) -> Self {
        // int -> массив uint32 (little-endian). Для 4 это [4].
        let mut words = Vec::new();
        let mut e = entropy;
        loop {
            words.push((e & 0xFFFF_FFFF) as u32);
            e >>= 32;
            if e == 0 {
                break;
            }
        }
        let mut pool = [0u32; 4];
        let mut hash_const = INIT_A;
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(words.get(i).copied().unwrap_or(0), &mut hash_const);
        }
        Self { pool }
    }

    pub fn generate_state_u32(&self, n: usize) -> Vec<u32> {
        let mut state = vec![0u32; n];
        let mut hash_const = INIT_B;
        for i in 0..n {
            let mut d = self.pool[i % self.pool.len()];
            d ^= hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            d = d.wrapping_mul(hash_const);
            d ^= d >> XSHIFT;
            state[i] = d;
        }
        state.iter_mut().for_each(|s| *s = mix32(*s));
        state
    }

    /// numpy отдаёт uint64 как пару uint32 (little-endian).
    pub fn generate_state_u64(&self, n: usize) -> Vec<u64> {
        self.generate_state_u32(2 * n)
            .chunks_exact(2)
            .map(|c| (c[0] as u64) | ((c[1] as u64) << 32))
            .collect()
    }
}

// ======================================================================
// 2. PCG64 — порт numpy.random.PCG64 (XSL RR 128/64)
// ======================================================================
const PCG_MULTIPLIER_128: u128 =
    0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

pub struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    pub fn seeded(entropy: u64) -> Self {
        let s = SeedSequence::new(entropy).generate_state_u64(4);
        let initstate = ((s[0] as u128) << 64) | s[1] as u128;
        let initseq = ((s[2] as u128) << 64) | s[3] as u128;

        let mut rng = Pcg64 { state: 0, inc: 0 };
        // pcg_setseq_128_srandom_r
        rng.state = 0;
        rng.inc = (initseq << 1) | 1;
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULTIPLIER_128).wrapping_add(self.inc);
    }
}

// ======================================================================
// 3. SOLID-трейты
// ======================================================================

/// Поставщик сырых 64-битных слов (DIP: абстракция над ГПСЧ).
pub trait WordSource {
    fn next_u64(&mut self) -> u64;
    #[inline]
    fn next_u32(&mut self) -> u32 {
        // numpy: next_uint32(pcg64) = (uint32_t)(next_uint64 >> 32)
        (self.next_u64() >> 32) as u32
    }
}

impl WordSource for Pcg64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.step();
        let rot = (self.state >> 122) as u32;
        let x = ((self.state >> 64) as u64) ^ (self.state as u64);
        x.rotate_right(rot)
    }
}

/// Равномерное целое в `[off, off + range)` — Lemire's method (как в numpy).
pub trait UniformInt: WordSource {
    fn bounded(&mut self, off: u32, range: u32) -> u32 {
        if range == 0 {
            return off.wrapping_add(self.next_u32());
        }
        let mut x = self.next_u32();
        let mut m = (x as u64) * (range as u64);
        let mut leftover = m as u32;
        if leftover < range {
            let threshold = (u32::MAX - range + 1) % range;
            while leftover < threshold {
                x = self.next_u32();
                m = (x as u64) * (range as u64);
                leftover = m as u32;
            }
        }
        off + (m >> 32) as u32
    }
}

impl<T: WordSource + ?Sized> UniformInt for T {}

/// Абстрактный источник бросков.
pub trait Dice {
    fn roll(&mut self) -> u32;
}

/// Кубик с `sides` гранями (SRP — только бросок).
pub struct Die<R: UniformInt> {
    rng: R,
    sides: u32,
}

impl<R: UniformInt> Die<R> {
    pub fn new(rng: R, sides: u32) -> Self {
        Self { rng, sides }
    }
}

impl<R: UniformInt> Dice for Die<R> {
    #[inline]
    fn roll(&mut self) -> u32 {
        self.rng.bounded(1, self.sides) // numpy: integers(1, sides+1)
    }
}

// ======================================================================
// 4. Симулятор: считает суммы бросков
// ======================================================================
pub struct Simulator<D: Dice> {
    dice: D,
    rolls: usize,
}

impl<D: Dice> Simulator<D> {
    pub fn new(dice: D, rolls: usize) -> Self {
        Self { dice, rolls }
    }

    /// Возвращает урон по каждой атаке (порядок как в numpy‑скрипте:
    /// сначала все `first`, затем все `second`).
    pub fn run(&mut self) -> Vec<u32> {
        let mut first = Vec::with_capacity(self.rolls);
        let mut second = Vec::with_capacity(self.rolls);
        for _ in 0..self.rolls {
            first.push(self.dice.roll());
        }
        for _ in 0..self.rolls {
            second.push(self.dice.roll());
        }
        first
            .into_iter()
            .zip(second)
            .map(|(a, b)| a + b)
            .collect()
    }
}

// ======================================================================
// 5. «ML»: онлайн MLE категориального распределения
// ======================================================================
pub struct FrequencyEstimator {
    low: u32,
    counts: Vec<u64>,
    total: u64,
}

impl FrequencyEstimator {
    pub fn new(low: u32, high: u32) -> Self {
        Self {
            low,
            counts: vec![0; (high - low + 1) as usize],
            total: 0,
        }
    }

    pub fn observe(&mut self, value: u32) {
        self.counts[(value - self.low) as usize] += 1;
        self.total += 1;
    }

    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    pub fn low(&self) -> u32 {
        self.low
    }

    /// Максимум правдоподобия; при равенстве — первое (меньшее) значение,
    /// как `value_counts().index[0]` у pandas.
    pub fn most_likely(&self) -> (u32, f64) {
        let mut best_idx = 0usize;
        let mut best = 0u64;
        for (i, &c) in self.counts.iter().enumerate() {
            if c > best {
                best = c;
                best_idx = i;
            }
        }
        (self.low + best_idx as u32, best as f64 / self.total as f64)
    }
}

// ======================================================================
// 6. Макрос отрисовки столбцов
// ======================================================================
macro_rules! render_bars {
    ($counts:expr, $low:expr, $width:expr) => {{
        let max = $counts.iter().copied().max().unwrap_or(0).max(1);
        for (i, &c) in $counts.iter().enumerate() {
            let n = ((c as f64 / max as f64) * $width as f64).round() as usize;
            println!(
                "{:>2} | {:<width$} {:>5}",
                $low + i as u32,
                "█".repeat(n),
                c,
                width = $width
            );
        }
    }};
}

// ======================================================================
// 7. main
// ======================================================================
fn main() {
    let rolls: usize = {
        let stdin = io::stdin();
        let mut line = String::new();
        stdin.lock().read_line(&mut line).expect("stdin");
        line.trim().parse().expect("integer expected")
    };
    if rolls == 0 {
        eprintln!("no rolls");
        return;
    }

    // --- сбор данных (как numpy: default_rng(4)) ---
    let rng = Pcg64::seeded(4);
    let dice = Die::new(rng, 4);
    let mut sim = Simulator::new(dice, rolls);
    let damages = sim.run();

    // --- подсчёт в несколько потоков (rayon) ---
    const LOW: u32 = 2; // 1+1
    const HIGH: u32 = 8; // 4+4
    const BUCKETS: usize = (HIGH - LOW + 1) as usize;

    let partial: Vec<[u64; BUCKETS]> = damages
        .par_chunks(4096)
        .map(|chunk| {
            let mut local = [0u64; BUCKETS];
            for &d in chunk {
                local[(d - LOW) as usize] += 1;
            }
            local
        })
        .collect();

    let mut counts = [0u64; BUCKETS];
    for p in &partial {
        for i in 0..BUCKETS {
            counts[i] += p[i];
        }
    }

    // --- «ML»-оценка (MLE) ---
    let mut model = FrequencyEstimator::new(LOW, HIGH);
    for &d in &damages {
        model.observe(d);
    }
    let (best, share) = model.most_likely();

    // --- вывод ---
    println!("Most likely damage: {}, share {:.3}", best, share);

    // --- столбцы ---
    println!();
    println!("Damage Rolls");
    println!("Damage | Rolls");
    render_bars!(counts, LOW, 40);
}
