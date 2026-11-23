use csv::ReaderBuilder;
use serde::Deserialize;
use std::fs::File;
use std::io;
use std::sync::Arc;
use std::thread;

// ============================================================
// 1. ДОМЕН (SRP): модель предметной области
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sky {
    Clear,
    Cloudy,
    Foggy,
}

impl Sky {
    const ALL: [Sky; 3] = [Sky::Clear, Sky::Cloudy, Sky::Foggy];

    fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "clear" => Some(Self::Clear),
            "cloudy" => Some(Self::Cloudy),
            "foggy" => Some(Self::Foggy),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Clear => "clear",
            Self::Cloudy => "cloudy",
            Self::Foggy => "foggy",
        }
    }
}

#[derive(Debug, Clone)]
struct Record {
    sky: Sky,
    rain: bool,
}

#[derive(Deserialize)]
struct RawRecord {
    sky: String,
    rain: String,
}

// ============================================================
// 2. АБСТРАКЦИЯ ДОСТУПА К ДАННЫМ (DIP — зависим от трейта)
// ============================================================

trait DataSource: Send + Sync {
    fn load(&self) -> io::Result<Vec<Record>>;
}

struct CsvDataSource {
    path: String,
}

impl DataSource for CsvDataSource {
    fn load(&self) -> io::Result<Vec<Record>> {
        let file = File::open(&self.path)?;
        let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
        let mut records = Vec::new();
        for raw in rdr.deserialize::<RawRecord>() {
            let raw = raw.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            if let Some(sky) = Sky::parse(&raw.sky) {
                records.push(Record { sky, rain: raw.rain == "yes" });
            }
        }
        Ok(records)
    }
}

// ============================================================
// 3. "ML"-ПРЕДИКТОР (OCP — легко добавлять новые)
// ============================================================

trait Predictor: Send + Sync {
    fn probability(&self, records: &[Record]) -> f64;
}

/// P(rain | sky == заданный)
struct SkyRainPredictor(Sky);

impl Predictor for SkyRainPredictor {
    fn probability(&self, records: &[Record]) -> f64 {
        let (mut n, mut rainy) = (0usize, 0usize);
        for r in records {
            if r.sky == self.0 {
                n += 1;
                if r.rain {
                    rainy += 1;
                }
            }
        }
        if n == 0 { 0.0 } else { rainy as f64 / n as f64 }
    }
}

/// P(rain) — приор
struct AnyDayPredictor;

impl Predictor for AnyDayPredictor {
    fn probability(&self, records: &[Record]) -> f64 {
        if records.is_empty() {
            return 0.0;
        }
        let rainy = records.iter().filter(|r| r.rain).count();
        rainy as f64 / records.len() as f64
    }
}

// ============================================================
// 4. МАКРОС для форматированного вывода
// ============================================================

macro_rules! report {
    ($label:expr, $value:expr) => {
        println!("{}: {:.2}", $label, $value);
    };
}

// ============================================================
// 5. MAIN — многопоточная агрегация
// ============================================================

fn main() -> io::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "mornings.csv".into());
    let records = Arc::new(CsvDataSource { path }.load()?);

    // Параллельно считаем вероятность для каждого неба
    let mut handles = Vec::new();
    for sky in Sky::ALL {
        let recs = Arc::clone(&records);
        handles.push(thread::spawn(move || {
            let p = SkyRainPredictor(sky).probability(&recs);
            (sky, p)
        }));
    }

    // Параллельно — вероятность дождя в любой день
    let recs = Arc::clone(&records);
    let any_handle = thread::spawn(move || AnyDayPredictor.probability(&recs));

    let mut sky_results = Vec::new();
    for h in handles {
        let (sky, p) = h.join().expect("thread panicked");
        sky_results.push((sky, p));
        report!(format!("Rain after a {} morning", sky.name()), p);
    }

    let any = any_handle.join().expect("thread panicked");
    report!("Rain on any day", any);

    // Лучшая подсказка — argmax
    let (best, _) = sky_results
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    println!("Best clue for rain: {}", best.name());

    Ok(())
}
