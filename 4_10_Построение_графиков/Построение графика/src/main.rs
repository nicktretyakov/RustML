//! Entry point.
//!
//! Wires the concrete implementations together and hands them to the
//! [`RiverAnalysis`] orchestrator. This is the *only* place that knows about
//! the concrete types — everywhere else depends on traits (SOLID).

mod analysis;
mod data;
mod macros;
mod ml;
mod model;
mod plot;
mod preprocess;

use plotters::style::{register_font, FontStyle};

use crate::analysis::RiverAnalysis;
use crate::data::CsvDataSource;
use crate::ml::SmartcoreLinearRegression;
use crate::plot::RiverLevelPlot;
use crate::preprocess::RayonPreprocessor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Register the TrueType font so the ab_glyph backend can render the title
    // and axis labels without consulting a system font database. The font is
    // loaded at run time from whichever candidate location exists (see
    // `load_font`), so the crate compiles even when the TTF is not present at
    // build time and regardless of the directory layout the user chose.
    register_font("sans-serif", FontStyle::Normal, load_font()?)
        .map_err(|_| -> Box<dyn std::error::Error> {
            "font registration failed (DejaVuSans.ttf)".into()
        })?;

    // Optional CLI overrides: `river_analysis <csv_path> <out_png>`.
    let args: Vec<String> = std::env::args().collect();
    let csv_path = args.get(1).map(|s| s.as_str()).unwrap_or("river.csv");
    let out_png = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("river_level.png");

    println!("Reading sensor data from '{csv_path}'");
    println!("Writing plot to '{out_png}'");

    // Dependency injection: pick concrete types once, here only.
    let analysis = RiverAnalysis::new(
        CsvDataSource::new(),
        RayonPreprocessor::new(),
        SmartcoreLinearRegression::new(),
        RiverLevelPlot::default_size(),
    );

    let report = analysis.run(csv_path, out_png)?;
    report.print();

    // Sanity checks using the `assert_close!` macro:
    // 1. The fitted line passes through the centroid, so the mean of the
    //    predictions equals the mean of the observed levels.
    let pred_mean =
        report.fit.predictions.iter().sum::<f64>() / report.fit.predictions.len() as f64;
    assert_close!(pred_mean, report.processed.mean_level, 1e-3);

    // 2. Heavier rainfall must raise the level => positive slope, and the model
    //    should explain most of the variance.
    assert!(
        report.fit.slope > 0.0,
        "expected a positive slope (rain raises the level), got {}",
        report.fit.slope
    );
    assert!(
        report.fit.r_squared > 0.8,
        "expected R-squared > 0.8, got {}",
        report.fit.r_squared
    );

    println!("\nDone. Plot written to '{out_png}'.");
    Ok(())
}

/// Locate and load the DejaVuSans TrueType font as `&'static [u8]`.
///
/// `plotters::style::register_font` requires a `&'static [u8]`, so the bytes
/// read from disk are leaked with `Box::leak` — the font is registered exactly
/// once and lives for the entire program, so the leak is intentional and
/// bounded. Several candidate locations are tried in order so the program runs
/// regardless of where the TTF sits on disk:
///
///   1. `<crate>/assets/DejaVuSans.ttf`   — the delivered layout
///   2. `<crate>/DejaVuSans.ttf`          — crate root
///   3. `assets/DejaVuSans.ttf`           — relative to the run CWD
///   4. `DejaVuSans.ttf`                  — relative to the run CWD
///   5. `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf` — Linux system font
///   6. `C:\Windows\Fonts\arial.ttf`      — Windows fallback (Arial)
///
/// The crate root is baked in at compile time via `CARGO_MANIFEST_DIR`, which
/// cargo always sets — no build script is required, so the crate compiles even
/// when the font file is absent at build time.
fn load_font() -> Result<&'static [u8], Box<dyn std::error::Error>> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let candidates: [String; 6] = [
        format!("{manifest}/assets/DejaVuSans.ttf"),
        format!("{manifest}/DejaVuSans.ttf"),
        "assets/DejaVuSans.ttf".to_string(),
        "DejaVuSans.ttf".to_string(),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf".to_string(),
        r"C:\Windows\Fonts\arial.ttf".to_string(),
    ];

    for path in &candidates {
        match std::fs::read(path) {
            Ok(bytes) => {
                // Leak the Vec into a 'static slice: the font outlives every
                // call site by design (registered once at startup).
                let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
                eprintln!("[river_analysis] using font: {path}");
                return Ok(leaked);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("failed to read font '{path}': {e}").into()),
        }
    }

    Err(format!(
        "DejaVuSans.ttf was not found in any candidate location.\n\
         Place the file at one of:\n  \
         - <crate>/assets/DejaVuSans.ttf\n  \
         - <crate>/DejaVuSans.ttf\n  \
         - /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
           (Debian/Ubuntu: `apt install fonts-dejavu-core`)\n\
         The font is required to render the plot's title and axis labels."
    )
    .into())
}
