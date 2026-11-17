//! Rendering of a [`SurveyReport`] to the pandas-style text the team expects.
//!
//! Two declarative macros keep the layout consistent and typo-resistant;
//! the alignment helpers reproduce the exact spacing of `pandas` so the
//! output is byte-identical to the reference report.

use crate::models::{AgeGroupStats, DeviceCount, SurveyReport};
use crate::traits::ReportFormatter;

/// Build a `Label: value` line.
macro_rules! kv_line {
    ($label:expr, $value:expr $(,)?) => {
        format!("{}: {}", $label, $value)
    };
}

/// Build the trailing `Name: <name>, dtype: <dtype>` line of a Series.
macro_rules! series_footer {
    ($name:expr, $dtype:expr $(,)?) => {
        format!("Name: {}, dtype: {}", $name, $dtype)
    };
}

/// Format an `f64` the way pandas prints it: integer values carry a trailing
/// `.0` (so `5.0` stays `5.0`, not `5`); everything else uses Rust's shortest
/// round-tripping representation.
fn fmt_float(v: f64) -> String {
    if !v.is_finite() {
        return v.to_string();
    }
    let s = format!("{}", v);
    if s.contains('.') {
        s
    } else {
        format!("{}.0", s)
    }
}

/// Right-justify `s` to `width` (leading padding).
fn rjust(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        format!("{}{}", " ".repeat(width - len), s)
    }
}

/// Left-justify `s` to `width` (trailing padding).
fn ljust(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(width - len))
    }
}

/// Reproduces the exact text layout the team is used to from pandas.
pub struct PandasStyleFormatter;

impl PandasStyleFormatter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PandasStyleFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl ReportFormatter for PandasStyleFormatter {
    fn format(&self, report: &SurveyReport) -> String {
        let mut out = String::new();
        out.push_str(&kv_line!("Missing scores", report.missing));
        out.push('\n');
        out.push_str(&kv_line!("Answers kept", report.kept));
        out.push('\n');
        out.push_str(&format_device_series(&report.device_counts));
        out.push('\n');
        out.push_str(&format_age_table(&report.age_stats));
        out
    }
}

/// Render the "device" Series: index name on its own line, one row per
/// device, then the pandas Series footer.
fn format_device_series(counts: &[DeviceCount]) -> String {
    const INDEX_NAME: &str = "device";

    // Index column = widest index value, but never narrower than the name.
    let index_width = counts
        .iter()
        .map(|d| d.device.chars().count())
        .chain(std::iter::once(INDEX_NAME.chars().count()))
        .max()
        .unwrap_or(INDEX_NAME.chars().count());

    // Value column = widest value string, padded up to the index width so the
    // two columns line up exactly like a pandas Series.
    let value_width = counts
        .iter()
        .map(|d| d.count.to_string().chars().count())
        .max()
        .unwrap_or(0)
        .max(index_width);

    let mut s = String::new();
    s.push_str(INDEX_NAME);
    s.push('\n');
    for d in counts {
        s.push_str(&ljust(&d.device, index_width));
        s.push_str(&rjust(&d.count.to_string(), value_width));
        s.push('\n');
    }
    s.push_str(&series_footer!("count", "int64"));
    s
}

/// Render the age-group DataFrame: a header row, the index-name row, then
/// one row per age group.
fn format_age_table(stats: &[AgeGroupStats]) -> String {
    const INDEX_NAME: &str = "age_group";
    const COL_NAMES: [&str; 2] = ["mean", "max"];
    const SEP: &str = "  ";

    let index_width = stats
        .iter()
        .map(|s| s.age_group.chars().count())
        .chain(std::iter::once(INDEX_NAME.chars().count()))
        .max()
        .unwrap_or(INDEX_NAME.chars().count());

    let mean_strs: Vec<String> = stats.iter().map(|s| fmt_float(s.mean)).collect();
    let max_strs: Vec<String> = stats.iter().map(|s| fmt_float(s.max)).collect();

    let col_widths: [usize; 2] = [
        COL_NAMES[0].chars().count().max(
            mean_strs
                .iter()
                .map(|s| s.chars().count())
                .max()
                .unwrap_or(0),
        ),
        COL_NAMES[1].chars().count().max(
            max_strs
                .iter()
                .map(|s| s.chars().count())
                .max()
                .unwrap_or(0),
        ),
    ];

    // Total line width = index column + separators + both data columns.
    let total_width = index_width + SEP.len() + col_widths[0] + SEP.len() + col_widths[1];

    let mut out = String::new();

    // Header line 1: blank index area, then right-justified column names.
    out.push_str(&" ".repeat(index_width));
    out.push_str(SEP);
    out.push_str(&rjust(COL_NAMES[0], col_widths[0]));
    out.push_str(SEP);
    out.push_str(&rjust(COL_NAMES[1], col_widths[1]));
    out.push('\n');

    // Header line 2: the index name, padded to the full width (matches the
    // trailing spaces pandas emits on this line).
    out.push_str(&ljust(INDEX_NAME, total_width));
    out.push('\n');

    // Data rows: left-justified index, right-justified values.
    for (i, s) in stats.iter().enumerate() {
        out.push_str(&ljust(&s.age_group, index_width));
        out.push_str(SEP);
        out.push_str(&rjust(&mean_strs[i], col_widths[0]));
        out.push_str(SEP);
        out.push_str(&rjust(&max_strs[i], col_widths[1]));
        out.push('\n');
    }

    // Drop the trailing newline; the caller adds the final one.
    if out.ends_with('\n') {
        out.pop();
    }
    out
}
