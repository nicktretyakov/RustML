//! Модуль `report` — вывод результатов (SRP, DIP).
//!
//! Trait [`Reporter`] абстрагирует способ вывода; [`StdoutReporter`] печатает
//! результат в stdout в формате, ожидаемом задачей.

use std::io::{self, Write};

use crate::stats::QuartileSummary;
use crate::macros::{format_metric, quarter_line};

/// Способ отчёта о результатах анализа.
pub trait Reporter {
    fn report_summary(&self, s: &QuartileSummary) -> io::Result<()>;
    fn report_client(&self, quarter: u8) -> io::Result<()>;
}

/// Печать в стандартный вывод.
pub struct StdoutReporter;

impl Reporter for StdoutReporter {
    fn report_summary(&self, s: &QuartileSummary) -> io::Result<()> {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        // Квартили в одну строку (как в Python-примере).
        writeln!(
            out,
            "Quartiles: {:.1}, {:.1}, {:.1} liters",
            s.q1, s.median, s.q3
        )?;
        writeln!(out, "{}", format_metric!("IQR", s.iqr))?;
        // Дополнительная ML-описательная статистика (mean, std) — из крейта statrs.
        writeln!(out, "Mean: {:.1} liters", s.mean)?;
        writeln!(out, "Std dev: {:.1} liters", s.std)?;
        Ok(())
    }

    fn report_client(&self, quarter: u8) -> io::Result<()> {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        writeln!(out, "{}", quarter_line!(quarter))?;
        Ok(())
    }
}
