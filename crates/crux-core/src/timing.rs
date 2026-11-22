//! `CRUX_TIMING=1` laps: each step's own time and the running total, on
//! stderr. Free when the variable isn't set.

use std::time::Instant;

/// Is `CRUX_TIMING` set?
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CRUX_TIMING").is_some())
}

pub struct Laps {
    label: String,
    start: Instant,
    last: std::cell::Cell<Instant>,
    on: bool,
}

impl Laps {
    pub fn new(label: impl Into<String>) -> Laps {
        let now = Instant::now();
        let on = on();
        Laps { label: if on { label.into() } else { String::new() }, start: now, last: std::cell::Cell::new(now), on }
    }

    /// Print `what` with the time since the last lap and since the start.
    pub fn lap(&self, what: &str) {
        if !self.on {
            return;
        }
        let now = Instant::now();
        eprintln!("{}: {what} {:.2} s (at {:.2} s)", self.label, (now - self.last.get()).as_secs_f64(), (now - self.start).as_secs_f64());
        self.last.set(now);
    }
}
