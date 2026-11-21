//! Watching a folder for new subs (N.I.N.A.'s output, milestone 14), and
//! the morning-after summary of a night.
//!
//! The watcher polls: it lists the folder (the scan's own rules, so WBPP
//! output and reject folders are skipped as usual) and reports a file once
//! its size and modification time have held still for one poll, so a sub
//! still being written is never read. Polling works the same on local
//! disks and SMB shares (`\\nas\…`), where change notifications are
//! unreliable; subs arrive minutes apart, so a few seconds' delay is fine.

use crate::config::Config;
use crate::grade::{Check, Grade};
use crate::pipeline::Triage;
use chrono::{NaiveDate, NaiveDateTime};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

pub struct Watcher {
    root: PathBuf,
    cfg: Config,
    known: HashMap<PathBuf, (u64, i64)>,
    pending: HashMap<PathBuf, (u64, i64)>,
}

fn stamp(p: &Path) -> Option<(u64, i64)> {
    crate::meta::file_stamp(p).ok()
}

impl Watcher {
    /// Start watching: files already there count as seen.
    pub fn new(root: &Path, cfg: &Config) -> Watcher {
        let files = crate::scan::list_files(root, cfg).files;
        let known = files.into_iter().filter_map(|f| stamp(&f).map(|s| (f, s))).collect();
        Watcher { root: root.to_path_buf(), cfg: cfg.clone(), known, pending: HashMap::new() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Files that are new (or changed) and have finished being written
    /// since the last poll.
    pub fn poll(&mut self) -> Vec<PathBuf> {
        let files = crate::scan::list_files(&self.root, &self.cfg).files;
        let mut ready = Vec::new();
        let mut still: HashMap<PathBuf, (u64, i64)> = HashMap::new();
        for f in files {
            let Some(s) = stamp(&f) else { continue };
            if self.known.get(&f) == Some(&s) {
                continue;
            }
            match self.pending.get(&f) {
                Some(prev) if *prev == s && s.0 > 0 => {
                    self.known.insert(f.clone(), s);
                    ready.push(f);
                }
                _ => {
                    still.insert(f, s);
                }
            }
        }
        self.pending = still;
        ready.sort();
        ready
    }

    /// Files seen but still changing.
    pub fn waiting(&self) -> usize {
        self.pending.len()
    }
}

/// One night of lights in a folder, the way you'd want it in the morning.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NightSummary {
    pub night: Option<NaiveDate>,
    pub lights: usize,
    pub good: usize,
    pub warn: usize,
    pub reject: usize,
    /// Exposure, hours: all lights, and those not rejected.
    pub hours: f64,
    pub hours_kept: f64,
    pub first: Option<NaiveDateTime>,
    pub last: Option<NaiveDateTime>,
    /// (target, filter, lights, hours).
    pub targets: Vec<(String, String, usize, f64)>,
    /// Why frames were flagged: (check, frames), most first.
    pub reasons: Vec<(String, usize)>,
    /// From here on most frames lost transparency.
    pub cloud_from: Option<NaiveDateTime>,
    /// Per filter: "H: flats OK (library H 2026-11-16)".
    pub flats: Vec<String>,
}

impl NightSummary {
    /// "84 subs: 71 good, 9 warn, 4 reject. Cloud from 03:40. Flats OK."
    pub fn headline(&self) -> String {
        if self.lights == 0 {
            return "No lights.".into();
        }
        let mut s = format!("{} sub{}: {} good, {} warn, {} reject ({:.1} h, {:.1} h kept).", self.lights, if self.lights == 1 { "" } else { "s" }, self.good, self.warn, self.reject, self.hours, self.hours_kept);
        if let Some(c) = self.cloud_from {
            s.push_str(&format!(" Cloud from {}.", c.format("%H:%M")));
        }
        if !self.flats.is_empty() {
            if self.flats.iter().all(|f| f.contains("flats OK")) {
                s.push_str(" Flats OK.");
            } else {
                s.push_str(&format!(" {}.", self.flats.iter().filter(|f| !f.contains("flats OK")).cloned().collect::<Vec<_>>().join(". ")));
            }
        }
        s
    }
}

fn is_raw_light(t: &Triage, i: usize) -> bool {
    let c = &t.scan.frames[i].class;
    c.frame_type == crate::classify::FrameType::Light && c.role == crate::classify::Role::Raw
}

/// The latest night with lights.
pub fn latest_night(t: &Triage) -> Option<NaiveDate> {
    t.reports.iter().filter(|r| is_raw_light(t, r.frame)).filter_map(|r| t.scan.frames[r.frame].class.night).max()
}

fn cloudy(r: &crate::grade::FrameReport) -> bool {
    r.transparency.as_ref().is_some_and(|x| x.value < 0.75) || r.flags.iter().any(|f| matches!(f.check, Check::Transparency | Check::Patchy))
}

/// Summary of one night's lights (frames outside reject folders).
pub fn night_summary(t: &Triage, night: NaiveDate) -> NightSummary {
    let mut s = NightSummary { night: Some(night), ..Default::default() };
    let mut rows: Vec<(&crate::grade::FrameReport, NaiveDateTime)> = Vec::new();
    let mut targets: BTreeMap<(String, String), (usize, f64)> = BTreeMap::new();
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for r in &t.reports {
        let f = &t.scan.frames[r.frame];
        if !is_raw_light(t, r.frame) || f.class.night != Some(night) || f.class.in_reject_dir {
            continue;
        }
        let exp = f.meta.exptime.unwrap_or(0.0) / 3600.0;
        s.lights += 1;
        s.hours += exp;
        match r.grade {
            Grade::Good => s.good += 1,
            Grade::Warn => s.warn += 1,
            Grade::Reject => s.reject += 1,
        }
        if r.grade != Grade::Reject {
            s.hours_kept += exp;
        }
        for fl in r.flags.iter().filter(|fl| fl.grade != Grade::Good) {
            *reasons.entry(fl.check.name().to_string()).or_default() += 1;
        }
        let obj = f.meta.object.clone().unwrap_or_else(|| "unknown".into());
        let e = targets.entry((obj, f.meta.filter.clone().unwrap_or_default())).or_default();
        e.0 += 1;
        e.1 += exp;
        if let Some(lt) = f.class.local_time {
            rows.push((r, lt));
        }
    }
    rows.sort_by_key(|x| x.1);
    s.first = rows.first().map(|x| x.1);
    s.last = rows.last().map(|x| x.1);
    // Cloud: the earliest cloudy frame after which most frames (≥ 60%, at
    // least 3) are cloudy.
    let n = rows.len();
    let mut suffix = vec![0usize; n + 1];
    for i in (0..n).rev() {
        suffix[i] = suffix[i + 1] + cloudy(rows[i].0) as usize;
    }
    for i in 0..n {
        let rest = n - i;
        if cloudy(rows[i].0) && suffix[i] >= 3 && suffix[i] * 10 >= rest * 6 {
            s.cloud_from = Some(rows[i].1);
            break;
        }
    }
    s.targets = targets.into_iter().map(|((o, f), (k, h))| (o, f, k, h)).collect();
    let mut rv: Vec<(String, usize)> = reasons.into_iter().collect();
    rv.sort_by(|a, b| b.1.cmp(&a.1));
    s.reasons = rv;
    for c in t.cal.choices.iter().filter(|c| c.night == night) {
        let what = match (&c.chosen, c.grade) {
            (None, _) => format!("{}: no flats", c.filter),
            (Some(o), Grade::Good) => format!("{}: flats OK ({})", c.filter, o.label),
            (Some(_), _) => format!("{}: {}", c.filter, c.reason),
        };
        s.flats.push(what);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_is_ready_once_it_stops_changing() {
        let dir = std::env::temp_dir().join(format!("crux-watch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("old.fits"), b"x").unwrap();
        let cfg = Config::default();
        let mut w = Watcher::new(&dir, &cfg);
        assert!(w.poll().is_empty());
        std::fs::write(dir.join("new.fits"), b"partial").unwrap();
        assert!(w.poll().is_empty(), "first sighting waits");
        assert_eq!(w.waiting(), 1);
        let ready = w.poll();
        assert_eq!(ready.len(), 1);
        assert!(ready[0].ends_with("new.fits"));
        assert!(w.poll().is_empty(), "reported once");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
