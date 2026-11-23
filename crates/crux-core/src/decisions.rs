//! Your decisions, shared by the GUI and the CLI: per-frame overrides
//! (keep or reject, pinned so they survive re-runs), threshold changes and
//! dismissed Inbox items. Stored as `decisions.json` in the cache folder,
//! never in the data, and moved along with files by `moves`.

use crate::cache::key;
use crate::config::Config;
use crate::grade::{Check, FrameReport, Grade};
use crate::scan::Frame;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const FILE: &str = "decisions.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedView {
    pub name: String,
    pub query: String,
}

/// A flat set you chose: a library set, or a flat group of that folder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlatPick {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library_id: Option<String>,
    /// The flat group's label in the folder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    pub night: chrono::NaiveDate,
    pub at: String,
}

/// "<folder>|H|2026-11-11": the flats for one night and filter of a folder's lights.
pub fn flat_key(root: &Path, filter: &str, night: chrono::NaiveDate) -> String {
    format!("{}|{}|{}", key(root), filter.to_uppercase(), night)
}

/// "<folder>|dark|300|6248x4176": the dark (or bias) for a family of lights.
pub fn offset_key(root: &Path, kind: &str, family: &crate::fingerprint::Family) -> String {
    offset_key_of(root, kind, family.exptime, family.binning, family.width, family.height)
}

/// `offset_key` from a light's own fields.
pub fn offset_key_of(root: &Path, kind: &str, exptime: Option<f64>, binning: u32, width: u32, height: u32) -> String {
    format!("{}|{}|{:.2}|{}|{}x{}", key(root), kind, exptime.unwrap_or(0.0), binning, width, height)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Override {
    /// Good (keep) or Reject.
    pub grade: Grade,
    /// Local time the decision was made.
    pub at: String,
}

/// Threshold keys the GUI's reject sliders change, with their config paths.
pub const THRESHOLDS: [(&str, &str); 4] = [
    ("transparency.reject", "transparency below"),
    ("trails.reject_sigma", "trail brighter than (× noise)"),
    ("focus.reject_ratio", "FWHM over night median"),
    ("sky.reject_ratio", "sky over panel's best"),
];

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Decisions {
    pub version: u32,
    /// Cache key (lower-case path) → override.
    #[serde(default)]
    pub overrides: BTreeMap<String, Override>,
    /// `thresholds.<key>` values that differ from the config, e.g.
    /// "transparency.reject" → 0.6.
    #[serde(default)]
    pub thresholds: BTreeMap<String, f64>,
    /// Inbox items you've dealt with.
    #[serde(default)]
    pub dismissed: BTreeSet<String>,
    /// Panel corrections (cache key → panel number): "these are really
    /// panel 13". FITS headers are never changed.
    #[serde(default)]
    pub panels: BTreeMap<String, u32>,
    /// Your layout rules (replace `[layout]` from crux.toml when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<crate::config::Layout>,
    /// Saved smart views.
    #[serde(default)]
    pub views: Vec<SavedView>,
    /// Flats you picked for a night's lights, by `flat_key`.
    #[serde(default)]
    pub flats: BTreeMap<String, FlatPick>,
    /// Darks and bias you picked for a family of lights, by `offset_key`:
    /// the cache key of the master (or the set's first file).
    #[serde(default)]
    pub offsets: BTreeMap<String, String>,
    #[serde(skip)]
    pub file: Option<PathBuf>,
}

impl Decisions {
    /// Load from `<cache dir>/decisions.json`; empty when it doesn't exist.
    pub fn load(cache_dir: &Path) -> Result<Decisions> {
        let file = cache_dir.join(FILE);
        let mut d = match std::fs::read(&file) {
            Ok(b) => serde_json::from_slice::<Decisions>(&b).with_context(|| format!("read {}", file.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Decisions { version: 1, ..Default::default() },
            Err(e) => return Err(e).with_context(|| format!("read {}", file.display())),
        };
        d.file = Some(file);
        Ok(d)
    }

    /// Write atomically. No-op without a file (cache disabled).
    pub fn save(&self) -> Result<()> {
        let Some(file) = &self.file else { return Ok(()) };
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        let tmp = file.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, file).with_context(|| format!("replace {}", file.display()))?;
        Ok(())
    }

    pub fn get(&self, path: &Path) -> Option<Grade> {
        self.overrides.get(&key(path)).map(|o| o.grade)
    }

    /// Set (Some) or clear (None) the override for a file.
    pub fn set(&mut self, path: &Path, grade: Option<Grade>) {
        match grade {
            Some(g) => {
                let at = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
                self.overrides.insert(key(path), Override { grade: g, at });
            }
            None => {
                self.overrides.remove(&key(path));
            }
        }
    }

    /// A file moved: its override and panel correction go with it.
    pub fn rename(&mut self, from: &Path, to: &Path) -> bool {
        let mut changed = false;
        if let Some(o) = self.overrides.remove(&key(from)) {
            self.overrides.insert(key(to), o);
            changed = true;
        }
        if let Some(p) = self.panels.remove(&key(from)) {
            self.panels.insert(key(to), p);
            changed = true;
        }
        changed
    }

    /// Take in another folder's decisions (folders combined into one):
    /// per-file pins and panels, flat and dark picks, dismissed items and
    /// views are added; your threshold changes and layout only when this
    /// folder has none. Returns whether anything changed.
    pub fn absorb(&mut self, other: &Decisions) -> bool {
        let before = serde_json::to_string(&*self).unwrap_or_default();
        for (k, v) in &other.overrides {
            self.overrides.entry(k.clone()).or_insert_with(|| v.clone());
        }
        for (k, v) in &other.panels {
            self.panels.entry(k.clone()).or_insert(*v);
        }
        for (k, v) in &other.flats {
            self.flats.entry(k.clone()).or_insert_with(|| v.clone());
        }
        for (k, v) in &other.offsets {
            self.offsets.entry(k.clone()).or_insert_with(|| v.clone());
        }
        self.dismissed.extend(other.dismissed.iter().cloned());
        for v in &other.views {
            if !self.views.iter().any(|x| x.name == v.name) {
                self.views.push(v.clone());
            }
        }
        if self.thresholds.is_empty() {
            self.thresholds = other.thresholds.clone();
        }
        if self.layout.is_none() {
            self.layout = other.layout.clone();
        }
        serde_json::to_string(&*self).unwrap_or_default() != before
    }

    /// Corrected panel for a file, if you've set one.
    pub fn panel(&self, path: &Path) -> Option<u32> {
        self.panels.get(&key(path)).copied()
    }

    /// Your layout rules over a layout.
    pub fn apply_layout_to(&self, l: &mut crate::config::Layout) {
        if let Some(x) = &self.layout {
            *l = x.clone();
        }
    }

    /// Your layout rules over the config's.
    pub fn apply_layout(&self, cfg: &mut Config) {
        if let Some(l) = &self.layout {
            cfg.layout = l.clone();
        }
    }

    /// Apply threshold changes to a config. Raising a reject limit past its
    /// warn limit moves the warn limit with it, so nothing falls between.
    pub fn apply_thresholds(&self, cfg: &mut Config) {
        apply_thresholds(cfg, &self.thresholds);
    }

    /// Pick (or clear, with `None`) the flats for one night and filter.
    pub fn set_flat(&mut self, root: &Path, filter: &str, night: chrono::NaiveDate, pick: Option<(Option<String>, Option<String>, chrono::NaiveDate)>) {
        let k = flat_key(root, filter, night);
        match pick {
            Some((library_id, group, night)) => {
                let at = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
                self.flats.insert(k, FlatPick { library_id, group, night, at });
            }
            None => {
                self.flats.remove(&k);
            }
        }
    }

    /// Put your flat choices into crux's: the picked set becomes `chosen`
    /// (crux's pick moves to `others`) and the reason says so.
    pub fn apply_flat_choices(&self, root: &Path, cal: &mut crate::calcheck::CalCheck, g: &crate::group::Grouping, library: Option<&crate::library::Library>) {
        use crate::calcheck::FlatOption;
        for c in cal.choices.iter_mut() {
            let Some(p) = self.flats.get(&flat_key(root, &c.filter, c.night)) else { continue };
            let is = |o: &FlatOption| match (&p.library_id, &p.group) {
                (Some(id), _) => o.library_id.as_deref() == Some(id.as_str()),
                (None, Some(label)) => o.group.is_some_and(|gid| g.groups.get(gid).is_some_and(|gr| &gr.label == label)),
                _ => false,
            };
            if c.chosen.as_ref().is_some_and(is) {
                c.pinned = true;
                c.reason = format!("Your choice, and crux's: {}", c.reason);
                continue;
            }
            let opt = if let Some(k) = c.others.iter().position(is) {
                c.others.remove(k)
            } else if let Some(s) = p.library_id.as_ref().and_then(|id| library.and_then(|l| l.sets.iter().find(|s| &s.id == id))) {
                FlatOption { label: format!("library {}", s.label()), group: None, library_id: Some(s.id.clone()), night: s.night, days: (s.night - c.night).num_days(), grade: s.grade, dust: None }
            } else if let Some(gr) = p.group.as_ref().and_then(|l| g.groups.iter().find(|gr| &gr.label == l)) {
                let n = gr.night.unwrap_or(p.night);
                FlatOption { label: gr.label.clone(), group: Some(gr.id), library_id: None, night: n, days: (n - c.night).num_days(), grade: Grade::Good, dust: None }
            } else {
                continue; // the set has gone
            };
            let was = c.chosen.take();
            let dust = match opt.dust {
                Some(d) => format!("dust {d:.2}"),
                None => "dust not checked".into(),
            };
            c.reason = format!(
                "Your choice: {} ({:+} nights, {dust}){}",
                opt.label,
                opt.days,
                was.as_ref().map(|w| format!("; crux picked {}", w.label)).unwrap_or_default()
            );
            c.grade = if opt.grade == Grade::Reject { Grade::Warn } else { Grade::Good };
            if let Some(w) = was {
                c.others.insert(0, w);
            }
            c.chosen = Some(opt);
            c.pinned = true;
        }
    }

    /// Apply overrides to grade reports: a pinned reject gets a Manual flag;
    /// a pinned keep becomes Good (crux's flags stay as evidence).
    pub fn apply_overrides(&self, frames: &[Frame], reports: &mut [FrameReport]) {
        for r in reports.iter_mut() {
            let Some(f) = frames.get(r.frame) else { continue };
            match self.get(&f.meta.path) {
                Some(Grade::Reject) => {
                    r.flags.retain(|x| x.check != Check::Manual);
                    r.flags.insert(0, crate::grade::Flag { check: Check::Manual, grade: Grade::Reject, reason: "rejected by you".into() });
                    r.grade = Grade::Reject;
                }
                Some(Grade::Good) | Some(Grade::Warn) => {
                    if r.grade != Grade::Good {
                        r.notes.insert(0, format!("kept by you (crux said {})", r.grade.name()));
                    }
                    r.grade = Grade::Good;
                }
                None => {}
            }
        }
    }
}

/// Set `thresholds.<key>` values on a config (see [`Decisions::apply_thresholds`]).
pub fn apply_thresholds(cfg: &mut Config, values: &BTreeMap<String, f64>) {
    let th = &mut cfg.thresholds;
    for (k, &v) in values {
        match k.as_str() {
            "transparency.reject" => {
                th.transparency.reject = v;
                th.transparency.warn = th.transparency.warn.max(v);
            }
            "transparency.warn" => th.transparency.warn = v,
            "trails.reject_sigma" => th.trails.reject_sigma = v,
            "focus.reject_ratio" => {
                th.focus.reject_ratio = v;
                th.focus.warn_ratio = th.focus.warn_ratio.min(v);
            }
            "focus.warn_ratio" => th.focus.warn_ratio = v,
            "sky.reject_ratio" => {
                th.sky.reject_ratio = v;
                th.sky.warn_ratio = th.sky.warn_ratio.min(v);
            }
            "sky.warn_ratio" => th.sky.warn_ratio = v,
            _ => {}
        }
    }
}

/// Current value of a slider threshold.
pub fn threshold(cfg: &Config, k: &str) -> Option<f64> {
    let th = &cfg.thresholds;
    Some(match k {
        "transparency.reject" => th.transparency.reject,
        "trails.reject_sigma" => th.trails.reject_sigma,
        "focus.reject_ratio" => th.focus.reject_ratio,
        "sky.reject_ratio" => th.sky.reject_ratio,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_rename() {
        let dir = std::env::temp_dir().join(format!("crux-dec-{}", std::process::id()));
        let mut d = Decisions::load(&dir).unwrap();
        d.set(Path::new("C:/x/A.fits"), Some(Grade::Reject));
        d.thresholds.insert("transparency.reject".into(), 0.8);
        d.dismissed.insert("masters".into());
        d.save().unwrap();
        let mut e = Decisions::load(&dir).unwrap();
        assert_eq!(e.get(Path::new("C:/x/A.fits")), Some(Grade::Reject));
        // Path keys ignore case on Windows only.
        assert_eq!(e.get(Path::new("c:/x/a.fits")).is_some(), cfg!(windows));
        assert!(e.rename(Path::new("C:/x/A.fits"), Path::new("C:/x/BAD/A.fits")));
        assert_eq!(e.get(Path::new("C:/x/BAD/A.fits")), Some(Grade::Reject));
        let mut cfg = Config::default();
        e.apply_thresholds(&mut cfg);
        assert_eq!(cfg.thresholds.transparency.reject, 0.8);
        assert!(cfg.thresholds.transparency.warn >= 0.8);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flat_choice_swaps_and_reverts() {
        use crate::calcheck::{CalCheck, FlatChoice, FlatOption};
        let n = |d: &str| d.parse::<chrono::NaiveDate>().unwrap();
        let opt = |id: &str, night: &str, days: i64| FlatOption { label: format!("library {id}"), group: None, library_id: Some(id.into()), night: n(night), days, grade: Grade::Good, dust: Some(0.4) };
        let choice = FlatChoice {
            filter: "H".into(),
            night: n("2026-11-09"),
            binning: 1,
            light_groups: vec![],
            n_lights: 56,
            n_stacked: 56,
            chosen: Some(opt("H_16", "2026-11-16", 7)),
            others: vec![opt("H_01", "2026-11-01", -8)],
            grade: Grade::Good,
            reason: "crux".into(),
            pinned: false,
        };
        let cal0 = CalCheck { choices: vec![choice], ..Default::default() };
        let root = Path::new("lmc");
        let g = crate::group::Grouping::default();
        let mut d = Decisions::default();
        d.set_flat(root, "h", n("2026-11-09"), Some((Some("H_01".into()), None, n("2026-11-01"))));
        let mut cal = cal0.clone();
        d.apply_flat_choices(root, &mut cal, &g, None);
        let c = &cal.choices[0];
        assert!(c.pinned);
        assert_eq!(c.chosen.as_ref().unwrap().library_id.as_deref(), Some("H_01"));
        assert_eq!(c.others[0].library_id.as_deref(), Some("H_16"));
        assert!(c.reason.starts_with("Your choice: library H_01"), "{}", c.reason);
        // Another folder's choice doesn't apply here.
        let mut other = cal0.clone();
        d.apply_flat_choices(Path::new("m42"), &mut other, &g, None);
        assert!(!other.choices[0].pinned);
        // Cleared: crux's choice again.
        d.set_flat(root, "H", n("2026-11-09"), None);
        let mut cal = cal0.clone();
        d.apply_flat_choices(root, &mut cal, &g, None);
        assert_eq!(cal.choices[0].chosen.as_ref().unwrap().library_id.as_deref(), Some("H_16"));
    }
}
