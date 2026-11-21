//! The flat library: flat sets reused across nights, wherever their files
//! are.
//!
//! Kept in SQLite (`index.db` in `[library] dir`, `%LOCALAPPDATA%\crux` by
//! default: the same database as the image index, in its own tables). Each
//! set records its filter, night, binning, size, frame count, grade, where
//! its files are, and where they'd go if filed (`<root>\<filter>\<night>\`;
//! moving them there is a plan action). Each set's median flat, 32×
//! binned, is stored with it; the dust fingerprint (band-passed) and
//! set-to-set agreement come from it.

use crate::calib::{self, Map};
use crate::config::Config;
use crate::grade::Grade;
use anyhow::{bail, Context, Result};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const LIBRARY_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibrarySet {
    pub id: String,
    pub filter: String,
    /// Night the flats belong to (dawn flats count with the evening before).
    pub night: NaiveDate,
    pub binning: u32,
    pub width: u32,
    pub height: u32,
    pub frames: usize,
    /// A single integrated master (WBPP) rather than raw frames.
    pub master: bool,
    /// In a reject folder (FLAT.bad): kept in the index but never matched.
    pub retired: bool,
    pub grade: Grade,
    pub reasons: Vec<String>,
    pub exposure_s: Option<f64>,
    /// Median level after the offset, fraction of full scale.
    pub level: Option<f64>,
    /// Worst frame change from the set's median, %.
    pub worst_change: Option<f64>,
    /// Error left in the lights it was checked against, % peak-to-peak.
    pub error_vs_lights: Option<f64>,
    pub instrument: Option<String>,
    pub telescope: Option<String>,
    /// Where the files are now.
    pub files: Vec<PathBuf>,
    /// Where they belong in the library.
    pub home: PathBuf,
    /// When this entry was last written (local time).
    pub updated: String,
    /// Other nights combined into this set.
    #[serde(default)]
    pub merged: Vec<NaiveDate>,
}

impl LibrarySet {
    pub fn label(&self) -> String {
        format!(
            "{} {}{}{}",
            self.filter,
            self.night,
            if self.master { " master" } else { "" },
            if self.retired { " (retired)" } else { "" }
        )
    }

    /// Usable for calibration.
    pub fn usable(&self) -> bool {
        !self.retired && self.grade != Grade::Reject
    }
}

/// Stable id: filter, night, binning, size, and the file name for a master.
pub fn set_id(filter: &str, night: NaiveDate, binning: u32, w: u32, h: u32, master_file: Option<&str>, retired: bool) -> String {
    let mut id = format!("{}_{}_bin{}_{}x{}", filter.to_uppercase(), night, binning, w, h);
    if let Some(f) = master_file {
        let stem: String = f.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
        id.push_str("_master_");
        id.push_str(&stem);
    }
    if retired {
        id.push_str("_retired");
    }
    id
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Library {
    pub version: u32,
    pub sets: Vec<LibrarySet>,
    /// Folders holding calibration frames, from the last refresh. Darks,
    /// bias and dark flats are found here when a folder has none.
    #[serde(default)]
    pub folders: Vec<CalFolder>,
    /// When the library was last refreshed (local time).
    #[serde(default)]
    pub refreshed: Option<String>,
    /// The folders that refresh searched.
    #[serde(default)]
    pub searched: Vec<PathBuf>,
    /// Where `--to-library` files sets (`[library] root`).
    #[serde(skip)]
    pub root: PathBuf,
    /// The folder holding the database (`[library] dir`).
    #[serde(skip)]
    pub dir: PathBuf,
    /// Reference maps added or changed since loading; `save` writes them.
    #[serde(skip)]
    pending: HashMap<String, Map>,
}

/// A folder with calibration frames in it (raw frames and masters counted
/// together; frames in reject folders left out).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalFolder {
    pub folder: PathBuf,
    pub flats: usize,
    pub darks: usize,
    pub bias: usize,
    pub dark_flats: usize,
    pub masters: usize,
}

impl CalFolder {
    /// Has darks, bias or dark flats.
    pub fn has_offsets(&self) -> bool {
        self.darks + self.bias + self.dark_flats > 0
    }
}

/// How a set compares with the other sets of its filter that share its dust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Consensus {
    /// Median difference from them, % peak-to-peak.
    pub median: f64,
    pub closest: f64,
    pub n: usize,
    pub grade: Grade,
}

impl Consensus {
    pub fn reason(&self) -> String {
        format!(
            "differs from the {} other sets with the same dust by a median {:.1}% (closest {:.1}%) — cloud or a twilight gradient in every frame?",
            self.n, self.median, self.closest
        )
    }
}

/// Grade a set from its differences with same-dust sets (needs at least 3).
pub fn consensus_of(diffs: &[f64], cfg: &Config) -> Option<Consensus> {
    let th = &cfg.thresholds.calibration;
    if diffs.len() < 3 {
        return None;
    }
    let median = crate::stats::median_f64(diffs)?;
    let closest = diffs.iter().copied().fold(f64::INFINITY, f64::min);
    let grade = if median > th.consensus_reject_pct {
        Grade::Reject
    } else if median > th.consensus_warn_pct {
        Grade::Warn
    } else {
        Grade::Good
    };
    Some(Consensus { median, closest, n: diffs.len(), grade })
}

/// Two library sets of the same filter, binning and size compared.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryPair {
    pub a: String,
    pub b: String,
    /// Smoothed ratio, % peak-to-peak.
    pub spread: f64,
    /// Dust fingerprint correlation.
    pub dust: Option<f64>,
    /// Both usable, agree within `combine_pct` and share their dust.
    pub combinable: bool,
}

impl Library {
    /// The database in a library folder.
    pub fn db_path(dir: &Path) -> PathBuf {
        dir.join("index.db")
    }

    /// An empty library for these settings (nothing written until `save`).
    pub fn new(cfg: &crate::config::Library) -> Library {
        Library::new_at(&cfg.dir, &cfg.root)
    }

    pub fn new_at(dir: &Path, root: &Path) -> Library {
        Library { version: LIBRARY_VERSION, root: root.to_path_buf(), dir: dir.to_path_buf(), ..Default::default() }
    }

    /// The library these settings name, or None when there isn't one yet.
    pub fn load(cfg: &crate::config::Library) -> Result<Option<Library>> {
        Library::load_at(&cfg.dir, &cfg.root)
    }

    pub fn load_at(dir: &Path, root: &Path) -> Result<Option<Library>> {
        let p = Self::db_path(dir);
        if !p.is_file() {
            return Ok(None);
        }
        let Some(mut l) = db::load(&p).with_context(|| format!("read the flat library in {}", p.display()))? else { return Ok(None) };
        l.root = root.to_path_buf();
        l.dir = dir.to_path_buf();
        Ok(Some(l))
    }

    /// Write the sets, folders and new reference maps (one transaction);
    /// maps of sets no longer in the library are dropped.
    pub fn save(&self) -> Result<()> {
        std::fs::create_dir_all(&self.dir).with_context(|| format!("create {}", self.dir.display()))?;
        let p = Self::db_path(&self.dir);
        db::save(&p, self, &self.pending).with_context(|| format!("write the flat library to {}", p.display()))
    }

    /// The whole library, maps included, in a database file of its own
    /// (for undo).
    pub fn snapshot(&self, file: &Path) -> Result<()> {
        if file.exists() {
            std::fs::remove_file(file).with_context(|| format!("replace {}", file.display()))?;
        }
        db::save(file, self, &self.references()).with_context(|| format!("write {}", file.display()))
    }

    /// Put a snapshot back as the library in `dir`.
    pub fn restore(file: &Path, dir: &Path) -> Result<()> {
        let mut l = db::load(file)?.with_context(|| format!("{} holds no flat library", file.display()))?;
        l.pending = db::all_refs(file)?;
        l.dir = dir.to_path_buf();
        l.save()
    }

    /// The set's median flat (median 1), 32× binned.
    pub fn reference(&self, id: &str) -> Option<Map> {
        if let Some(m) = self.pending.get(id) {
            return Some(m.clone());
        }
        db::read_ref(&Self::db_path(&self.dir), id).ok().flatten()
    }

    /// Every set's reference map, read in one go.
    pub fn references(&self) -> HashMap<String, Map> {
        let mut out = if self.dir.as_os_str().is_empty() { HashMap::new() } else { db::all_refs(&Self::db_path(&self.dir)).unwrap_or_default() };
        out.retain(|id, _| self.sets.iter().any(|s| &s.id == id));
        for (k, v) in &self.pending {
            out.insert(k.clone(), v.clone());
        }
        out
    }

    pub fn fingerprint(&self, id: &str) -> Option<Map> {
        let r = self.reference(id)?;
        Some(Map { data: calib::dust_fingerprint(&r.data, r.w, r.h), ..r })
    }

    /// Add or replace a set (by id) and write its reference map. Returns
    /// true when the set is new. Call `save` afterwards.
    pub fn upsert(&mut self, set: LibrarySet, reference: &Map) -> Result<bool> {
        self.pending.insert(set.id.clone(), reference.clone());
        match self.sets.iter_mut().find(|s| s.id == set.id) {
            Some(s) => {
                *s = set;
                Ok(false)
            }
            None => {
                self.sets.push(set);
                self.sets.sort_by(|a, b| (&a.filter, a.night, &a.id).cmp(&(&b.filter, b.night, &b.id)));
                Ok(true)
            }
        }
    }

    /// Combine sets into the first: its files, frame count and nights grow,
    /// its reference becomes the mean of theirs, and the others leave the
    /// library. Call `save` afterwards.
    pub fn merge(&mut self, ids: &[String]) -> Result<String> {
        let Some(first) = ids.first() else { return Ok(String::new()) };
        let refs: Vec<Map> = ids.iter().filter_map(|id| self.reference(id)).collect();
        if refs.len() == ids.len() && refs.iter().all(|r| r.data.len() == refs[0].data.len()) {
            let n = refs.len() as f32;
            let mut mean = refs[0].clone();
            for (i, v) in mean.data.iter_mut().enumerate() {
                *v = refs.iter().map(|r| r.data[i]).sum::<f32>() / n;
            }
            self.pending.insert(first.clone(), mean);
        }
        let others: Vec<LibrarySet> = self.sets.iter().filter(|s| ids[1..].contains(&s.id)).cloned().collect();
        let Some(f) = self.sets.iter_mut().find(|s| &s.id == first) else { bail!("{first} isn't in the library") };
        for o in &others {
            f.files.extend(o.files.iter().cloned());
            f.frames += o.frames;
            f.merged.push(o.night);
            f.merged.extend(o.merged.iter().copied());
        }
        f.merged.sort();
        f.updated = chrono::Local::now().format("%Y-%m-%d %H:%M").to_string();
        let msg = format!("combined {} into {} ({} frames)", others.iter().map(|o| o.label()).collect::<Vec<_>>().join(", "), f.label(), f.frames);
        self.sets.retain(|s| !ids[1..].contains(&s.id));
        Ok(msg)
    }

    /// Every pair of sets with the same filter, binning and size.
    pub fn pairs(&self, cfg: &Config) -> Vec<LibraryPair> {
        let th = &cfg.thresholds.calibration;
        use rayon::prelude::*;
        let mut all = self.references();
        let refs: Vec<Option<Map>> = self.sets.iter().map(|s| all.remove(&s.id)).collect();
        // Each fingerprint once (not once per pair).
        let fps: Vec<Option<Vec<f32>>> = refs.par_iter().map(|r| r.as_ref().map(|r| calib::dust_fingerprint(&r.data, r.w, r.h))).collect();
        let mut todo = Vec::new();
        for i in 0..self.sets.len() {
            for j in i + 1..self.sets.len() {
                let (a, b) = (&self.sets[i], &self.sets[j]);
                if !a.filter.eq_ignore_ascii_case(&b.filter) || (a.binning, a.width, a.height) != (b.binning, b.width, b.height) {
                    continue;
                }
                if let (Some(ra), Some(rb)) = (&refs[i], &refs[j]) {
                    if ra.data.len() == rb.data.len() {
                        todo.push((i, j));
                    }
                }
            }
        }
        todo.par_iter()
            .map(|&(i, j)| {
                let (a, b) = (&self.sets[i], &self.sets[j]);
                let (ra, rb) = (refs[i].as_ref().unwrap(), refs[j].as_ref().unwrap());
                let (spread, _) = calib::compare_flats(&ra.data, &rb.data, ra.w, ra.h);
                let dust = calib::correlate(fps[i].as_ref().unwrap(), fps[j].as_ref().unwrap(), ra.w, ra.h);
                let combinable = a.usable() && b.usable() && !a.master && !b.master && spread <= th.combine_pct && dust.is_some_and(|d| d >= th.dust_same_train);
                LibraryPair { a: a.id.clone(), b: b.id.clone(), spread, dust, combinable }
            })
            .collect()
    }

    /// Each set against the usable sets of its filter that share its dust.
    pub fn consensus(&self, pairs: &[LibraryPair], cfg: &Config) -> std::collections::HashMap<String, Consensus> {
        let same = cfg.thresholds.calibration.dust_same_train;
        let usable = |id: &str| self.sets.iter().any(|s| s.id == id && s.usable());
        let mut diffs: std::collections::HashMap<&str, Vec<f64>> = std::collections::HashMap::new();
        for p in pairs.iter().filter(|p| p.dust.is_some_and(|d| d >= same)) {
            if usable(&p.b) {
                diffs.entry(&p.a).or_default().push(p.spread);
            }
            if usable(&p.a) {
                diffs.entry(&p.b).or_default().push(p.spread);
            }
        }
        diffs.into_iter().filter_map(|(id, v)| consensus_of(&v, cfg).map(|c| (id.to_string(), c))).collect()
    }

    /// Sets of one filter that can be combined: every pair in a group is
    /// combinable. Greedy, in night order; groups of two or more.
    pub fn combinable_groups(&self, pairs: &[LibraryPair]) -> Vec<Vec<String>> {
        let ok = |a: &str, b: &str| pairs.iter().any(|p| p.combinable && ((p.a == a && p.b == b) || (p.a == b && p.b == a)));
        let mut order: Vec<&LibrarySet> = self.sets.iter().filter(|s| s.usable() && !s.master).collect();
        order.sort_by(|a, b| (&a.filter, a.night).cmp(&(&b.filter, b.night)));
        let mut groups: Vec<Vec<String>> = Vec::new();
        let mut used = std::collections::HashSet::new();
        for s in &order {
            if used.contains(&s.id) {
                continue;
            }
            let mut g = vec![s.id.clone()];
            for t in &order {
                if t.id != s.id && !used.contains(&t.id) && t.filter == s.filter && g.iter().all(|m| ok(m, &t.id)) {
                    g.push(t.id.clone());
                }
            }
            if g.len() >= 2 {
                used.extend(g.iter().cloned());
                groups.push(g);
            }
        }
        groups
    }

    /// Usable sets grouped by optical train per filter: chained by dust
    /// correlation at or above `dust_same_train`. Each inner list is in
    /// night order. Unusable sets are left out: cloud can wreck a set's
    /// median flat, and with it the fingerprint.
    pub fn trains(&self, pairs: &[LibraryPair], cfg: &Config) -> Vec<Vec<String>> {
        let th = cfg.thresholds.calibration.dust_same_train;
        let n = self.sets.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], i: usize) -> usize {
            let mut i = i;
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let idx = |id: &str| self.sets.iter().position(|s| s.id == id);
        for p in pairs {
            if p.dust.is_some_and(|d| d >= th) {
                if let (Some(a), Some(b)) = (idx(&p.a), idx(&p.b)) {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    parent[ra] = rb;
                }
            }
        }
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
        for i in (0..n).filter(|&i| self.sets[i].usable()) {
            let r = find(&mut parent, i);
            groups.entry(r).or_default().push(i);
        }
        let mut out: Vec<Vec<String>> = groups
            .into_values()
            .map(|mut v| {
                v.sort_by_key(|&i| self.sets[i].night);
                v.into_iter().map(|i| self.sets[i].id.clone()).collect()
            })
            .collect();
        out.sort_by_key(|v| v.first().and_then(|id| self.sets.iter().find(|s| &s.id == id)).map(|s| (s.filter.clone(), s.night)));
        out
    }
}

/// Where a set belongs: `<root>\<FILTER>\<night>\`, or under `FLAT.bad` when retired.
pub fn home_for(root: &Path, filter: &str, night: NaiveDate, retired: bool) -> PathBuf {
    let base = if retired { root.join("FLAT.bad") } else { root.to_path_buf() };
    base.join(filter.to_uppercase()).join(night.to_string())
}

/// Library entries for every flat set in a triage: raw sets, plus masters
/// that weren't traced to a raw set in the same scan. Each comes with its
/// reference map for `Library::upsert`.
pub fn sets_from_triage(t: &crate::pipeline::Triage, root: &Path) -> Vec<(LibrarySet, Map)> {
    let g = &t.grouping;
    let now = chrono::Local::now().format("%Y-%m-%d %H:%M").to_string();
    let mut out = Vec::new();
    for s in &t.cal.sets {
        let gr = &g.groups[s.group];
        if gr.kind != crate::classify::FrameType::Flat || (gr.master && gr.built_from.is_some()) {
            continue;
        }
        let (Some((r, w, h)), Some(night)) = (t.cal.references.get(&gr.id), gr.night) else { continue };
        let filter = gr.filter.clone().unwrap_or_else(|| "none".into()).to_uppercase();
        let f0 = &t.scan.frames[gr.members[0]].meta;
        let master_file = gr.master.then(|| f0.file_name().to_string());
        let binning = gr.binning.unwrap_or(1);
        let set = LibrarySet {
            id: set_id(&filter, night, binning, gr.width, gr.height, master_file.as_deref(), gr.retired),
            filter: filter.clone(),
            night,
            binning,
            width: gr.width,
            height: gr.height,
            frames: gr.members.len(),
            master: gr.master,
            retired: gr.retired,
            grade: s.grade,
            reasons: s.flags.iter().filter(|f| f.grade != Grade::Good).map(|f| f.reason.clone()).collect(),
            exposure_s: gr.exptime,
            level: s.flat_level,
            worst_change: s.worst_change,
            error_vs_lights: s.vs_lights.as_ref().filter(|v| v.reliable).map(|v| v.error.spread),
            instrument: f0.instrume.clone(),
            telescope: f0.telescop.clone(),
            files: gr.members.iter().map(|&i| t.scan.frames[i].meta.path.clone()).collect(),
            home: home_for(root, &filter, night, gr.retired),
            updated: now.clone(),
            merged: Vec::new(),
        };
        out.push((set, Map { w: *w, h: *h, data: r.clone() }));
    }
    out
}

// ---------------------------------------------------------------- refresh

/// What a refresh is doing, for progress lines.
#[derive(Debug, Clone)]
pub enum RefreshEvent {
    /// Looking at folder `index` (from 1) of `total`.
    Folder { index: usize, total: usize, dir: PathBuf },
    /// Calibration frames measured (or read from the cache) in this folder.
    Progress { done: usize, total: usize },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RefreshReport {
    pub searched: Vec<PathBuf>,
    /// Folders looked at.
    pub folders: usize,
    /// The ones with calibration frames.
    pub cal_folders: Vec<CalFolder>,
    pub added: Vec<String>,
    pub updated: usize,
    /// Sets whose files are gone, or that a rescan no longer finds.
    pub removed: Vec<String>,
    pub sets: usize,
    pub usable: usize,
    /// Calibration frames measured this time (the rest came from caches).
    pub measured: usize,
    /// Image files sitting directly in a searched folder: not scanned.
    pub loose: Vec<(PathBuf, usize)>,
    pub errors: Vec<String>,
    pub seconds: f64,
    pub index: PathBuf,
}

impl RefreshReport {
    pub fn summary(&self) -> String {
        let mut changes = String::new();
        if !self.added.is_empty() {
            changes.push_str(&format!(", {} new", self.added.len()));
        }
        if !self.removed.is_empty() {
            changes.push_str(&format!(", {} removed", self.removed.len()));
        }
        let offsets = self.cal_folders.iter().filter(|f| f.has_offsets()).count();
        format!(
            "{} flat sets ({} usable{changes}) · darks, bias or dark flats in {offsets} folder{} · {} folders searched in {:.0} s",
            self.sets,
            self.usable,
            if offsets == 1 { "" } else { "s" },
            self.folders,
            self.seconds
        )
    }
}

/// The folders a refresh looks in: each search root's subfolders (not
/// hidden, `_…` or WBPP output), plus the library root when it isn't
/// inside one of them.
pub fn refresh_folders(roots: &[PathBuf], library_root: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for r in roots {
        let Ok(rd) = std::fs::read_dir(r) else { continue };
        let mut v: Vec<PathBuf> = rd
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .map(|e| e.path())
            .filter(|p| !p.file_name().is_some_and(|n| crate::autopilot::hidden_or_output(&n.to_string_lossy())))
            .collect();
        v.sort();
        out.extend(v);
    }
    if library_root.is_dir() && !out.iter().any(|f| inside(library_root, f)) {
        out.push(library_root.to_path_buf());
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|p| seen.insert(crate::paths::key(p)));
    out
}

/// `p` is `dir` or inside it.
pub fn inside(p: &Path, dir: &Path) -> bool {
    PathBuf::from(crate::paths::key(p)).starts_with(PathBuf::from(crate::paths::key(dir)))
}

fn count_cal(dir: &Path, frames: &[crate::scan::Frame]) -> CalFolder {
    use crate::classify::{FrameType, Role};
    let mut c = CalFolder { folder: dir.to_path_buf(), ..Default::default() };
    for f in frames {
        if f.class.in_reject_dir || f.class.role == Role::Processed {
            continue;
        }
        let n = match f.class.frame_type {
            FrameType::Flat => &mut c.flats,
            FrameType::Dark => &mut c.darks,
            FrameType::Bias => &mut c.bias,
            FrameType::DarkFlat => &mut c.dark_flats,
            _ => continue,
        };
        *n += 1;
        if f.class.role == Role::Master {
            c.masters += 1;
        }
    }
    c
}

/// Scan every folder in `roots` for calibration frames, in place: flat sets
/// are measured, checked and added to (or updated in) the library
/// (`[library] dir`); folders with darks, bias or dark flats are listed
/// for processing to search. Nothing in the data folders changes. Each
/// folder's measurements go to its own cache (under `cache_root`, one
/// subfolder each, when given), so a refresh after the first is quick.
pub fn refresh(
    cfg: &Config,
    roots: &[PathBuf],
    cache_root: Option<&Path>,
    progress: &(dyn Fn(RefreshEvent) + Sync),
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<RefreshReport> {
    use std::sync::atomic::Ordering;
    let t0 = std::time::Instant::now();
    let root = &cfg.library.root;
    let mut lib = Library::load(&cfg.library)?.unwrap_or_else(|| Library::new(&cfg.library));
    let folders = refresh_folders(roots, root);
    let mut rep = RefreshReport { searched: roots.to_vec(), folders: folders.len(), index: Library::db_path(&cfg.library.dir), ..Default::default() };
    for r in roots {
        let n = std::fs::read_dir(r)
            .map(|rd| {
                rd.flatten()
                    .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
                    .filter(|e| e.path().extension().and_then(|x| x.to_str()).is_some_and(|x| cfg.scan.extensions.iter().any(|w| w.eq_ignore_ascii_case(x))))
                    .count()
            })
            .unwrap_or(0);
        if n > 0 {
            rep.loose.push((r.clone(), n));
        }
    }
    // Files of combined sets: their nights' sets aren't added back one by one.
    let merged: std::collections::HashSet<String> = lib.sets.iter().filter(|s| !s.merged.is_empty()).flat_map(|s| s.files.iter().map(|f| crate::paths::key(f))).collect();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut cal_folders = Vec::new();
    for (k, dir) in folders.iter().enumerate() {
        anyhow::ensure!(!cancel.load(Ordering::Relaxed), "cancelled");
        progress(RefreshEvent::Folder { index: k + 1, total: folders.len(), dir: dir.clone() });
        let mut nc = crate::cache::Cache::disabled();
        let h = crate::pipeline::run(dir, cfg, &mut nc, false, &|_, _, _| {});
        let c = count_cal(dir, &h.scan.frames);
        if c.flats + c.darks + c.bias + c.dark_flats == 0 {
            continue;
        }
        let cdir = crate::autopilot::cache_for(cache_root, dir, cfg);
        let mut cache = crate::cache::Cache::load(&cdir);
        let t = crate::pipeline::run_calibration(
            dir,
            cfg,
            &mut cache,
            &|st, done, total| {
                if st == crate::pipeline::Stage::Calibration {
                    progress(RefreshEvent::Progress { done, total });
                }
            },
            cancel,
        );
        let _ = cache.save();
        anyhow::ensure!(!cancel.load(Ordering::Relaxed), "cancelled");
        rep.measured += t.cal_analysed;
        let mut errs: Vec<(&usize, &String)> = t.cal_errors.iter().collect();
        errs.sort();
        rep.errors.extend(errs.into_iter().map(|(i, e)| format!("{}: {e}", t.scan.frames[*i].meta.path.display())));
        cal_folders.push(count_cal(dir, &t.scan.frames));
        for (mut set, map) in sets_from_triage(&t, root) {
            if !merged.is_empty() && set.files.iter().all(|f| merged.contains(&crate::paths::key(f))) {
                continue;
            }
            // A rescan without the lights keeps what the lights said.
            if let Some(old) = lib.sets.iter().find(|s| s.id == set.id) {
                let same = old.files.len() == set.files.len() && old.files.iter().zip(&set.files).all(|(a, b)| crate::paths::key(a) == crate::paths::key(b));
                if same && set.error_vs_lights.is_none() && old.error_vs_lights.is_some() {
                    set.error_vs_lights = old.error_vs_lights;
                    set.grade = set.grade.max(old.grade);
                    for r in &old.reasons {
                        if !set.reasons.contains(r) {
                            set.reasons.push(r.clone());
                        }
                    }
                }
            }
            seen.insert(set.id.clone());
            let label = set.label();
            if lib.upsert(set, &map)? {
                rep.added.push(label);
            } else {
                rep.updated += 1;
            }
        }
    }
    trace_masters(&mut lib.sets);
    // Sets whose files are gone, or that a scanned folder no longer yields.
    let removed = &mut rep.removed;
    lib.sets.retain(|s| {
        let gone = s.files.iter().all(|f| !f.exists());
        let stale = s.merged.is_empty() && !seen.contains(&s.id) && !s.files.is_empty() && s.files.iter().all(|f| folders.iter().any(|d| inside(f, d)));
        if gone || stale {
            removed.push(s.label());
        }
        !(gone || stale)
    });
    // Folders added by hand outside the search keep their entries.
    let mut keep: Vec<CalFolder> = lib.folders.iter().filter(|f| f.folder.is_dir() && !folders.iter().any(|d| inside(&f.folder, d))).cloned().collect();
    keep.extend(cal_folders.iter().cloned());
    keep.sort_by_key(|f| crate::paths::key(&f.folder));
    lib.folders = keep;
    lib.refreshed = Some(chrono::Local::now().format("%Y-%m-%d %H:%M").to_string());
    lib.searched = roots.to_vec();
    lib.save()?;
    rep.cal_folders = cal_folders;
    rep.sets = lib.sets.len();
    let cons = lib.consensus(&lib.pairs(cfg), cfg);
    rep.usable = lib.sets.iter().filter(|s| s.usable() && cons.get(&s.id).is_none_or(|c| c.grade != Grade::Reject)).count();
    rep.seconds = t0.elapsed().as_secs_f64();
    Ok(rep)
}

/// Masters built elsewhere from raw flats crux rejects. WBPP masters get
/// copied between target folders (or rebuilt in another project from the
/// same raw flats), so the raw set may sit in another folder than the
/// master: a master of the same filter, binning, size and night as an
/// unusable raw set was probably built from it, and is rejected with it
/// (`group::trace_masters` does the same within a folder, by time span).
fn trace_masters(sets: &mut [LibrarySet]) {
    let bad: Vec<(String, (u32, u32, u32), NaiveDate, String, PathBuf)> = sets
        .iter()
        .filter(|s| !s.master && !s.usable())
        .map(|s| (s.filter.clone(), (s.binning, s.width, s.height), s.night, s.label(), s.files.first().and_then(|p| p.parent()).map(Path::to_path_buf).unwrap_or_default()))
        .collect();
    for s in sets.iter_mut().filter(|s| s.master && s.usable()) {
        let Some(b) = bad.iter().find(|b| b.0.eq_ignore_ascii_case(&s.filter) && b.1 == (s.binning, s.width, s.height) && b.2 == s.night) else { continue };
        s.grade = Grade::Reject;
        s.reasons.push(format!("probably built from the raw {} flats in {}, which crux rejects (same filter and night)", b.3, b.4.display()));
    }
}

/// Folders the library knows hold darks, bias or dark flats.
pub fn offset_folders(cfg: &Config) -> Vec<PathBuf> {
    Library::load(&cfg.library)
        .ok()
        .flatten()
        .map(|l| l.folders.into_iter().filter(|f| f.has_offsets() && f.folder.is_dir()).map(|f| f.folder).collect())
        .unwrap_or_default()
}

#[cfg(feature = "sqlite")]
mod db {
    //! SQLite storage, in tables of their own beside the image index's.
    //! Sets and folders are JSON in a column; maps are blobs.
    use super::*;
    use rusqlite::{params, Connection, OptionalExtension};

    const SCHEMA: &str = "
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS library_sets (id TEXT PRIMARY KEY, filter TEXT NOT NULL, night TEXT NOT NULL, record TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS library_folders (key TEXT PRIMARY KEY, record TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS library_refs (id TEXT PRIMARY KEY, map BLOB NOT NULL);
    ";

    fn open(p: &Path) -> Result<Connection> {
        let c = Connection::open(p)?;
        c.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        c.execute_batch(SCHEMA)?;
        Ok(c)
    }

    fn meta(c: &Connection, key: &str) -> Result<Option<String>> {
        Ok(c.query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0)).optional()?)
    }

    /// None when the database holds no library (or an older version of one).
    pub fn load(p: &Path) -> Result<Option<Library>> {
        let c = open(p)?;
        let Some(v) = meta(&c, "library_version")? else { return Ok(None) };
        if v.parse::<u32>().ok() != Some(LIBRARY_VERSION) {
            return Ok(None);
        }
        let mut l = Library { version: LIBRARY_VERSION, ..Default::default() };
        l.refreshed = meta(&c, "library_refreshed")?;
        l.searched = meta(&c, "library_searched")?.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        let mut st = c.prepare("SELECT record FROM library_sets ORDER BY filter, night, id")?;
        for r in st.query_map([], |r| r.get::<_, String>(0))? {
            l.sets.push(serde_json::from_str(&r?).context("a library set")?);
        }
        let mut st = c.prepare("SELECT record FROM library_folders ORDER BY key")?;
        for r in st.query_map([], |r| r.get::<_, String>(0))? {
            l.folders.push(serde_json::from_str(&r?).context("a library folder")?);
        }
        Ok(Some(l))
    }

    pub fn save(p: &Path, l: &Library, maps: &HashMap<String, Map>) -> Result<()> {
        let mut c = open(p)?;
        let tx = c.transaction()?;
        let put = |k: &str, v: Option<String>| -> Result<()> {
            match v {
                Some(v) => tx.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)", params![k, v])?,
                None => tx.execute("DELETE FROM meta WHERE key = ?1", [k])?,
            };
            Ok(())
        };
        put("library_version", Some(LIBRARY_VERSION.to_string()))?;
        put("library_refreshed", l.refreshed.clone())?;
        put("library_searched", Some(serde_json::to_string(&l.searched)?))?;
        tx.execute("DELETE FROM library_sets", [])?;
        tx.execute("DELETE FROM library_folders", [])?;
        {
            let mut st = tx.prepare("INSERT OR REPLACE INTO library_sets (id, filter, night, record) VALUES (?1, ?2, ?3, ?4)")?;
            for s in &l.sets {
                st.execute(params![s.id, s.filter, s.night.to_string(), serde_json::to_string(s)?])?;
            }
            let mut st = tx.prepare("INSERT OR REPLACE INTO library_folders (key, record) VALUES (?1, ?2)")?;
            for f in &l.folders {
                st.execute(params![crate::paths::key(&f.folder), serde_json::to_string(f)?])?;
            }
            let mut st = tx.prepare("INSERT OR REPLACE INTO library_refs (id, map) VALUES (?1, ?2)")?;
            for (id, m) in maps.iter().filter(|(id, _)| l.sets.iter().any(|s| &s.id == *id)) {
                st.execute(params![id, m.to_bytes()])?;
            }
        }
        tx.execute("DELETE FROM library_refs WHERE id NOT IN (SELECT id FROM library_sets)", [])?;
        tx.commit()?;
        Ok(())
    }

    pub fn read_ref(p: &Path, id: &str) -> Result<Option<Map>> {
        if !p.is_file() {
            return Ok(None);
        }
        let c = open(p)?;
        let b: Option<Vec<u8>> = c.query_row("SELECT map FROM library_refs WHERE id = ?1", [id], |r| r.get(0)).optional()?;
        Ok(b.and_then(|b| Map::from_bytes(&b)))
    }

    pub fn all_refs(p: &Path) -> Result<HashMap<String, Map>> {
        if !p.is_file() {
            return Ok(HashMap::new());
        }
        let c = open(p)?;
        let mut st = c.prepare("SELECT id, map FROM library_refs")?;
        let rows = st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))?;
        let mut out = HashMap::new();
        for r in rows {
            let (id, b) = r?;
            if let Some(m) = Map::from_bytes(&b) {
                out.insert(id, m);
            }
        }
        Ok(out)
    }
}

/// Without SQLite there's no flat library yet (Linux and macOS builds,
/// until SQLite is built there).
#[cfg(not(feature = "sqlite"))]
mod db {
    use super::*;
    pub fn load(_: &Path) -> Result<Option<Library>> {
        Ok(None)
    }
    pub fn save(_: &Path, _: &Library, _: &HashMap<String, Map>) -> Result<()> {
        bail!("this build of crux has no SQLite, which the flat library needs")
    }
    pub fn read_ref(_: &Path, _: &str) -> Result<Option<Map>> {
        Ok(None)
    }
    pub fn all_refs(_: &Path) -> Result<HashMap<String, Map>> {
        Ok(HashMap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(filter: &str, day: u32, master: bool, grade: Grade, dir: &str) -> LibrarySet {
        let night = NaiveDate::from_ymd_opt(2026, 10, day).unwrap();
        LibrarySet {
            id: set_id(filter, night, 1, 6248, 4176, master.then_some("m.xisf"), false),
            filter: filter.into(),
            night,
            binning: 1,
            width: 6248,
            height: 4176,
            frames: if master { 1 } else { 21 },
            master,
            retired: false,
            grade,
            reasons: Vec::new(),
            exposure_s: None,
            level: None,
            worst_change: None,
            error_vs_lights: None,
            instrument: None,
            telescope: None,
            files: vec![PathBuf::from(dir).join("f.xisf")],
            home: PathBuf::new(),
            updated: String::new(),
            merged: Vec::new(),
        }
    }

    #[test]
    fn masters_from_rejected_raw_sets_elsewhere_are_rejected() {
        let mut sets = vec![
            set("H", 10, false, Grade::Reject, "a"),
            set("H", 10, true, Grade::Good, "b"),
            set("O", 10, true, Grade::Good, "b"),
            set("H", 8, true, Grade::Good, "b"),
        ];
        trace_masters(&mut sets);
        assert_eq!(sets[1].grade, Grade::Reject);
        assert!(sets[1].reasons[0].contains("H 2026-10-10"), "{:?}", sets[1].reasons);
        assert_eq!((sets[2].grade, sets[3].grade), (Grade::Good, Grade::Good));
    }

    /// Sets and maps round-trip through the database, which the image index
    /// shares; a snapshot puts a merged library back as it was.
    #[cfg(feature = "sqlite")]
    #[test]
    fn database_round_trip_merge_and_restore() {
        let dir = std::env::temp_dir().join(format!("crux-libdb-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let map = |v: f32| Map { w: 2, h: 2, data: vec![v; 4] };
        let mut lib = Library::new_at(&dir, &dir.join("filed"));
        let (a, b) = (set("H", 8, false, Grade::Good, "a"), set("H", 9, false, Grade::Good, "b"));
        let (ia, ib) = (a.id.clone(), b.id.clone());
        assert!(lib.upsert(a, &map(1.0)).unwrap());
        assert!(lib.upsert(b, &map(3.0)).unwrap());
        lib.folders.push(CalFolder { folder: dir.join("darks"), darks: 5, ..Default::default() });
        lib.save().unwrap();

        // The image index in the same file leaves the library alone.
        let mut ix = crate::index::Index::load(&dir.join("index.json")).unwrap();
        ix.set_root(&dir.join("data"), Vec::new(), false);
        ix.save().unwrap();

        let mut l = Library::load_at(&dir, &dir.join("filed")).unwrap().unwrap();
        assert_eq!((l.sets.len(), l.folders.len()), (2, 1));
        assert_eq!(l.reference(&ib), Some(map(3.0)));
        assert_eq!(l.pairs(&Config::default()).len(), 1);

        let snap = dir.join("before.library.db");
        l.snapshot(&snap).unwrap();
        l.merge(&[ia.clone(), ib.clone()]).unwrap();
        l.save().unwrap();
        let l = Library::load_at(&dir, &dir).unwrap().unwrap();
        assert_eq!(l.sets.len(), 1);
        assert_eq!(l.reference(&ia), Some(map(2.0)));
        assert_eq!(l.reference(&ib), None, "a merged set's map goes with it");

        Library::restore(&snap, &dir).unwrap();
        let l = Library::load_at(&dir, &dir).unwrap().unwrap();
        assert_eq!(l.sets.len(), 2);
        assert_eq!((l.reference(&ia), l.reference(&ib)), (Some(map(1.0)), Some(map(3.0))));
        assert!(crate::index::Index::load(&dir.join("index.json")).unwrap().roots.len() == 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn inside_is_by_component() {
        assert!(inside(Path::new("C:/data/a/b.fits"), Path::new("C:/data/a")));
        assert!(!inside(Path::new("C:/data/ab/b.fits"), Path::new("C:/data/a")));
    }
}
