//! Targets across the library (milestone 15).
//!
//! The library is the folders you choose (`[library] search`): `C:\PixInsight`,
//! or `C:\2025` and `C:\2026`, or `\\nas\astronomy\2025`… crux never looks
//! outside them. Each folder directly inside one is a *unit*: normally one
//! target (`2026-10-05_M 17`), sometimes one night of it, or calibration
//! frames only (`2026-10-08 FLATS`).
//!
//! A target is the lights' OBJECT without the panel ("LMC" for "LMC Panel
//! 26"), with catalogue names written one way ("M42", "m 42" and "M 42" are
//! all "M 42"). Where the lights point checks the names: one name pointing
//! at two places, or two names on one field, get a note.
//!
//! Combining: a target found in several units of one library folder moves
//! into `<library folder>\<target>\<unit>\…`, through the usual plan, apply
//! and undo. Calibration frames stay where they are: the calibration
//! library finds them wherever they are (`library::refresh`).

use crate::classify::{FrameType, Role};
use crate::config::Config;
use crate::pipeline::Triage;
use crate::plan::{Move, MoveKind, Plan, PLAN_VERSION};
use anyhow::{bail, Result};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// Two different names closer than this are probably one target.
const SAME_FIELD_DEG: f64 = 0.25;
/// One name in two units further apart than this is probably two targets.
const APART_DEG: f64 = 2.0;

/// One folder directly inside a library folder.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Unit {
    pub path: PathBuf,
    /// The library folder it's in.
    pub root: PathBuf,
    pub name: String,
    pub files: usize,
    /// Raw lights outside reject folders, by target.
    pub targets: BTreeMap<String, usize>,
    /// Raw lights in reject folders (BAD, *.bad).
    pub rejected: usize,
    pub flats: usize,
    pub darks: usize,
    pub bias: usize,
    pub dark_flats: usize,
    pub masters: usize,
}

impl Unit {
    pub fn lights(&self) -> usize {
        self.targets.values().sum()
    }
    pub fn calibration(&self) -> usize {
        self.flats + self.darks + self.bias + self.dark_flats
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Target {
    pub name: String,
    /// The OBJECT values seen ("LMC Panel 1" … "LMC Panel 36").
    pub objects: Vec<String>,
    /// Units holding its lights (indices into `LibraryTargets::units`).
    pub units: Vec<usize>,
    pub lights: usize,
    /// Seconds of raw lights by filter (upper case).
    pub secs: BTreeMap<String, f64>,
    pub nights: Vec<NaiveDate>,
    /// Where the lights point (RA, Dec in degrees): plate solutions where
    /// crux has them, else the headers.
    pub centre: Option<(f64, f64)>,
    pub notes: Vec<String>,
}

impl Target {
    pub fn hours(&self) -> f64 {
        self.secs.values().sum::<f64>() / 3600.0
    }
    pub fn panels(&self) -> usize {
        self.objects.iter().filter(|o| crate::index::panel_of(o).is_some()).count()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LibraryTargets {
    pub roots: Vec<PathBuf>,
    pub units: Vec<Unit>,
    pub targets: Vec<Target>,
    /// Image files directly in a library folder (not in any unit).
    pub loose: Vec<PathBuf>,
    pub notes: Vec<String>,
}

/// What combining a target would do.
#[derive(Debug, Clone, Default)]
pub struct Combine {
    pub root: PathBuf,
    /// `<root>\<target>`.
    pub dest: PathBuf,
    /// Units that move into `dest`.
    pub moving: Vec<usize>,
    /// A unit that already is `dest` (an earlier combine).
    pub existing: Option<usize>,
    /// Units that can't move, and why.
    pub blocked: Vec<(usize, String)>,
}

/// The target name, catalogue numbers written one way: "M42", "m 42",
/// "M_42" → "M 42"; "NGC2070" → "NGC 2070"; "sh2-240" → "Sh2-240".
/// Other names keep their spelling (spaces tidied).
pub fn normalise(object: &str) -> String {
    let t = crate::index::target_of(&object.replace('_', " "));
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    let low = t.to_lowercase();
    // Longest prefixes first, so "messier" wins over "m".
    const CATS: &[(&str, &str)] = &[
        ("messier", "M"),
        ("caldwell", "C"),
        ("barnard", "B"),
        ("abell", "Abell"),
        ("sh2-", "Sh2-"),
        ("sh2", "Sh2-"),
        ("sh 2-", "Sh2-"),
        ("ngc", "NGC"),
        ("ldn", "LDN"),
        ("lbn", "LBN"),
        ("vdb", "vdB"),
        ("ced", "Ced"),
        ("rcw", "RCW"),
        ("ugc", "UGC"),
        ("pgc", "PGC"),
        ("arp", "Arp"),
        ("mel", "Mel"),
        ("gum", "Gum"),
        ("ic", "IC"),
        ("cr", "Cr"),
        ("m", "M"),
        ("c", "C"),
        ("b", "B"),
    ];
    for (p, canon) in CATS {
        let Some(rest) = low.strip_prefix(p) else { continue };
        let rest = rest.trim_start_matches([' ', '-']);
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            continue;
        }
        let tail = &rest[digits.len()..];
        // Allow one letter after the number ("NGC 6188b"); nothing else.
        let suffix = match tail.len() {
            0 => String::new(),
            1 if tail.chars().all(|c| c.is_ascii_alphabetic()) => tail.to_string(),
            _ => continue,
        };
        let num = digits.trim_start_matches('0');
        let num = if num.is_empty() { "0" } else { num };
        return if canon.ends_with('-') { format!("{canon}{num}{suffix}") } else { format!("{canon} {num}{suffix}") };
    }
    t
}

/// One name per field for a folder's OBJECT spellings: "NGC3372" and
/// "NGC 3372" (one ASIAIR plan typed twice) are one field, named after the
/// spelling most frames carry, so a folder with one spelling keeps its
/// output paths. Panels stay apart ("LMC Panel 1" ≠ "LMC Panel 2").
pub fn field_names<'a>(objects: impl Iterator<Item = &'a str>) -> HashMap<String, String> {
    let key = |o: &str| (normalise(o).to_lowercase(), crate::index::panel_of(o));
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for o in objects {
        *counts.entry(o).or_default() += 1;
    }
    let mut best: HashMap<(String, Option<u32>), (&str, usize)> = HashMap::new();
    for (&o, &n) in &counts {
        let e = best.entry(key(o)).or_insert((o, n));
        if n > e.1 || (n == e.1 && o < e.0) {
            *e = (o, n);
        }
    }
    counts.keys().map(|&o| (o.to_string(), best[&key(o)].0.to_string())).collect()
}

/// A filter name written one way: "Ha"/"H-alpha" → "H", "OIII" → "O",
/// "SII" → "S", "Lum"/"Luminance" → "L", "Red" → "R"… Others upper-cased.
pub fn filter_name(f: &str) -> String {
    let u: String = f.trim().to_uppercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    match u.as_str() {
        "H" | "HA" | "HALPHA" | "HALFA" => "H",
        "O" | "OIII" | "O3" => "O",
        "S" | "SII" | "S2" => "S",
        "L" | "LUM" | "LUMINANCE" | "CLEAR" => "L",
        "R" | "RED" => "R",
        "G" | "GREEN" => "G",
        "B" | "BLUE" => "B",
        "" => "?",
        _ => return f.trim().to_uppercase(),
    }
    .to_string()
}

/// Folders directly inside a library folder that are never units: hidden,
/// `_…` (crux's `_exports`, `_Reclaimable`), WBPP outputs.
pub fn skip_unit(name: &str) -> bool {
    crate::autopilot::hidden_or_output(name) || name.starts_with('$')
}

/// Mean direction of points on the sky (RA, Dec degrees).
fn centre(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    if points.is_empty() {
        return None;
    }
    let r = std::f64::consts::PI / 180.0;
    let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
    for &(ra, dec) in points {
        x += (dec * r).cos() * (ra * r).cos();
        y += (dec * r).cos() * (ra * r).sin();
        z += (dec * r).sin();
    }
    let n = (x * x + y * y + z * z).sqrt();
    if n < 1e-9 {
        return None;
    }
    Some(((y.atan2(x) / r).rem_euclid(360.0), (z / n).asin() / r))
}

fn sep(a: (f64, f64), b: (f64, f64)) -> f64 {
    crate::astro::separation(a.0, a.1, b.0, b.1)
}

/// Targets in the library folders. `scans`: each library folder with its
/// header triage (`pipeline::run(root, …, false, …)`). `solved`: a plate
/// solution's centre for a file version, when crux has one.
pub fn find(scans: &[(PathBuf, &Triage)], solved: &dyn Fn(&Path, u64, i64) -> Option<(f64, f64)>) -> LibraryTargets {
    let mut out = LibraryTargets { roots: scans.iter().map(|s| s.0.clone()).collect(), ..Default::default() };
    let mut unit_ix: HashMap<String, usize> = HashMap::new();
    // Per target key: display names seen, objects, units, lights, secs,
    // nights, positions; per (target, unit): positions.
    #[derive(Default)]
    struct Acc {
        names: BTreeMap<String, usize>,
        objects: BTreeSet<String>,
        units: BTreeSet<usize>,
        lights: usize,
        secs: BTreeMap<String, f64>,
        nights: BTreeSet<NaiveDate>,
        points: Vec<(f64, f64)>,
        unit_points: BTreeMap<usize, Vec<(f64, f64)>>,
        object_points: BTreeMap<String, Vec<(f64, f64)>>,
    }
    let mut acc: BTreeMap<String, Acc> = BTreeMap::new();
    for (root, t) in scans {
        for f in &t.scan.frames {
            let Ok(rel) = f.meta.path.strip_prefix(root) else { continue };
            let mut comps = rel.components();
            let first = comps.next().map(|c| c.as_os_str().to_string_lossy().to_string()).unwrap_or_default();
            if comps.next().is_none() {
                out.loose.push(f.meta.path.clone());
                continue;
            }
            if skip_unit(&first) {
                continue;
            }
            let upath = root.join(&first);
            let ui = *unit_ix.entry(crate::paths::key(&upath)).or_insert_with(|| {
                out.units.push(Unit { path: upath.clone(), root: root.clone(), name: first.clone(), ..Default::default() });
                out.units.len() - 1
            });
            let u = &mut out.units[ui];
            u.files += 1;
            let c = &f.class;
            if c.role == Role::Master {
                u.masters += 1;
            }
            if c.role != Role::Raw {
                continue;
            }
            match c.frame_type {
                FrameType::Flat => u.flats += 1,
                FrameType::Dark => u.darks += 1,
                FrameType::Bias => u.bias += 1,
                FrameType::DarkFlat => u.dark_flats += 1,
                FrameType::Light if c.in_reject_dir => u.rejected += 1,
                FrameType::Light => {
                    let obj = f.meta.object.clone().unwrap_or_default();
                    let name = if obj.trim().is_empty() { u.name.clone() } else { normalise(&obj) };
                    let key = name.to_lowercase();
                    *u.targets.entry(name.clone()).or_default() += 1;
                    let a = acc.entry(key).or_default();
                    *a.names.entry(name).or_default() += 1;
                    if !obj.trim().is_empty() {
                        a.objects.insert(obj.trim().to_string());
                    }
                    a.units.insert(ui);
                    a.lights += 1;
                    *a.secs.entry(filter_name(f.meta.filter.as_deref().unwrap_or(""))).or_default() += f.meta.exptime.unwrap_or(0.0);
                    if let Some(n) = c.night {
                        a.nights.insert(n);
                    }
                    let pos = solved(&f.meta.path, f.meta.file_size, f.meta.mtime_ns).or(f.meta.ra.zip(f.meta.dec));
                    if let Some(p) = pos {
                        a.points.push(p);
                        a.unit_points.entry(ui).or_default().push(p);
                        a.object_points.entry(obj.trim().to_string()).or_default().push(p);
                    }
                }
                _ => {}
            }
        }
    }
    // A unit's target counts may name a spelling that lost the vote: use
    // the target's display name everywhere.
    let mut object_centres: Vec<(usize, Vec<(String, (f64, f64))>)> = Vec::new();
    for (_, a) in acc {
        let name = a.names.iter().max_by_key(|(n, c)| (**c, std::cmp::Reverse((*n).clone()))).map(|x| x.0.clone()).unwrap_or_default();
        let mut notes = Vec::new();
        let unit_centres: Vec<(usize, (f64, f64))> = a.unit_points.iter().filter_map(|(u, p)| centre(p).map(|c| (*u, c))).collect();
        let mosaic = a.objects.iter().filter(|o| crate::index::panel_of(o).is_some()).count() > 1;
        if !mosaic {
            for (i, (ua, ca)) in unit_centres.iter().enumerate() {
                for (ub, cb) in &unit_centres[i + 1..] {
                    let d = sep(*ca, *cb);
                    if d > APART_DEG {
                        notes.push(format!(
                            "points {d:.1}° apart in {} and {}: two targets with one name?",
                            out.units[*ua].name, out.units[*ub].name
                        ));
                    }
                }
            }
        }
        let ti = out.targets.len();
        object_centres.push((ti, a.object_points.iter().filter_map(|(o, p)| centre(p).map(|c| (o.clone(), c))).collect()));
        out.targets.push(Target {
            name,
            objects: a.objects.into_iter().collect(),
            units: a.units.into_iter().collect(),
            lights: a.lights,
            secs: a.secs,
            nights: a.nights.into_iter().collect(),
            centre: centre(&a.points),
            notes,
        });
    }
    for u in out.units.iter_mut() {
        let mut fixed = BTreeMap::new();
        for (n, c) in std::mem::take(&mut u.targets) {
            let shown = out.targets.iter().find(|t| t.name.eq_ignore_ascii_case(&n)).map(|t| t.name.clone()).unwrap_or(n);
            *fixed.entry(shown).or_insert(0) += c;
        }
        u.targets = fixed;
    }
    // Two names on one field: the same target, or one inside a mosaic.
    for i in 0..object_centres.len() {
        for j in i + 1..object_centres.len() {
            let (ti, oi) = (&object_centres[i].0, &object_centres[i].1);
            let (tj, oj) = (&object_centres[j].0, &object_centres[j].1);
            let best = oi.iter().flat_map(|a| oj.iter().map(move |b| (sep(a.1, b.1), a.0.clone(), b.0.clone()))).min_by(|x, y| x.0.total_cmp(&y.0));
            let Some((d, a, b)) = best.filter(|b| b.0 < SAME_FIELD_DEG * 2.0) else { continue };
            let (ni, nj) = (out.targets[*ti].name.clone(), out.targets[*tj].name.clone());
            let in_mosaic = |t: &Target| t.panels() > 1;
            let (mi, mj) = (in_mosaic(&out.targets[*ti]), in_mosaic(&out.targets[*tj]));
            if mi || mj {
                if d < SAME_FIELD_DEG * 2.0 {
                    let (inner, outer, panel) = if mi { (*tj, ni.clone(), a) } else { (*ti, nj.clone(), b) };
                    out.targets[inner].notes.push(format!("inside the {outer} mosaic ({panel} is {d:.2}° away)"));
                }
            } else if d < SAME_FIELD_DEG {
                out.targets[*ti].notes.push(format!("same field as {nj} ({d:.2}° apart): one target with two names?"));
                out.targets[*tj].notes.push(format!("same field as {ni} ({d:.2}° apart): one target with two names?"));
            }
        }
    }
    out.targets.sort_by(|a, b| b.nights.last().cmp(&a.nights.last()).then(a.name.cmp(&b.name)));
    if !out.loose.is_empty() {
        out.notes.push(format!("{} image files sit directly in a library folder, not in a folder of their own: they aren't counted", out.loose.len()));
    }
    out
}

/// A folder name for a target ("NGC 2070"), safe on every platform.
pub fn folder_name(target: &str) -> String {
    let s = crate::process::job::safe_name(target);
    if s.is_empty() {
        "Target".into()
    } else {
        s
    }
}

impl LibraryTargets {
    /// Targets found in more than one unit, with what combining them would do.
    pub fn combinable(&self) -> Vec<(usize, Combine)> {
        (0..self.targets.len()).filter_map(|i| self.combine(i).map(|c| (i, c))).collect()
    }

    /// What combining target `ti` would do, if it's in several units of one
    /// library folder (the one holding most of them).
    pub fn combine(&self, ti: usize) -> Option<Combine> {
        let t = self.targets.get(ti)?;
        if t.units.len() < 2 {
            return None;
        }
        let mut by_root: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for &u in &t.units {
            by_root.entry(crate::paths::key(&self.units[u].root)).or_default().push(u);
        }
        let units = by_root.into_values().max_by_key(|v| v.len())?;
        if units.len() < 2 {
            return None;
        }
        let root = self.units[units[0]].root.clone();
        let dest = root.join(folder_name(&t.name));
        let mut c = Combine { root, dest: dest.clone(), ..Default::default() };
        for u in units {
            let unit = &self.units[u];
            if crate::paths::key(&unit.path) == crate::paths::key(&dest) {
                c.existing = Some(u);
                continue;
            }
            let others: Vec<&String> = unit.targets.keys().filter(|n| !n.eq_ignore_ascii_case(&t.name)).collect();
            if !others.is_empty() {
                c.blocked.push((u, format!("also holds {}", others.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))));
                continue;
            }
            if dest.join(&unit.name).exists() {
                c.blocked.push((u, format!("{} already exists", dest.join(&unit.name).display())));
                continue;
            }
            c.moving.push(u);
        }
        (c.moving.len() + c.existing.is_some() as usize >= 2).then_some(c)
    }

    /// The move plan for combining target `ti`: every file of each moving
    /// unit (not just images) to `<dest>\<unit>\…`. `cache_root`: the CLI's
    /// `--cache-dir` (each folder's cache a subfolder of it), else crux's
    /// usual per-folder caches.
    pub fn combine_plan(&self, ti: usize, cfg: &Config, cache_root: Option<&Path>) -> Result<Plan> {
        let Some(c) = self.combine(ti) else { bail!("{} isn't in several folders of one library folder", self.targets.get(ti).map(|t| t.name.as_str()).unwrap_or("?")) };
        let t = &self.targets[ti];
        let mut moves = Vec::new();
        for &u in &c.moving {
            let unit = &self.units[u];
            for e in walkdir::WalkDir::new(&unit.path).follow_links(false).into_iter() {
                let e = e?;
                if !e.file_type().is_file() {
                    continue;
                }
                let md = e.metadata()?;
                let rel = e.path().strip_prefix(&unit.path)?;
                let mtime_ns = md.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos() as i64).unwrap_or(0);
                moves.push(Move {
                    from: e.path().to_path_buf(),
                    to: c.dest.join(&unit.name).join(rel),
                    size: md.len(),
                    mtime_ns,
                    kind: MoveKind::Organise,
                    reason: format!("combine {} into {}", t.name, c.dest.display()),
                });
            }
        }
        let mut notes: Vec<String> = c.blocked.iter().map(|(u, why)| format!("{} stays where it is: {why}", self.units[*u].name)).collect();
        notes.push("Processed outputs in the workspace stay under the old folder names; processing the combined folder makes new ones.".into());
        let conflicts = crate::plan::find_conflicts(&moves);
        Ok(Plan {
            version: PLAN_VERSION,
            created: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            root: c.root.clone(),
            // The combined folder's cache: the library index and decisions
            // follow the files through it (and back on undo).
            cache_dir: cache_of(&c.dest, cache_root),
            library_root: Some(cfg.library.root.clone()),
            library_dir: Some(cfg.library.dir.clone()),
            moves,
            combine: Vec::new(),
            conflicts,
            notes,
            target: Some(t.name.clone()),
        })
    }
}

/// Targets in the library folders, from their headers (a fraction of a
/// second for thousands of files). Plate solutions in `index` place the
/// lights more exactly than the mount's RA/DEC. A drive root or the home
/// folder is refused: choose the folders that hold imaging data.
pub fn scan(cfg: &Config, roots: &[PathBuf], index: Option<&crate::index::Index>) -> LibraryTargets {
    let mut refused = Vec::new();
    let triages: Vec<(PathBuf, Triage)> = roots
        .iter()
        .filter(|r| {
            let bad = crate::paths::too_broad(r);
            if bad {
                refused.push(format!("{} is a whole drive or your home folder: choose the folders inside it that hold your imaging data", r.display()));
            }
            !bad && r.is_dir()
        })
        .map(|r| {
            let mut c = crate::cache::Cache::disabled();
            (r.clone(), crate::pipeline::run(r, cfg, &mut c, false, &|_, _, _| {}))
        })
        .collect();
    let scans: Vec<(PathBuf, &Triage)> = triages.iter().map(|(r, t)| (r.clone(), t)).collect();
    let solved = |p: &Path, size: u64, mtime: i64| index.and_then(|ix| ix.solution(p, size, mtime)).flatten().map(|s| (s.ra, s.dec));
    let mut lt = find(&scans, &solved);
    for r in roots.iter().filter(|r| !r.is_dir()) {
        refused.push(format!("{} isn't there (unplugged, or not on the network?)", r.display()));
    }
    lt.notes.splice(0..0, refused);
    lt
}

/// Does `dir` hold several targets in folders of their own (a library
/// folder, not one target)? `t`: its header triage.
pub fn looks_like_library(dir: &Path, t: &Triage) -> bool {
    let solved = |_: &Path, _: u64, _: i64| None;
    let lt = find(&[(dir.to_path_buf(), t)], &solved);
    lt.targets.len() >= 2 && lt.units.iter().filter(|u| u.lights() > 0).count() >= 2
}

/// A folder's cache: `<cache_root>\<folder name>` (as `autopilot::cache_for`)
/// or crux's managed one.
fn cache_of(dir: &Path, cache_root: Option<&Path>) -> Option<PathBuf> {
    match cache_root {
        Some(r) => Some(r.join(crate::process::job::safe_name(&dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "folder".into())))),
        None => crate::cachedir::for_folder(dir),
    }
}

/// After a combine was applied: bring each unit's cache (measurements,
/// maps, thumbnails) and decisions into the combined folder's, under the
/// new paths, so nothing is measured again and no pin is lost. The units'
/// own caches stay (an undo finds them again); `auto_clean` empties them
/// once unused.
pub fn adopt_caches(plan: &Plan) -> Vec<String> {
    let mut notes = Vec::new();
    let Some(dest_cache) = &plan.cache_dir else { return notes };
    // Units: the folder directly under the plan's root that each move left.
    let mut by_unit: BTreeMap<PathBuf, Vec<(PathBuf, PathBuf)>> = BTreeMap::new();
    for m in &plan.moves {
        let Ok(rel) = m.from.strip_prefix(&plan.root) else { continue };
        let Some(first) = rel.components().next() else { continue };
        by_unit.entry(plan.root.join(first.as_os_str())).or_default().push((m.from.clone(), m.to.clone()));
    }
    // The units' caches sit where the combined folder's does: beside it
    // under an explicit cache root, else crux's managed caches.
    let managed = crate::cachedir::for_folder(&plan.root.join("x")).and_then(|p| p.parent().map(Path::to_path_buf));
    let cache_root = dest_cache.parent().filter(|p| managed.as_deref().is_none_or(|m| crate::paths::key(m) != crate::paths::key(p)));
    let mut dst = crate::cache::Cache::load(dest_cache);
    let mut dst_dec = crate::decisions::Decisions::load(dest_cache).unwrap_or_default();
    let mut dec_changed = false;
    for (unit, moves) in &by_unit {
        let Some(src_dir) = cache_of(unit, cache_root).filter(|d| d.join("cache.json").is_file() || d.join("decisions.json").is_file()) else { continue };
        let src = crate::cache::Cache::load(&src_dir);
        let n = dst.adopt(&src, moves);
        if let Ok(mut d) = crate::decisions::Decisions::load(&src_dir) {
            for (from, to) in moves {
                d.rename(from, to);
            }
            dec_changed |= dst_dec.absorb(&d);
        }
        if n > 0 {
            notes.push(format!("{n} measured files from {} kept", unit.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default()));
        }
    }
    if let Err(e) = dst.save() {
        notes.push(format!("couldn't save the combined folder's cache: {e:#}"));
    }
    if dec_changed {
        if dst_dec.file.is_none() {
            dst_dec.file = Some(dest_cache.join("decisions.json"));
        }
        if let Err(e) = dst_dec.save() {
            notes.push(format!("couldn't save your decisions for the combined folder: {e:#}"));
        }
    }
    notes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_names_merge_spellings_not_panels() {
        let objs = ["NGC 3372", "NGC 3372", "NGC3372", "LMC Panel 1", "LMC Panel 2", "LMC Panel 2"];
        let f = field_names(objs.iter().copied());
        assert_eq!(f["NGC3372"], "NGC 3372");
        assert_eq!(f["NGC 3372"], "NGC 3372");
        assert_eq!(f["LMC Panel 1"], "LMC Panel 1");
        assert_eq!(f["LMC Panel 2"], "LMC Panel 2");
        let one = field_names(["M42", "M42"].iter().copied());
        assert_eq!(one["M42"], "M42", "a single spelling keeps its name");
    }

    #[test]
    fn catalogue_names() {
        assert_eq!(normalise("M42"), "M 42");
        assert_eq!(normalise("m 42"), "M 42");
        assert_eq!(normalise("M_42"), "M 42");
        assert_eq!(normalise("Messier 42"), "M 42");
        assert_eq!(normalise("NGC2070"), "NGC 2070");
        assert_eq!(normalise("ngc 2070"), "NGC 2070");
        assert_eq!(normalise("NGC 6188b"), "NGC 6188b");
        assert_eq!(normalise("IC434"), "IC 434");
        assert_eq!(normalise("IC 2944"), "IC 2944");
        assert_eq!(normalise("sh2-240"), "Sh2-240");
        assert_eq!(normalise("Sh2 240"), "Sh2-240");
        assert_eq!(normalise("LMC Panel 26"), "LMC");
        assert_eq!(normalise("M 42 Panel 1"), "M 42");
        assert_eq!(normalise("  Rosette   Nebula "), "Rosette Nebula");
        assert_eq!(normalise("Mars"), "Mars");
        assert_eq!(normalise("Moon"), "Moon");
        assert_eq!(normalise("Cocoon"), "Cocoon");
        assert_eq!(normalise("M 1a2"), "M 1a2");
    }

    #[test]
    fn filter_names() {
        for (a, b) in [("Ha", "H"), ("H-alpha", "H"), ("OIII", "O"), ("SII", "S"), ("Luminance", "L"), ("Lum", "L"), ("Red", "R"), ("b", "B"), ("", "?"), ("Lext", "LEXT")] {
            assert_eq!(filter_name(a), b, "{a}");
        }
    }

    #[test]
    fn centre_wraps_ra() {
        let c = centre(&[(359.9, 10.0), (0.1, 10.0)]).unwrap();
        assert!(c.0 < 0.01 || c.0 > 359.99, "{c:?}");
        assert!((c.1 - 10.0).abs() < 1e-3);
    }
}
