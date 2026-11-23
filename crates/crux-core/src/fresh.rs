//! Start again from a fresh scan (GUI: Settings → Start fresh; CLI:
//! `crux fresh`).
//!
//! Clears, in one go, what crux has remembered and made, so the next scan
//! works everything out again:
//! - every folder's measurements, maps and thumbnails (rebuilt by the next
//!   scan);
//! - the caches of folders that no longer exist;
//! - the library index and flat library (moved aside as a backup);
//! - processing outputs in the workspace: only the intermediate files
//!   (registered and calibrated frames, network and preview caches), or
//!   everything crux made there (masters, mosaics, images and recipes too);
//! - if asked, your decisions (kept and rejected frames, flat choices, panel
//!   corrections, layout rules and saved views).
//!
//! Your data is never touched. In a workspace only the folders crux made
//! are deleted: those holding `crux-process.json`, `crux-integration.json`,
//! `crux-mosaic.json`, `crux-plan.json`, or an `image` folder with a recipe
//! or a prepared cache. Training bundles (`train`, or any folder with a
//! bundle's `manifest.json`) and everything else stay. A workspace that is
//! a drive or the home folder, or that is inside a library folder or holds
//! one, is refused.

use crate::cachedir::{self, dir_size, CacheFolder};
use std::path::{Path, PathBuf};

/// What to do with the workspace's processing outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Work {
    /// Leave them.
    Keep,
    /// Registered and calibrated frames, network and preview caches:
    /// processing again rebuilds them; masters and images stay.
    #[default]
    Intermediates,
    /// Every folder crux made in the workspace, images and recipes included.
    All,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    /// Every folder's measurements, maps and thumbnails.
    pub measurements: bool,
    /// Caches of folders that no longer exist (their decisions too).
    pub missing: bool,
    /// The library index and flat library (moved aside as a backup).
    pub index: bool,
    pub work: Work,
    /// Your decisions in every cache.
    pub decisions: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { measurements: true, missing: true, index: true, work: Work::Intermediates, decisions: false }
    }
}

impl Options {
    pub fn anything(&self) -> bool {
        self.measurements || self.missing || self.index || self.work != Work::Keep || self.decisions
    }
}

/// One folder of processing outputs in a workspace.
#[derive(Clone, Debug)]
pub struct WorkFolder {
    pub dir: PathBuf,
    /// Its rebuildable parts.
    pub intermediates: Vec<PathBuf>,
    pub intermediate_bytes: u64,
    pub bytes: u64,
}

/// What's there to clear, with sizes.
#[derive(Clone, Debug, Default)]
pub struct Survey {
    /// Managed caches (`%LOCALAPPDATA%\crux\folders`).
    pub caches: Vec<CacheFolder>,
    /// `--cache-dir`: one cache shared by every folder.
    pub explicit: Option<PathBuf>,
    pub explicit_bytes: (u64, u64),
    /// The library index (`index.json` path), when there is one.
    pub index: Option<PathBuf>,
    pub index_bytes: u64,
    /// Workspaces looked through.
    pub workspaces: Vec<PathBuf>,
    pub work: Vec<WorkFolder>,
    /// Files crux wrote at a workspace's top (the autopilot report).
    pub loose: Vec<PathBuf>,
    /// Workspaces left alone, and why.
    pub refused: Vec<String>,
}

impl Survey {
    pub fn measurement_bytes(&self) -> u64 {
        self.caches.iter().filter(|c| c.folder_exists).map(|c| c.rebuildable).sum::<u64>() + self.explicit_bytes.0
    }
    pub fn missing(&self) -> impl Iterator<Item = &CacheFolder> {
        self.caches.iter().filter(|c| !c.folder_exists)
    }
    pub fn missing_bytes(&self) -> u64 {
        self.missing().map(|c| c.rebuildable + c.kept).sum()
    }
    /// Decisions, plans and journals of folders that still exist.
    pub fn decision_bytes(&self) -> u64 {
        self.caches.iter().filter(|c| c.folder_exists).map(|c| c.kept).sum::<u64>() + self.explicit_bytes.1
    }
    pub fn work_bytes(&self, w: Work) -> u64 {
        match w {
            Work::Keep => 0,
            Work::Intermediates => self.work.iter().map(|f| f.intermediate_bytes).sum(),
            Work::All => self.work.iter().map(|f| f.bytes).sum::<u64>() + self.loose.iter().map(|p| dir_size(p)).sum::<u64>(),
        }
    }
    /// Bytes `o` would free (the index is moved aside, not freed).
    pub fn bytes(&self, o: &Options) -> u64 {
        let mut n = self.work_bytes(o.work);
        if o.measurements || o.decisions {
            n += self.measurement_bytes();
        }
        if o.missing || o.decisions {
            n += self.missing_bytes();
        }
        if o.decisions {
            n += self.decision_bytes();
        }
        n
    }
}

/// Look at what there is, measuring sizes. `index` is the library index's
/// `index.json` path, `explicit_cache` a `--cache-dir`, `library_roots` the
/// library folders (no workspace may overlap them).
pub fn survey(index: Option<&Path>, explicit_cache: Option<&Path>, workspaces: &[PathBuf], library_roots: &[PathBuf]) -> Survey {
    let mut s = Survey { caches: if explicit_cache.is_some() { Vec::new() } else { cachedir::list() }, ..Default::default() };
    if let Some(e) = explicit_cache.filter(|e| e.is_dir()) {
        let rb: u64 = cachedir::REBUILDABLE.iter().map(|n| dir_size(&e.join(n))).sum();
        s.explicit_bytes = (rb, dir_size(&e.join(crate::decisions::FILE)));
        s.explicit = Some(e.to_path_buf());
    }
    if let Some(ix) = index.filter(|ix| crate::index::Index::exists(ix)) {
        let db = crate::index::Index::db_path(ix);
        s.index_bytes = [db.clone(), PathBuf::from(format!("{}-wal", db.display())), ix.to_path_buf()].iter().map(|p| dir_size(p)).sum();
        s.index = Some(ix.to_path_buf());
    }
    let mut seen = std::collections::HashSet::new();
    for ws in workspaces {
        if !seen.insert(crate::paths::key(ws)) || !ws.is_dir() {
            continue;
        }
        if let Err(why) = workspace_ok(ws, library_roots) {
            s.refused.push(why);
            continue;
        }
        s.workspaces.push(ws.clone());
        let Ok(rd) = std::fs::read_dir(ws) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            let Ok(t) = e.file_type() else { continue };
            if t.is_dir() && is_work_folder(&p) {
                let intermediates = intermediates_of(&p);
                let intermediate_bytes = intermediates.iter().map(|q| dir_size(q)).sum();
                s.work.push(WorkFolder { bytes: dir_size(&p), dir: p, intermediates, intermediate_bytes });
            } else if t.is_file() && e.file_name().to_str().is_some_and(|n| LOOSE.contains(&n)) {
                s.loose.push(p);
            }
        }
    }
    s.work.sort_by(|a, b| a.dir.cmp(&b.dir));
    s
}

/// Files crux writes at the top of a workspace.
const LOOSE: [&str; 2] = ["autopilot.json", "autopilot.html"];
/// Files only crux's processing writes into a folder's outputs.
const MARKERS: [&str; 4] = [crate::process::job::MANIFEST, "crux-integration.json", "crux-mosaic.json", "crux-plan.json"];

/// May this workspace be cleared? Not a drive, not the home folder, not
/// inside a library folder or holding one.
pub fn workspace_ok(ws: &Path, library_roots: &[PathBuf]) -> Result<(), String> {
    if crate::paths::too_broad(ws) {
        return Err(format!("{} is a whole drive or your home folder", ws.display()));
    }
    for r in library_roots {
        if crate::library::inside(ws, r) || crate::library::inside(r, ws) {
            return Err(format!("{} overlaps the library folder {}", ws.display(), r.display()));
        }
    }
    Ok(())
}

/// A folder of processing outputs crux made (never a training bundle).
fn is_work_folder(d: &Path) -> bool {
    if d.file_name().is_some_and(|n| n.eq_ignore_ascii_case("train")) || d.join("manifest.json").is_file() {
        return false;
    }
    MARKERS.iter().any(|m| d.join(m).is_file()) || image_dirs(d).iter().any(|t| t.join("recipe.toml").is_file() || t.join("prepared.cache").is_file())
}

/// `<folder>\image\<target>` folders.
fn image_dirs(d: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(d.join("image")).map(|rd| rd.flatten().filter(|e| e.file_type().is_ok_and(|t| t.is_dir())).map(|e| e.path()).collect()).unwrap_or_default()
}

/// What processing again rebuilds: registered and calibrated frames, and
/// each target's network results and prepared preview.
fn intermediates_of(d: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = ["registered", "calibrated"].iter().map(|n| d.join(n)).filter(|p| p.exists()).collect();
    for t in image_dirs(d) {
        v.extend(["enhanced", "starless"].iter().map(|n| t.join(n)).filter(|p| p.exists()));
        v.extend(crate::render::prepcache::slots(&t.join("prepared.cache")).into_iter().filter(|p| p.exists()));
    }
    v
}

/// What a run did.
#[derive(Clone, Debug, Default)]
pub struct Done {
    pub freed: u64,
    /// Where the old index went.
    pub index_backup: Option<PathBuf>,
    pub caches_emptied: usize,
    pub caches_forgotten: usize,
    pub work_folders: usize,
    pub errors: Vec<String>,
}

impl Done {
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.caches_emptied > 0 {
            parts.push(format!("{} folder cache{} emptied", self.caches_emptied, if self.caches_emptied == 1 { "" } else { "s" }));
        }
        if self.caches_forgotten > 0 {
            parts.push(format!("{} removed", self.caches_forgotten));
        }
        if self.work_folders > 0 {
            parts.push(format!("work files cleared in {} folder{}", self.work_folders, if self.work_folders == 1 { "" } else { "s" }));
        }
        if self.index_backup.is_some() {
            parts.push("index started again".into());
        }
        let mut s = if parts.is_empty() { "Nothing to clear".to_string() } else { parts.join(", ") };
        s.push_str(&format!(" ({} freed)", cachedir::human(self.freed)));
        if !self.errors.is_empty() {
            s.push_str(&format!("; {} couldn't be removed", self.errors.len()));
        }
        s
    }
}

fn remove(p: &Path) -> std::io::Result<()> {
    if std::fs::symlink_metadata(p)?.is_dir() {
        std::fs::remove_dir_all(p)
    } else {
        std::fs::remove_file(p)
    }
}

/// Clear what `o` asks for, from what `s` found. The index goes first:
/// when it can't be moved (crux open in another window), nothing else is
/// touched.
pub fn run(s: &Survey, o: &Options) -> anyhow::Result<Done> {
    let mut d = Done::default();
    if o.index {
        if let Some(ix) = &s.index {
            d.index_backup = crate::index::Index::reset(ix)?;
        }
    }
    for c in &s.caches {
        let forget = o.decisions || (o.missing && !c.folder_exists);
        let res = if forget {
            cachedir::forget(&c.dir).map(|n| {
                d.caches_forgotten += 1;
                n
            })
        } else if o.measurements && c.rebuildable > 0 {
            cachedir::empty(&c.dir).map(|n| {
                d.caches_emptied += 1;
                n
            })
        } else {
            Ok(0)
        };
        match res {
            Ok(n) => d.freed += n,
            Err(e) => d.errors.push(format!("{e:#}")),
        }
    }
    if let Some(e) = &s.explicit {
        if (o.measurements || o.decisions) && s.explicit_bytes.0 > 0 {
            match cachedir::empty(e) {
                Ok(n) => {
                    d.freed += n;
                    d.caches_emptied += 1;
                }
                Err(err) => d.errors.push(format!("{err:#}")),
            }
        }
        if o.decisions {
            let f = e.join(crate::decisions::FILE);
            if f.is_file() {
                let n = dir_size(&f);
                match std::fs::remove_file(&f) {
                    Ok(()) => d.freed += n,
                    Err(err) => d.errors.push(format!("remove {}: {err}", f.display())),
                }
            }
        }
    }
    let targets: Vec<(Vec<PathBuf>, bool)> = match o.work {
        Work::Keep => Vec::new(),
        Work::Intermediates => s.work.iter().map(|f| (f.intermediates.clone(), true)).collect(),
        Work::All => s.work.iter().map(|f| (vec![f.dir.clone()], true)).chain(std::iter::once((s.loose.clone(), false))).collect(),
    };
    for (paths, counts) in targets {
        let mut any = false;
        for p in paths {
            let n = dir_size(&p);
            match remove(&p) {
                Ok(()) => {
                    d.freed += n;
                    any = true;
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => d.errors.push(format!("remove {}: {e}", p.display())),
            }
        }
        if any && counts {
            d.work_folders += 1;
        }
    }
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(p: &Path, bytes: usize) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, vec![0u8; bytes]).unwrap();
    }

    /// Only crux's own output folders are found; intermediates are the
    /// rebuildable parts; bundles, other folders and the library are left.
    #[test]
    fn clears_only_what_crux_made() {
        let root = std::env::temp_dir().join(format!("crux-fresh-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let ws = root.join("work");
        let a = ws.join("2026-10-05_M 17");
        touch(&a.join(crate::process::job::MANIFEST), 10);
        touch(&a.join("registered/M 17/H/x_r.xisf"), 1000);
        touch(&a.join("integrated/M 17/masterLight_M 17_H.xisf"), 500);
        touch(&a.join("image/M 17/M 17 Natural.jpg"), 50);
        touch(&a.join("image/M 17/recipe.toml"), 5);
        touch(&a.join("image/M 17/enhanced/H_x.xisf"), 300);
        touch(&a.join("image/M 17/prepared.cache"), 40);
        // An image-only target (renders from masters elsewhere).
        let b = ws.join("bench target");
        touch(&b.join("image/NGC 1/recipe.toml"), 5);
        touch(&b.join("image/NGC 1/NGC 1 Natural.xisf"), 100);
        // Not crux's: bundles, a folder of the user's, a loose file.
        touch(&ws.join("train/M 17/manifest.json"), 5);
        touch(&ws.join("train/M 17/pairs_0.fits"), 200);
        touch(&ws.join("notes/readme.txt"), 5);
        touch(&ws.join("autopilot.html"), 7);
        touch(&ws.join("mine.txt"), 3);
        let lib = root.join("data");
        touch(&lib.join("2026-10-05_M 17/LIGHT/x.fits"), 100);

        let s = survey(None, None, std::slice::from_ref(&ws), std::slice::from_ref(&lib));
        assert!(s.refused.is_empty(), "{:?}", s.refused);
        let names: Vec<String> = s.work.iter().map(|f| f.dir.file_name().unwrap().to_string_lossy().to_string()).collect();
        assert_eq!(names, ["2026-10-05_M 17", "bench target"]);
        assert_eq!(s.work[0].intermediate_bytes, 1000 + 300 + 40);
        assert_eq!(s.work[1].intermediate_bytes, 0);
        assert_eq!(s.loose.len(), 1);

        let o = Options { measurements: false, missing: false, index: false, work: Work::Intermediates, decisions: false };
        let d = run(&s, &o).unwrap();
        assert_eq!(d.freed, 1340);
        assert!(!a.join("registered").exists() && !a.join("image/M 17/enhanced").exists() && !a.join("image/M 17/prepared.cache").exists());
        assert!(a.join("integrated/M 17/masterLight_M 17_H.xisf").is_file() && a.join("image/M 17/recipe.toml").is_file());

        let s = survey(None, None, std::slice::from_ref(&ws), std::slice::from_ref(&lib));
        let d = run(&s, &Options { work: Work::All, ..o }).unwrap();
        assert_eq!(d.work_folders, 2);
        assert!(!a.exists() && !b.exists() && !ws.join("autopilot.html").exists());
        assert!(ws.join("train/M 17/pairs_0.fits").is_file() && ws.join("notes/readme.txt").is_file() && ws.join("mine.txt").is_file());
        assert!(lib.join("2026-10-05_M 17/LIGHT/x.fits").is_file());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A workspace inside or around a library folder is refused.
    #[test]
    fn refuses_overlapping_workspaces() {
        let lib = std::env::temp_dir().join("crux-fresh-lib");
        assert!(workspace_ok(&lib.join("work"), std::slice::from_ref(&lib)).is_err());
        assert!(workspace_ok(lib.parent().unwrap(), std::slice::from_ref(&lib)).is_err());
        assert!(workspace_ok(&std::env::temp_dir().join("crux-fresh-other"), std::slice::from_ref(&lib)).is_ok());
        assert!(workspace_ok(Path::new(if cfg!(windows) { "C:\\" } else { "/" }), &[]).is_err());
    }
}
