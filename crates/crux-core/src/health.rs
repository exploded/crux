//! Library health: where the space goes, what can be rebuilt, duplicates,
//! empty folders. Reads sizes and times only (hashes only to confirm
//! duplicate candidates). Never deletes: reclaiming is a move plan to
//! `_Reclaimable`, which you empty yourself.

use crate::classify::{FrameType, Role};
use crate::config::wildcard_match;
use crate::index::IndexFile;
use crate::plan::{find_conflicts, Move, MoveKind, Plan, PLAN_VERSION};
use crate::scan::Frame;
use anyhow::Result;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// WBPP intermediates that can be regenerated from raw subs and masters.
/// Mosaic and drizzle folders aren't here: they may hold finished work.
pub const REBUILDABLE: [&str; 7] = ["calibrated", "registered", "logs", "*.OLD", "fastIntegration", "cosmetized", "debayered"];

/// Folder name for reclaimed intermediates, beside the library folders.
pub const RECLAIM_DIR: &str = "_Reclaimable";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Part {
    Raw,
    Calibration,
    Masters,
    Outputs,
    Rebuildable,
    Other,
}

impl Part {
    pub const ALL: [Part; 6] = [Part::Raw, Part::Calibration, Part::Masters, Part::Outputs, Part::Rebuildable, Part::Other];
    pub fn name(self) -> &'static str {
        match self {
            Part::Raw => "Raw subs",
            Part::Calibration => "Calibration frames",
            Part::Masters => "Masters",
            Part::Outputs => "Outputs",
            Part::Rebuildable => "Rebuildable",
            Part::Other => "Other",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FolderSpace {
    pub root: PathBuf,
    pub name: String,
    /// Bytes per part.
    pub parts: HashMap<Part, u64>,
    /// Rebuildable folders: (path, bytes, files).
    pub rebuildable: Vec<(PathBuf, u64, usize)>,
    pub files: usize,
}

impl FolderSpace {
    pub fn total(&self) -> u64 {
        self.parts.values().sum()
    }
    pub fn get(&self, p: Part) -> u64 {
        self.parts.get(&p).copied().unwrap_or(0)
    }
}

fn is_rebuildable(name: &str) -> bool {
    REBUILDABLE.iter().any(|p| wildcard_match(&p.to_lowercase(), &name.to_lowercase()))
}

/// Sizes of everything under `root`, by part. `frames` are the folder's
/// scanned frames (headers), which say what each image file is.
pub fn space(root: &Path, frames: &[Frame]) -> FolderSpace {
    let kinds: HashMap<String, (FrameType, Role)> = frames.iter().map(|f| (crate::cache::key(&f.meta.path), (f.class.frame_type, f.class.role))).collect();
    let mut s = FolderSpace { root: root.to_path_buf(), name: root.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), ..Default::default() };
    let mut reb: HashMap<PathBuf, (u64, usize)> = HashMap::new();
    for e in walkdir::WalkDir::new(root).follow_links(false).into_iter().flatten() {
        if !e.file_type().is_file() {
            continue;
        }
        let p = e.path();
        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
        s.files += 1;
        let rel = p.strip_prefix(root).unwrap_or(p);
        // The outermost rebuildable folder this file sits in.
        let mut acc = root.to_path_buf();
        let mut rdir = None;
        for c in rel.parent().map(|x| x.components().collect::<Vec<_>>()).unwrap_or_default() {
            acc.push(c);
            if c.as_os_str() == ".crux" {
                rdir = None;
                break;
            }
            if is_rebuildable(&c.as_os_str().to_string_lossy()) {
                rdir = Some(acc.clone());
                break;
            }
        }
        if rel.components().any(|c| c.as_os_str() == ".crux") {
            *s.parts.entry(Part::Other).or_default() += size;
            continue;
        }
        let part = if let Some(d) = rdir {
            let x = reb.entry(d).or_default();
            x.0 += size;
            x.1 += 1;
            Part::Rebuildable
        } else {
            match kinds.get(&crate::cache::key(p)) {
                Some((_, Role::Master)) => Part::Masters,
                Some((_, Role::Processed)) => Part::Outputs,
                Some((FrameType::Light, _)) => Part::Raw,
                Some((FrameType::Flat | FrameType::Dark | FrameType::DarkFlat | FrameType::Bias, _)) => Part::Calibration,
                Some((FrameType::Unknown, _)) => Part::Other,
                None => {
                    let ext = p.extension().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default();
                    if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "tif" | "tiff" | "xisf" | "fits" | "fit" | "psd") {
                        Part::Outputs
                    } else {
                        Part::Other
                    }
                }
            }
        };
        *s.parts.entry(part).or_default() += size;
    }
    let mut r: Vec<(PathBuf, u64, usize)> = reb.into_iter().map(|(p, (b, n))| (p, b, n)).collect();
    r.sort_by_key(|x| std::cmp::Reverse(x.1));
    s.rebuildable = r;
    s
}

/// A plan moving rebuildable folders to `<parent>\_Reclaimable\<folder>\…`.
/// Nothing is deleted; you empty `_Reclaimable` yourself.
pub fn reclaim_plan(parent: &Path, spaces: &[FolderSpace], cache_dir: Option<&Path>) -> Plan {
    let mut moves = Vec::new();
    let mut notes = Vec::new();
    let dest_root = parent.join(RECLAIM_DIR);
    for s in spaces {
        for (dir, _, _) in &s.rebuildable {
            let rel_dir = dir.strip_prefix(parent).unwrap_or(dir);
            for e in walkdir::WalkDir::new(dir).follow_links(false).into_iter().flatten() {
                if !e.file_type().is_file() {
                    continue;
                }
                let p = e.path();
                let Ok((size, mtime_ns)) = crate::meta::file_stamp(p) else {
                    notes.push(format!("couldn't read {}", p.display()));
                    continue;
                };
                let rel = p.strip_prefix(dir).unwrap_or(p);
                moves.push(Move {
                    from: p.to_path_buf(),
                    to: dest_root.join(rel_dir).join(rel),
                    size,
                    mtime_ns,
                    kind: MoveKind::Reclaim,
                    reason: format!("rebuildable: {}", rel_dir.display()),
                });
            }
        }
    }
    let conflicts = find_conflicts(&moves);
    Plan {
        version: PLAN_VERSION,
        created: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        root: parent.to_path_buf(),
        cache_dir: cache_dir.map(Path::to_path_buf),
        library_root: None,
        library_dir: None,
        moves,
        combine: Vec::new(),
        conflicts,
        notes,
        target: None,
    }
}

/// Files that look the same (same name and size) in more than one place.
pub fn duplicate_candidates(files: &[IndexFile]) -> Vec<Vec<&IndexFile>> {
    let mut by: HashMap<(String, u64), Vec<&IndexFile>> = HashMap::new();
    for f in files {
        let name = f.path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        by.entry((name, f.size)).or_default().push(f);
    }
    let mut out: Vec<Vec<&IndexFile>> = by.into_values().filter(|v| v.len() > 1).collect();
    out.sort_by_key(|v| std::cmp::Reverse(v[0].size * v.len() as u64));
    out
}

pub fn sha256_file(p: &Path) -> Result<String> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 4 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// Confirmed duplicates: candidates whose contents hash the same.
#[derive(Debug, Clone)]
pub struct Duplicate {
    pub size: u64,
    pub sha256: String,
    pub paths: Vec<PathBuf>,
}

pub fn confirm_duplicates(cands: &[Vec<&IndexFile>], progress: &dyn Fn(usize, usize)) -> Vec<Duplicate> {
    let total = cands.iter().map(|c| c.len()).sum();
    let mut done = 0;
    let mut out = Vec::new();
    for c in cands {
        let mut by: HashMap<String, Vec<PathBuf>> = HashMap::new();
        for f in c {
            let h = f.sha256.clone().or_else(|| sha256_file(&f.path).ok());
            done += 1;
            progress(done, total);
            if let Some(h) = h {
                by.entry(h).or_default().push(f.path.clone());
            }
        }
        for (h, paths) in by {
            if paths.len() > 1 {
                out.push(Duplicate { size: c[0].size, sha256: h, paths });
            }
        }
    }
    out
}

/// Folders under `root` with no files anywhere inside.
pub fn empty_dirs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in walkdir::WalkDir::new(root).min_depth(1).follow_links(false).into_iter().flatten() {
        if e.file_type().is_dir() && walkdir::WalkDir::new(e.path()).into_iter().flatten().all(|x| !x.file_type().is_file()) {
            // Only the outermost empty folder.
            if !out.iter().any(|o: &PathBuf| e.path().starts_with(o)) {
                out.push(e.path().to_path_buf());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_and_reclaim() {
        let d = std::env::temp_dir().join(format!("crux-health-{}", std::process::id()));
        let lib = d.join("lib");
        let t = lib.join("2026-11-05_T");
        for (rel, n) in [("LIGHT/a.fits", 100), ("calibrated/a_c.xisf", 300), ("calibrated/sub/b_c.xisf", 50), ("master.OLD/m.xisf", 70), ("out.jpg", 20), ("notes.txt", 5)] {
            let p = t.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, vec![1u8; n]).unwrap();
        }
        std::fs::create_dir_all(t.join("empty/inner")).unwrap();
        let s = space(&t, &[]);
        assert_eq!(s.get(Part::Rebuildable), 420);
        assert_eq!(s.rebuildable.len(), 2);
        assert_eq!(s.get(Part::Outputs), 120); // a.fits (no header info here) + out.jpg
        assert_eq!(s.get(Part::Other), 5);
        let p = reclaim_plan(&lib, &[s], None);
        assert_eq!(p.moves.len(), 3);
        assert!(p.moves.iter().all(|m| m.to.starts_with(lib.join(RECLAIM_DIR).join("2026-11-05_T"))));
        assert!(p.moves.iter().any(|m| m.to.ends_with(r"calibrated\sub\b_c.xisf") || m.to.ends_with("calibrated/sub/b_c.xisf")));
        assert!(p.conflicts.is_empty());
        assert_eq!(empty_dirs(&t), vec![t.join("empty")]);
        std::fs::remove_dir_all(&d).ok();
    }
}
