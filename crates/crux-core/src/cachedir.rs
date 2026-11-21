//! Where each folder's cache lives, and cleaning caches up.
//!
//! A folder's cache (headers, measurements, maps, thumbnails, hot-pixel
//! lists, your decisions, move plans and journals) lives in the per-user
//! data folder, one subfolder per data folder:
//! `%LOCALAPPDATA%\crux\folders\<folder name>-<hash of its path>`, beside
//! the library index (`%LOCALAPPDATA%\crux\index.db`). Nothing is written
//! into the data. `--cache-dir` still names one exact folder, and
//! `[cache] in_folder = true` keeps the old `<folder>\.crux`.
//!
//! A folder scanned before this change has its `.crux` copied across the
//! first time (the old one is left alone and listed, so you can remove it).
//!
//! Cleaning only removes what crux can rebuild (`cache.json`, `maps`,
//! `thumbs`, `spikes`); your decisions, plans and journals stay unless the
//! whole cache is forgotten.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The file in each managed cache folder that says whose it is.
pub const OWNER_FILE: &str = "folder.json";
/// Rebuildable contents of a cache folder.
pub const REBUILDABLE: [&str; 4] = ["cache.json", "maps", "thumbs", "spikes"];
/// The old in-folder cache's name.
pub const LEGACY: &str = ".crux";

/// The folder holding every managed cache.
pub fn root() -> Option<PathBuf> {
    crate::paths::data_dir().map(|d| d.join("folders"))
}

/// The managed cache folder for a data folder (not created).
pub fn for_folder(dir: &Path) -> Option<PathBuf> {
    let abs = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
    let key = crate::paths::key(&abs);
    let mut h: u64 = 0xcbf29ce484222325;
    for b in key.trim_end_matches(['\\', '/']).bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let name: String = abs
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "drive".into())
        .chars()
        .map(|c| if c.is_alphanumeric() || " -_.".contains(c) { c } else { '_' })
        .take(48)
        .collect();
    let name = name.trim().trim_end_matches('.');
    Some(root()?.join(format!("{name}-{:08x}", h as u32)))
}

/// Who a managed cache belongs to.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Owner {
    pub folder: PathBuf,
    /// Seconds since 1970 when crux last opened the folder.
    pub last_used: u64,
    /// The old `<folder>\.crux`, when one was copied across.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy: Option<PathBuf>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn read_owner(cache: &Path) -> Option<Owner> {
    serde_json::from_slice(&std::fs::read(cache.join(OWNER_FILE)).ok()?).ok()
}

fn write_owner(cache: &Path, o: &Owner) -> Result<()> {
    std::fs::create_dir_all(cache).with_context(|| format!("create {}", cache.display()))?;
    std::fs::write(cache.join(OWNER_FILE), serde_json::to_vec_pretty(o)?).with_context(|| format!("write {}", cache.join(OWNER_FILE).display()))
}

/// The cache folder to use for `dir`: `explicit` (`--cache-dir`) if given,
/// else `<dir>\.crux` when `in_folder`, else the managed folder (marked as
/// used now, adopting an old `.crux` the first time). Falls back to
/// `<dir>\.crux` only when there's no per-user data folder.
pub fn resolve(explicit: Option<&Path>, dir: &Path, in_folder: bool) -> PathBuf {
    if let Some(e) = explicit {
        return e.to_path_buf();
    }
    if in_folder {
        return dir.join(LEGACY);
    }
    let Some(cache) = for_folder(dir) else { return dir.join(LEGACY) };
    let mut owner = read_owner(&cache).unwrap_or_default();
    let abs = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
    owner.folder = abs.clone();
    owner.last_used = now();
    let legacy = abs.join(LEGACY);
    if legacy.join("cache.json").is_file() {
        if !cache.join("cache.json").exists() {
            // First time: bring the old cache across (copy; the old one stays).
            let _ = copy_tree(&legacy, &cache);
        }
        owner.legacy = Some(legacy);
    } else {
        owner.legacy = None;
    }
    let _ = write_owner(&cache, &owner);
    cache
}

fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)?.flatten() {
        let (src, dst) = (e.path(), to.join(e.file_name()));
        if e.file_type()?.is_dir() {
            copy_tree(&src, &dst)?;
        } else if !dst.exists() {
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

/// The library index (`index.json` path; SQLite beside it) that goes with
/// a cache folder: the shared one for managed caches, else the one inside
/// it (`--cache-dir` keeps everything together, as before).
pub fn index_for(cache: &Path) -> PathBuf {
    match (root(), crate::paths::data_dir()) {
        (Some(r), Some(d)) if cache.parent().is_some_and(|p| crate::paths::key(p) == crate::paths::key(&r)) => d.join("index.json"),
        _ => cache.join("index.json"),
    }
}

/// The library index without a folder: inside `--cache-dir` if given,
/// else in the per-user data folder.
pub fn default_index(explicit: Option<&Path>) -> Option<PathBuf> {
    match explicit {
        Some(e) => Some(e.join("index.json")),
        None => crate::paths::data_dir().map(|d| d.join("index.json")),
    }
}

/// One managed cache, for the cleanup list.
#[derive(Debug, Clone)]
pub struct CacheFolder {
    pub dir: PathBuf,
    /// The data folder it belongs to (None if `folder.json` is missing).
    pub folder: Option<PathBuf>,
    pub folder_exists: bool,
    pub last_used: Option<SystemTime>,
    /// Bytes crux can rebuild (measurements, maps, thumbnails, hot pixels).
    pub rebuildable: u64,
    /// Bytes it keeps when emptied (decisions, plans, journals).
    pub kept: u64,
    /// An old `<folder>\.crux` still in the data folder, and its size.
    pub legacy: Option<(PathBuf, u64)>,
}

pub fn dir_size(p: &Path) -> u64 {
    match std::fs::symlink_metadata(p) {
        Ok(m) if m.is_dir() => std::fs::read_dir(p).map(|rd| rd.flatten().map(|e| dir_size(&e.path())).sum()).unwrap_or(0),
        Ok(m) => m.len(),
        Err(_) => 0,
    }
}

fn describe(dir: &Path) -> CacheFolder {
    let owner = read_owner(dir);
    let rebuildable = REBUILDABLE.iter().map(|n| dir_size(&dir.join(n))).sum::<u64>();
    let kept = dir_size(dir).saturating_sub(rebuildable);
    let folder = owner.as_ref().map(|o| o.folder.clone());
    CacheFolder {
        dir: dir.to_path_buf(),
        folder_exists: folder.as_ref().is_some_and(|f| f.is_dir()),
        folder,
        last_used: owner.as_ref().map(|o| UNIX_EPOCH + Duration::from_secs(o.last_used)),
        rebuildable,
        kept,
        legacy: owner.and_then(|o| o.legacy).filter(|l| l.join("cache.json").is_file()).map(|l| {
            let n = dir_size(&l);
            (l, n)
        }),
    }
}

/// Every managed cache, biggest first.
pub fn list() -> Vec<CacheFolder> {
    let Some(r) = root() else { return Vec::new() };
    let Ok(rd) = std::fs::read_dir(&r) else { return Vec::new() };
    let mut v: Vec<CacheFolder> = rd.flatten().filter(|e| e.file_type().is_ok_and(|t| t.is_dir())).map(|e| describe(&e.path())).collect();
    v.sort_by(|a, b| b.rebuildable.cmp(&a.rebuildable));
    v
}

/// The cache folder described, for one data folder.
pub fn describe_for(dir: &Path) -> Option<CacheFolder> {
    let c = for_folder(dir)?;
    c.is_dir().then(|| describe(&c))
}

fn remove(p: &Path) -> Result<()> {
    let m = std::fs::symlink_metadata(p)?;
    if m.is_dir() {
        std::fs::remove_dir_all(p).with_context(|| format!("remove {}", p.display()))
    } else {
        std::fs::remove_file(p).with_context(|| format!("remove {}", p.display()))
    }
}

/// Is `dir` a cache folder crux may clean? A managed one, a `.crux`, or one
/// holding `cache.json` (an explicit `--cache-dir`).
fn cleanable(dir: &Path) -> bool {
    dir.file_name().is_some_and(|n| n == LEGACY)
        || root().is_some_and(|r| dir.parent().is_some_and(|p| crate::paths::key(p) == crate::paths::key(&r)))
        || dir.join("cache.json").is_file()
}

/// Remove what crux can rebuild from a cache folder; returns bytes freed.
/// Decisions, plans and journals stay. The next scan re-measures.
pub fn empty(dir: &Path) -> Result<u64> {
    anyhow::ensure!(cleanable(dir), "{} isn't a crux cache folder", dir.display());
    let mut freed = 0;
    for n in REBUILDABLE.iter().copied().chain(["cache.json.tmp"]) {
        let p = dir.join(n);
        if p.exists() {
            let size = dir_size(&p);
            remove(&p)?;
            freed += size;
        }
    }
    Ok(freed)
}

/// Remove a managed cache completely, decisions and journals included
/// (for a folder that no longer exists). Returns bytes freed.
pub fn forget(dir: &Path) -> Result<u64> {
    let managed = root().is_some_and(|r| dir.parent().is_some_and(|p| crate::paths::key(p) == crate::paths::key(&r)));
    anyhow::ensure!(managed, "{} isn't a managed cache folder", dir.display());
    let size = dir_size(dir);
    remove(dir)?;
    Ok(size)
}

/// Remove an old `<folder>\.crux` cache from a data folder (it was copied
/// across). Only a folder named `.crux` holding `cache.json`.
pub fn remove_legacy(legacy: &Path) -> Result<u64> {
    anyhow::ensure!(
        legacy.file_name().is_some_and(|n| n == LEGACY) && legacy.join("cache.json").is_file(),
        "{} isn't an old crux cache",
        legacy.display()
    );
    let size = dir_size(legacy);
    remove(legacy)?;
    Ok(size)
}

/// Empty the managed caches of folders not opened for `days` days (or
/// whose folder no longer exists, after the same wait). Runs at most once a
/// day; `days` = 0 turns it off. Returns (caches emptied, bytes freed).
pub fn auto_clean(days: u32) -> (usize, u64) {
    if days == 0 {
        return (0, 0);
    }
    let Some(r) = root() else { return (0, 0) };
    let stamp = r.join(".last_clean");
    let recent = std::fs::metadata(&stamp).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|e| e < Duration::from_secs(86_400));
    if recent || !r.is_dir() {
        return (0, 0);
    }
    let _ = std::fs::write(&stamp, b"");
    let limit = Duration::from_secs(days as u64 * 86_400);
    let (mut n, mut bytes) = (0, 0);
    for c in list() {
        let old = c.last_used.and_then(|t| t.elapsed().ok()).is_some_and(|e| e > limit);
        if old && c.rebuildable > 0 {
            if let Ok(b) = empty(&c.dir) {
                n += 1;
                bytes += b;
            }
        }
    }
    (n, bytes)
}

/// "1.2 GB", "48 MB", "310 KB".
pub fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.0} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} KB", b / 1e3)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_stable_and_distinct() {
        let a = for_folder(Path::new("/data/2026-11-05_LMC_Mosaic")).unwrap();
        let b = for_folder(Path::new("/data/2026-11-05_LMC_Mosaic/")).unwrap();
        let c = for_folder(Path::new("/other/2026-11-05_LMC_Mosaic")).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.file_name().unwrap().to_string_lossy().starts_with("2026-11-05_LMC_Mosaic-"));
        assert_eq!(index_for(&a), crate::paths::data_dir().unwrap().join("index.json"));
        assert_eq!(index_for(Path::new("/scratch/cache")), Path::new("/scratch/cache").join("index.json"));
    }

    #[test]
    fn empty_keeps_decisions() {
        let d = std::env::temp_dir().join(format!("crux-cachedir-{}", std::process::id())).join(LEGACY);
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("maps")).unwrap();
        std::fs::create_dir_all(d.join("plans")).unwrap();
        std::fs::write(d.join("cache.json"), b"{}").unwrap();
        std::fs::write(d.join("maps").join("a.map"), [0u8; 100]).unwrap();
        std::fs::write(d.join("decisions.json"), b"{}").unwrap();
        std::fs::write(d.join("plans").join("p.json"), b"{}").unwrap();
        assert_eq!(empty(&d).unwrap(), 102);
        assert!(!d.join("cache.json").exists() && !d.join("maps").exists());
        assert!(d.join("decisions.json").exists() && d.join("plans").join("p.json").exists());
        // Not a cache folder: refused.
        assert!(empty(d.parent().unwrap()).is_err());
        let _ = std::fs::remove_dir_all(d.parent().unwrap());
    }
}
