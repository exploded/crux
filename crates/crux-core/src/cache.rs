//! Sidecar cache keyed by path + size + mtime, so a rescan is near-instant.
//!
//! Stored as JSON at `<cache folder>/cache.json` (see `cachedir` for where
//! that is). Entries for paths outside the scanned root are kept, so one
//! cache file can serve several folders when `--cache-dir` points
//! somewhere shared.

use crate::calib::Map;
use crate::meta::FrameMeta;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Bump when the cached structures change shape or meaning.
pub const CACHE_VERSION: u32 = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub size: u64,
    pub mtime_ns: i64,
    pub meta: FrameMeta,
    /// Pixel analysis results (added by later milestones), as JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis: Option<serde_json::Value>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Cache {
    pub version: u32,
    pub entries: HashMap<String, Entry>,
    #[serde(skip)]
    pub file: Option<PathBuf>,
    #[serde(skip)]
    pub dirty: bool,
    /// Binned maps, when there's no cache folder to keep them in.
    #[serde(skip)]
    pub memory_maps: HashMap<String, Map>,
    /// Thumbnails, likewise.
    #[serde(skip)]
    pub memory_thumbs: HashMap<String, Thumb>,
}

pub fn key(path: &Path) -> String {
    crate::paths::key(path)
}

/// Bring an older cache up to date, or `None` to start empty.
/// 4 → 5: XISF entries said one plane for RGB images (the channel count
/// is in the geometry, not NAXIS3), so a PixInsight colour image passed
/// as a raw light. Only those entries are forgotten; FITS measurements stay.
fn migrate(mut c: Cache) -> Option<Cache> {
    match c.version {
        CACHE_VERSION => Some(c),
        4 => {
            c.entries.retain(|_, e| e.meta.format != crate::meta::FileFormat::Xisf);
            c.version = CACHE_VERSION;
            c.dirty = true;
            Some(c)
        }
        _ => None,
    }
}

impl Cache {
    pub fn disabled() -> Self {
        Cache { version: CACHE_VERSION, ..Default::default() }
    }

    /// Load from `dir/cache.json`; a missing, unreadable or outdated file gives an empty cache.
    pub fn load(dir: &Path) -> Self {
        let file = dir.join("cache.json");
        let mut c = std::fs::read(&file)
            .ok()
            .and_then(|b| serde_json::from_slice::<Cache>(&b).ok())
            .and_then(migrate)
            .unwrap_or_else(|| Cache { version: CACHE_VERSION, ..Default::default() });
        c.file = Some(file);
        c
    }

    /// A cached entry, only if the file hasn't changed.
    pub fn get(&self, path: &Path, size: u64, mtime_ns: i64) -> Option<&Entry> {
        self.entries.get(&key(path)).filter(|e| e.size == size && e.mtime_ns == mtime_ns)
    }

    pub fn put(&mut self, meta: FrameMeta) {
        let k = key(&meta.path);
        let analysis = self
            .entries
            .get(&k)
            .filter(|e| e.size == meta.file_size && e.mtime_ns == meta.mtime_ns)
            .and_then(|e| e.analysis.clone());
        self.entries.insert(k, Entry { size: meta.file_size, mtime_ns: meta.mtime_ns, meta, analysis });
        self.dirty = true;
    }

    pub fn set_analysis(&mut self, path: &Path, value: serde_json::Value) {
        if let Some(e) = self.entries.get_mut(&key(path)) {
            e.analysis = Some(value);
            self.dirty = true;
        }
    }

    /// Forget entries under `root` that weren't seen in this scan, and map
    /// files that no current entry owns.
    pub fn prune(&mut self, root: &Path, seen: &std::collections::HashSet<String>) {
        let prefix = key(root);
        let before = self.entries.len();
        self.entries.retain(|k, _| !k.starts_with(&prefix) || seen.contains(k));
        self.dirty |= self.entries.len() != before;
        let live: std::collections::HashSet<String> = self
            .entries
            .iter()
            .flat_map(|(k, e)| [map_name(k, e.size, e.mtime_ns), thumb_name(k, e.size, e.mtime_ns)])
            .collect();
        for (dir, ext) in [(self.maps_dir(), ".map"), (self.thumbs_dir(), ".thm")] {
            let Some(rd) = dir.and_then(|d| std::fs::read_dir(d).ok()) else { continue };
            for f in rd.flatten() {
                let name = f.file_name().to_string_lossy().to_string();
                if name.ends_with(ext) && !live.contains(&name) {
                    let _ = std::fs::remove_file(f.path());
                }
            }
        }
    }

    /// A file moved: keep its entry and map under the new path (size and
    /// mtime are unchanged by a move).
    pub fn rename(&mut self, from: &Path, to: &Path) {
        let Some(mut e) = self.entries.remove(&key(from)) else { return };
        e.meta.path = to.to_path_buf();
        if let Some(dir) = self.maps_dir() {
            let (a, b) = (dir.join(map_name(&key(from), e.size, e.mtime_ns)), dir.join(map_name(&key(to), e.size, e.mtime_ns)));
            if a.exists() && !b.exists() {
                let _ = std::fs::rename(a, b);
            }
        }
        if let Some(m) = self.memory_maps.remove(&map_name(&key(from), e.size, e.mtime_ns)) {
            self.memory_maps.insert(map_name(&key(to), e.size, e.mtime_ns), m);
        }
        if let Some(dir) = self.thumbs_dir() {
            let (a, b) = (dir.join(thumb_name(&key(from), e.size, e.mtime_ns)), dir.join(thumb_name(&key(to), e.size, e.mtime_ns)));
            if a.exists() && !b.exists() {
                let _ = std::fs::rename(a, b);
            }
        }
        if let Some(t) = self.memory_thumbs.remove(&thumb_name(&key(from), e.size, e.mtime_ns)) {
            self.memory_thumbs.insert(thumb_name(&key(to), e.size, e.mtime_ns), t);
        }
        self.entries.insert(key(to), e);
        self.dirty = true;
    }

    /// Take entries from another folder's cache for files that moved from
    /// there to here (`(from, to)` pairs): the entry under its new path,
    /// with its map and thumbnail copied across. Entries already here stay.
    /// Returns how many were taken.
    pub fn adopt(&mut self, other: &Cache, moves: &[(PathBuf, PathBuf)]) -> usize {
        let mut n = 0;
        for (from, to) in moves {
            let Some(e) = other.entries.get(&key(from)) else { continue };
            if self.entries.contains_key(&key(to)) {
                continue;
            }
            let mut e = e.clone();
            e.meta.path = to.clone();
            for (src, dst, name) in [
                (other.maps_dir(), self.maps_dir(), map_name as fn(&str, u64, i64) -> String),
                (other.thumbs_dir(), self.thumbs_dir(), thumb_name),
            ] {
                let (Some(src), Some(dst)) = (src, dst) else { continue };
                let a = src.join(name(&key(from), e.size, e.mtime_ns));
                let b = dst.join(name(&key(to), e.size, e.mtime_ns));
                if a.is_file() && !b.exists() && std::fs::create_dir_all(&dst).is_ok() {
                    let _ = std::fs::copy(&a, &b);
                }
            }
            self.entries.insert(key(to), e);
            self.dirty = true;
            n += 1;
        }
        n
    }

    fn thumbs_dir(&self) -> Option<PathBuf> {
        self.file.as_ref().and_then(|f| f.parent()).map(|d| d.join("thumbs"))
    }

    /// The stretched thumbnail stored for a file, if it hasn't changed since.
    pub fn get_thumb(&self, path: &Path, size: u64, mtime_ns: i64) -> Option<Thumb> {
        let name = thumb_name(&key(path), size, mtime_ns);
        if let Some(t) = self.memory_thumbs.get(&name) {
            return Some(t.clone());
        }
        let b = std::fs::read(self.thumbs_dir()?.join(&name)).ok()?;
        Thumb::from_bytes(&b)
    }

    pub fn has_thumb(&self, path: &Path, size: u64, mtime_ns: i64) -> bool {
        let name = thumb_name(&key(path), size, mtime_ns);
        self.memory_thumbs.contains_key(&name) || self.thumbs_dir().is_some_and(|d| d.join(&name).exists())
    }

    pub fn put_thumb(&mut self, path: &Path, size: u64, mtime_ns: i64, t: &Thumb) {
        let name = thumb_name(&key(path), size, mtime_ns);
        let written = self.thumbs_dir().is_some_and(|d| {
            std::fs::create_dir_all(&d).is_ok() && {
                let tmp = d.join(format!("{name}.tmp"));
                std::fs::write(&tmp, t.to_bytes()).is_ok() && std::fs::rename(&tmp, d.join(&name)).is_ok()
            }
        });
        if !written {
            self.memory_thumbs.insert(name, t.clone());
        }
    }

    /// Folder holding the cache (and decisions, plans, maps), if any.
    pub fn dir(&self) -> Option<&Path> {
        self.file.as_ref().and_then(|f| f.parent())
    }

    fn maps_dir(&self) -> Option<PathBuf> {
        self.file.as_ref().and_then(|f| f.parent()).map(|d| d.join("maps"))
    }

    /// The binned map stored for a file, if it hasn't changed since.
    pub fn get_map(&self, path: &Path, size: u64, mtime_ns: i64) -> Option<Map> {
        let name = map_name(&key(path), size, mtime_ns);
        if let Some(m) = self.memory_maps.get(&name) {
            return Some(m.clone());
        }
        let b = std::fs::read(self.maps_dir()?.join(&name)).ok()?;
        Map::from_bytes(&b)
    }

    pub fn put_map(&mut self, path: &Path, size: u64, mtime_ns: i64, map: &Map) {
        let name = map_name(&key(path), size, mtime_ns);
        let written = self.maps_dir().is_some_and(|d| {
            std::fs::create_dir_all(&d).is_ok() && {
                let tmp = d.join(format!("{name}.tmp"));
                std::fs::write(&tmp, map.to_bytes()).is_ok() && std::fs::rename(&tmp, d.join(&name)).is_ok()
            }
        });
        if !written {
            self.memory_maps.insert(name, map.clone());
        }
    }

    /// Write atomically (temp file, then rename). No-op when clean or disabled.
    pub fn save(&mut self) -> Result<()> {
        let Some(file) = &self.file else { return Ok(()) };
        if !self.dirty {
            return Ok(());
        }
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        let tmp = file.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(self)?).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, file).with_context(|| format!("replace {}", file.display()))?;
        self.dirty = false;
        Ok(())
    }
}

/// File name for a frame's thumbnail (independent of the analysis version).
pub fn thumb_name(key: &str, size: u64, mtime_ns: i64) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    let s = format!("{key}|{size}|{mtime_ns}|thumb{}", crate::stretch::THUMB_VERSION);
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}.thm")
}

/// A stretched 8-bit greyscale thumbnail.
#[derive(Debug, Clone, PartialEq)]
pub struct Thumb {
    pub w: u32,
    pub h: u32,
    pub data: Vec<u8>,
}

impl Thumb {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(16 + self.data.len());
        b.extend_from_slice(b"CRUXTHM1");
        b.extend_from_slice(&self.w.to_le_bytes());
        b.extend_from_slice(&self.h.to_le_bytes());
        b.extend_from_slice(&self.data);
        b
    }
    pub fn from_bytes(b: &[u8]) -> Option<Thumb> {
        if b.len() < 16 || &b[..8] != b"CRUXTHM1" {
            return None;
        }
        let w = u32::from_le_bytes(b[8..12].try_into().ok()?);
        let h = u32::from_le_bytes(b[12..16].try_into().ok()?);
        let data = b[16..].to_vec();
        (data.len() == (w * h) as usize).then_some(Thumb { w, h, data })
    }
}

/// File name for a frame's map: a hash of path, size, mtime and the analysis
/// version, so a changed file or a new version never reads a stale map.
fn map_name(key: &str, size: u64, mtime_ns: i64) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    let s = format!("{key}|{size}|{mtime_ns}|{}", crate::analysis::ANALYSIS_VERSION);
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}.map")
}
