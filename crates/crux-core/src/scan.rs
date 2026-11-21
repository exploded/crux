//! Walk a folder, read every FITS/XISF header in parallel (cache first),
//! and classify each frame.

use crate::cache::{self, Cache};
use crate::classify::{classify, Classification};
use crate::config::{wildcard_match, Config};
use crate::meta::{file_stamp, format_of, read_meta, FrameMeta};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub meta: FrameMeta,
    pub class: Classification,
}

impl Frame {
    pub fn path(&self) -> &Path {
        &self.meta.path
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScanStats {
    pub files: usize,
    pub cache_hits: usize,
    pub errors: usize,
    pub skipped_dirs: Vec<PathBuf>,
    pub seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub root: PathBuf,
    pub frames: Vec<Frame>,
    pub errors: Vec<(PathBuf, String)>,
    pub stats: ScanStats,
}

/// Image files under `root`, folders skipped by `[scan] exclude_dirs`, and
/// online-only cloud files left alone (reading one would download it).
#[derive(Debug, Default)]
pub struct Listing {
    pub files: Vec<PathBuf>,
    pub skipped: Vec<PathBuf>,
    pub cloud: Vec<PathBuf>,
}

/// List image files under `root`, skipping excluded folders and
/// online-only cloud placeholders.
pub fn list_files(root: &Path, cfg: &Config) -> Listing {
    let mut skipped = Vec::new();
    let mut files = Vec::new();
    let mut cloud = Vec::new();
    let mut it = walkdir::WalkDir::new(root).follow_links(false).into_iter();
    while let Some(entry) = it.next() {
        let Ok(e) = entry else { continue };
        if e.file_type().is_dir() {
            if e.depth() > 0 {
                let name = e.file_name().to_string_lossy();
                if cfg.scan.exclude_dirs.iter().any(|p| wildcard_match(p, &name)) {
                    skipped.push(e.path().to_path_buf());
                    it.skip_current_dir();
                }
            }
            continue;
        }
        let ok_ext = e
            .path()
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| cfg.scan.extensions.iter().any(|w| w.eq_ignore_ascii_case(x)));
        if ok_ext && format_of(e.path()).is_some() {
            // The entry's metadata comes from the directory listing, so
            // checking it doesn't open (or download) the file.
            if e.metadata().is_ok_and(|m| crate::paths::cloud_only(&m)) {
                cloud.push(e.path().to_path_buf());
            } else {
                files.push(e.path().to_path_buf());
            }
        }
    }
    files.sort();
    cloud.sort();
    Listing { files, skipped, cloud }
}

/// Scan `root`. `progress(done, total)` is called from worker threads.
pub fn scan(root: &Path, cfg: &Config, cache: &mut Cache, progress: &(dyn Fn(usize, usize) + Sync)) -> ScanResult {
    let t0 = Instant::now();
    let root = std::path::absolute(root).unwrap_or_else(|_| root.to_path_buf());
    let Listing { files, skipped, cloud } = list_files(&root, cfg);
    let total = files.len();
    let done = AtomicUsize::new(0);
    let hits = AtomicUsize::new(0);

    let cache_ref: &Cache = cache;
    let results: Vec<(PathBuf, Result<(FrameMeta, bool), String>)> = files
        .par_iter()
        .map(|p| {
            let r = (|| {
                let (size, mtime) = file_stamp(p).map_err(|e| e.to_string())?;
                if let Some(e) = cache_ref.get(p, size, mtime) {
                    hits.fetch_add(1, Ordering::Relaxed);
                    let mut m = e.meta.clone();
                    m.path = p.clone();
                    return Ok((m, false));
                }
                read_meta(p).map(|m| (m, true)).map_err(|e| format!("{e:#}"))
            })();
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            progress(n, total);
            (p.clone(), r)
        })
        .collect();

    let mut frames = Vec::with_capacity(total);
    let mut errors = Vec::new();
    let mut seen = HashSet::new();
    for (p, r) in results {
        seen.insert(cache::key(&p));
        match r {
            Ok((mut meta, fresh)) => {
                if fresh {
                    cache.put(meta.clone());
                }
                crate::meta::fill_from_name(&mut meta);
                let class = classify(&meta, &root, cfg);
                frames.push(Frame { meta, class });
            }
            Err(e) => errors.push((p, e)),
        }
    }
    for p in cloud {
        seen.insert(cache::key(&p));
        errors.push((p, "online-only in OneDrive or another cloud folder: not downloaded (make it available offline to include it)".into()));
    }
    cache.prune(&root, &seen);

    ScanResult {
        stats: ScanStats {
            files: total,
            cache_hits: hits.into_inner(),
            errors: errors.len(),
            skipped_dirs: skipped,
            seconds: t0.elapsed().as_secs_f64(),
        },
        root,
        frames,
        errors,
    }
}
