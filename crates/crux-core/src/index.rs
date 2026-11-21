//! The library index: one record per image file across every folder crux
//! has read, so smart views can query the whole library without rescanning
//! it. Folders opened in full carry grades and measurements; the others
//! carry what their headers say. Plate solutions are kept here too.
//!
//! Stored in SQLite (`index.db` in the cache folder; the `sqlite` feature,
//! on by default), else as `index.json`. Callers pass the `index.json`
//! path either way; an old `index.json` is imported into the database the
//! first time and kept as `index.json.bak`.

use crate::classify::{FrameType, Role};
use crate::grade::{FrameReport, Grade};
use crate::pipeline::Triage;
use anyhow::{Context, Result};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

pub const INDEX_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexFile {
    pub path: PathBuf,
    /// The library folder it was found under.
    pub root: PathBuf,
    pub size: u64,
    pub mtime_ns: i64,
    pub kind: FrameType,
    pub role: Role,
    /// In a reject folder (BAD, *.bad).
    pub rejected_dir: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    /// Object without the panel ("LMC" for "LMC Panel 26").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub panel: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub night: Option<NaiveDate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exptime: Option<f64>,
    /// crux's grade (None until the folder has been opened in full).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grade: Option<Grade>,
    /// Main reject reason, e.g. "cloud".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fwhm: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ecc: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stars: Option<u32>,
    /// Sky, e-/px/s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sky: Option<f64>,
    /// Target altitude, degrees.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alt: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pier: Option<String>,
    /// SHA-256, once something needed it (duplicate checks).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

impl Default for IndexFile {
    fn default() -> Self {
        IndexFile {
            path: PathBuf::new(),
            root: PathBuf::new(),
            size: 0,
            mtime_ns: 0,
            kind: FrameType::Unknown,
            role: Role::Raw,
            rejected_dir: false,
            object: None,
            target: None,
            panel: None,
            filter: None,
            night: None,
            exptime: None,
            grade: None,
            reason: None,
            t: None,
            fwhm: None,
            ecc: None,
            stars: None,
            sky: None,
            alt: None,
            pier: None,
            sha256: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RootInfo {
    pub scanned: String,
    /// Opened in full (grades and measurements), not just headers.
    pub analysed: bool,
    pub files: usize,
}

/// A plate solution for one file version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolveRecord {
    pub size: u64,
    pub mtime_ns: i64,
    /// None: tried and failed.
    pub solution: Option<crate::solve::Solution>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Index {
    pub version: u32,
    pub roots: BTreeMap<String, RootInfo>,
    pub files: Vec<IndexFile>,
    /// Path key → plate solution.
    #[serde(default)]
    pub solves: BTreeMap<String, SolveRecord>,
    /// The `index.json` path callers gave (the database sits beside it).
    #[serde(skip)]
    pub file: Option<PathBuf>,
}

/// "LMC" from "LMC Panel 26"; the whole object otherwise.
pub fn target_of(object: &str) -> String {
    let low = object.to_lowercase();
    let t = match low.find(" panel") {
        Some(i) => object[..i].trim(),
        None => object.trim(),
    };
    t.to_string()
}

/// 26 from "LMC Panel 26".
pub fn panel_of(object: &str) -> Option<u32> {
    let low = object.to_lowercase();
    let i = low.find("panel")?;
    object[i + 5..].trim_start_matches([' ', '_', '-']).chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()
}

/// Index records for one folder. With `reports` (an opened folder) they
/// carry grades and measurements.
pub fn records(t: &Triage, reports: Option<&[FrameReport]>) -> Vec<IndexFile> {
    let by: HashMap<usize, &FrameReport> = reports.map(|r| r.iter().chain(&t.cal.reports).map(|x| (x.frame, x)).collect()).unwrap_or_default();
    t.scan
        .frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let m = &f.meta;
            let r = by.get(&i);
            let lm = t.metrics.get(&i);
            IndexFile {
                path: m.path.clone(),
                root: t.scan.root.clone(),
                size: m.file_size,
                mtime_ns: m.mtime_ns,
                kind: f.class.frame_type,
                role: f.class.role,
                rejected_dir: f.class.in_reject_dir,
                object: m.object.clone(),
                target: m.object.as_deref().map(target_of),
                panel: m.object.as_deref().and_then(panel_of),
                filter: m.filter.clone().map(|x| x.to_uppercase()),
                night: f.class.night,
                exptime: m.exptime,
                grade: r.map(|r| r.grade),
                reason: r.filter(|r| r.grade == Grade::Reject).and_then(|r| crate::plan::main_reject(r)).map(|c| crate::plan::reason_slug(c).to_string()),
                t: r.and_then(|r| r.transparency.as_ref()).map(|x| x.value),
                fwhm: lm.and_then(|x| x.fwhm).map(f64::from),
                ecc: lm.and_then(|x| x.ecc).map(f64::from),
                stars: lm.map(|x| x.n_stars),
                sky: r.and_then(|r| r.sky_e_s),
                alt: r.and_then(|r| r.sky.and_then(|s| s.target_alt)),
                pier: m.pierside.clone(),
                sha256: None,
            }
        })
        .collect()
}

impl Index {
    /// The SQLite file for an `index.json` path.
    pub fn db_path(json: &Path) -> PathBuf {
        json.with_extension("db")
    }

    /// Start again: move the index database (with the flat library, which
    /// shares it) aside as `index.db.<date-time>.bak`, and an old JSON index
    /// likewise. The next open and Refresh calibration library rebuild them
    /// from the files. Returns where the backup went. Nothing may have the
    /// database open (Windows refuses the rename).
    pub fn reset(json: &Path) -> Result<Option<PathBuf>> {
        let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
        let mut main = None;
        let db = Self::db_path(json);
        for p in [db.clone(), json.to_path_buf()] {
            if !p.is_file() {
                continue;
            }
            #[cfg(feature = "sqlite")]
            if p == db {
                // Fold the write-ahead log in first, so the backup is whole.
                if let Ok(c) = rusqlite::Connection::open(&p) {
                    let _ = c.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
                }
            }
            let bak = PathBuf::from(format!("{}.{stamp}.bak", p.display()));
            std::fs::rename(&p, &bak).with_context(|| format!("move {} aside (is crux open in another window?)", p.display()))?;
            for side in ["-wal", "-shm"] {
                let s = PathBuf::from(format!("{}{side}", p.display()));
                if s.is_file() {
                    let _ = std::fs::remove_file(&s);
                }
            }
            main.get_or_insert(bak);
        }
        Ok(main)
    }

    /// Is there an index (database or JSON) at this `index.json` path?
    pub fn exists(json: &Path) -> bool {
        (cfg!(feature = "sqlite") && Self::db_path(json).is_file()) || json.is_file()
    }

    pub fn load(path: &Path) -> Result<Index> {
        #[cfg(feature = "sqlite")]
        {
            let db = Self::db_path(path);
            if db.is_file() {
                let mut ix = db::load(&db).with_context(|| format!("read {}", db.display()))?;
                ix.file = Some(path.to_path_buf());
                return Ok(ix);
            }
        }
        let mut ix = match std::fs::read(path) {
            Ok(b) => serde_json::from_slice::<Index>(&b)
                .ok()
                .filter(|i| i.version == INDEX_VERSION)
                .unwrap_or(Index { version: INDEX_VERSION, ..Default::default() }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Index { version: INDEX_VERSION, ..Default::default() },
            Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
        };
        ix.file = Some(path.to_path_buf());
        Ok(ix)
    }

    pub fn save(&self) -> Result<()> {
        let Some(file) = &self.file else { return Ok(()) };
        if let Some(d) = file.parent() {
            std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))?;
        }
        #[cfg(feature = "sqlite")]
        {
            let db = Self::db_path(file);
            db::save(&db, self).with_context(|| format!("write {}", db.display()))?;
            // Imported: keep the old JSON as a backup, out of the way.
            if file.is_file() {
                let _ = std::fs::rename(file, file.with_extension("json.bak"));
            }
            Ok(())
        }
        #[cfg(not(feature = "sqlite"))]
        {
            let tmp = file.with_extension("json.tmp");
            std::fs::write(&tmp, serde_json::to_vec(self)?).with_context(|| format!("write {}", tmp.display()))?;
            std::fs::rename(&tmp, file).with_context(|| format!("replace {}", file.display()))?;
            Ok(())
        }
    }

    /// The stored solution for this file version: `Some(None)` = tried and
    /// failed. Solutions from an older solver count as missing.
    pub fn solution(&self, path: &Path, size: u64, mtime_ns: i64) -> Option<Option<&crate::solve::Solution>> {
        self.solves
            .get(&crate::paths::key(path))
            .filter(|r| r.size == size && r.mtime_ns == mtime_ns)
            .filter(|r| r.solution.as_ref().is_none_or(|s| s.version >= crate::solve::SOLVE_VERSION))
            .map(|r| r.solution.as_ref())
    }

    pub fn set_solution(&mut self, path: &Path, size: u64, mtime_ns: i64, solution: Option<crate::solve::Solution>) {
        self.solves.insert(crate::paths::key(path), SolveRecord { size, mtime_ns, solution });
    }

    /// Replace one folder's records. Header-only records keep grades,
    /// measurements and hashes from an earlier full read of the same file
    /// (same path, size and time).
    pub fn set_root(&mut self, root: &Path, mut recs: Vec<IndexFile>, analysed: bool) {
        let k = crate::cache::key(root);
        let old: HashMap<String, IndexFile> = self
            .files
            .iter()
            .filter(|f| crate::cache::key(&f.root) == k)
            .map(|f| (crate::cache::key(&f.path), f.clone()))
            .collect();
        for r in recs.iter_mut() {
            if let Some(o) = old.get(&crate::cache::key(&r.path)).filter(|o| o.size == r.size && o.mtime_ns == r.mtime_ns) {
                if r.sha256.is_none() {
                    r.sha256 = o.sha256.clone();
                }
                if !analysed && r.grade.is_none() {
                    r.grade = o.grade;
                    r.reason = o.reason.clone();
                    r.t = o.t;
                    r.fwhm = r.fwhm.or(o.fwhm);
                    r.ecc = r.ecc.or(o.ecc);
                    r.stars = r.stars.or(o.stars);
                    r.sky = o.sky;
                    r.alt = o.alt;
                }
            }
        }
        let was_analysed = self.roots.get(&k).is_some_and(|r| r.analysed);
        self.files.retain(|f| crate::cache::key(&f.root) != k);
        self.roots.insert(
            k,
            RootInfo { scanned: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(), analysed: analysed || was_analysed, files: recs.len() },
        );
        self.files.extend(recs);
    }

    /// Files that moved: their records follow (no re-reading).
    pub fn rename(&mut self, moves: &[(PathBuf, PathBuf)]) {
        let map: HashMap<String, &PathBuf> = moves.iter().map(|(f, t)| (crate::cache::key(f), t)).collect();
        for f in self.files.iter_mut() {
            if let Some(t) = map.get(&crate::cache::key(&f.path)) {
                f.path = (*t).clone();
            }
        }
        for (from, to) in moves {
            if let Some(r) = self.solves.remove(&crate::paths::key(from)) {
                self.solves.insert(crate::paths::key(to), r);
            }
        }
    }
}

#[cfg(feature = "sqlite")]
mod db {
    //! SQLite storage. Records are JSON in a column, with the fields smart
    //! views and processing filter on as indexed columns.
    use super::*;
    use rusqlite::{params, Connection};

    const SCHEMA: &str = "
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS roots (key TEXT PRIMARY KEY, scanned TEXT NOT NULL, analysed INTEGER NOT NULL, files INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS files (
            key TEXT PRIMARY KEY, root TEXT NOT NULL, path TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
            kind TEXT, target TEXT, filter TEXT, night TEXT, grade TEXT, record TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS files_root ON files(root);
        CREATE INDEX IF NOT EXISTS files_target ON files(target, filter, night);
        CREATE TABLE IF NOT EXISTS solves (key TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, solution TEXT);
    ";

    fn open(p: &Path) -> Result<Connection> {
        let c = Connection::open(p)?;
        c.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        c.execute_batch(SCHEMA)?;
        Ok(c)
    }

    pub fn load(p: &Path) -> Result<Index> {
        let c = open(p)?;
        let version: u32 = c
            .query_row("SELECT value FROM meta WHERE key = 'version'", [], |r| r.get::<_, String>(0))
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mut ix = Index { version: INDEX_VERSION, ..Default::default() };
        if version != INDEX_VERSION {
            return Ok(ix);
        }
        let mut st = c.prepare("SELECT key, scanned, analysed, files FROM roots")?;
        let rows = st.query_map([], |r| Ok((r.get::<_, String>(0)?, RootInfo { scanned: r.get(1)?, analysed: r.get::<_, i64>(2)? != 0, files: r.get::<_, i64>(3)? as usize })))?;
        for r in rows {
            let (k, v) = r?;
            ix.roots.insert(k, v);
        }
        let mut st = c.prepare("SELECT record FROM files ORDER BY rowid")?;
        for r in st.query_map([], |r| r.get::<_, String>(0))? {
            if let Ok(f) = serde_json::from_str::<IndexFile>(&r?) {
                ix.files.push(f);
            }
        }
        let mut st = c.prepare("SELECT key, size, mtime_ns, solution FROM solves")?;
        let rows = st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, Option<String>>(3)?)))?;
        for r in rows {
            let (k, size, mtime_ns, sol) = r?;
            let solution = sol.and_then(|s| serde_json::from_str(&s).ok());
            ix.solves.insert(k, SolveRecord { size: size as u64, mtime_ns, solution });
        }
        Ok(ix)
    }

    pub fn save(p: &Path, ix: &Index) -> Result<()> {
        let mut c = open(p)?;
        let tx = c.transaction()?;
        tx.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('version', ?1)", params![INDEX_VERSION.to_string()])?;
        tx.execute("DELETE FROM roots", [])?;
        tx.execute("DELETE FROM files", [])?;
        tx.execute("DELETE FROM solves", [])?;
        {
            let mut st = tx.prepare("INSERT INTO roots (key, scanned, analysed, files) VALUES (?1, ?2, ?3, ?4)")?;
            for (k, r) in &ix.roots {
                st.execute(params![k, r.scanned, r.analysed as i64, r.files as i64])?;
            }
            let mut st = tx.prepare(
                "INSERT OR REPLACE INTO files (key, root, path, size, mtime_ns, kind, target, filter, night, grade, record) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            )?;
            for f in &ix.files {
                st.execute(params![
                    crate::paths::key(&f.path),
                    crate::paths::key(&f.root),
                    f.path.to_string_lossy(),
                    f.size as i64,
                    f.mtime_ns,
                    f.kind.name(),
                    f.target,
                    f.filter,
                    f.night.map(|n| n.to_string()),
                    f.grade.map(|g| format!("{g:?}").to_lowercase()),
                    serde_json::to_string(f)?
                ])?;
            }
            let mut st = tx.prepare("INSERT INTO solves (key, size, mtime_ns, solution) VALUES (?1, ?2, ?3, ?4)")?;
            for (k, r) in &ix.solves {
                let sol = match &r.solution {
                    Some(x) => Some(serde_json::to_string(x)?),
                    None => None,
                };
                st.execute(params![k, r.size as i64, r.mtime_ns, sol])?;
            }
        }
        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(target_of("LMC Panel 26"), "LMC");
        assert_eq!(panel_of("LMC Panel 26"), Some(26));
        assert_eq!(panel_of("M 42 panel_3"), Some(3));
        assert_eq!(target_of("NGC 7293"), "NGC 7293");
        assert_eq!(panel_of("NGC 7293"), None);
    }

    #[test]
    fn keeps_grades_on_header_rescan() {
        let mut ix = Index::default();
        let root = PathBuf::from(r"C:\x");
        let f = IndexFile { path: root.join("a.fits"), root: root.clone(), size: 10, mtime_ns: 5, grade: Some(Grade::Reject), t: Some(0.3), ..Default::default() };
        ix.set_root(&root, vec![f.clone()], true);
        ix.set_root(&root, vec![IndexFile { grade: None, t: None, ..f.clone() }], false);
        assert_eq!(ix.files[0].grade, Some(Grade::Reject));
        assert!(ix.roots[&crate::cache::key(&root)].analysed);
        // A changed file loses them.
        ix.set_root(&root, vec![IndexFile { grade: None, t: None, mtime_ns: 6, ..f }], false);
        assert_eq!(ix.files[0].grade, None);
    }

    #[test]
    fn saves_and_imports() {
        let dir = std::env::temp_dir().join(format!("crux-index-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let json = dir.join("index.json");
        // An old JSON index is read, then saved (to the database with sqlite).
        let root = dir.join("data");
        let f = IndexFile { path: root.join("a.fits"), root: root.clone(), size: 10, mtime_ns: 5, grade: Some(Grade::Warn), ..Default::default() };
        let mut old = Index { version: INDEX_VERSION, ..Default::default() };
        old.set_root(&root, vec![f.clone()], true);
        std::fs::write(&json, serde_json::to_vec(&old).unwrap()).unwrap();
        let mut ix = Index::load(&json).unwrap();
        assert_eq!(ix.files.len(), 1);
        ix.set_solution(&f.path, 10, 5, None);
        ix.save().unwrap();
        let back = Index::load(&json).unwrap();
        assert_eq!(back.files, ix.files);
        assert_eq!(back.solution(&f.path, 10, 5), Some(None));
        assert_eq!(back.solution(&f.path, 11, 5), None);
        if cfg!(feature = "sqlite") {
            assert!(Index::db_path(&json).is_file());
            assert!(!json.is_file() && json.with_extension("json.bak").is_file());
        }
        assert!(Index::exists(&json));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
