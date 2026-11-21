//! Apply a plan, and undo it.
//!
//! - **Never overwrites.** Every destination is checked before anything
//!   moves; one clash stops the whole plan.
//! - **Never deletes.** A move on one volume is a rename. Across volumes the
//!   file is copied, synced, checked (size and SHA-256), renamed into place,
//!   and only then is the source removed.
//! - **Journaled.** Each completed move is appended to a JSON-lines journal
//!   and synced before the next one starts, so an interrupted run can still
//!   be undone exactly.

use crate::cache::Cache;
use crate::library::Library;
use crate::plan::{find_conflicts, Plan};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};

pub const JOURNAL_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    Rename,
    Copy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Record {
    Start {
        version: u32,
        plan: Option<PathBuf>,
        root: PathBuf,
        started: String,
        cache_dir: Option<PathBuf>,
        library_root: Option<PathBuf>,
        /// For an undo journal: the journal it reverses.
        undo_of: Option<PathBuf>,
    },
    /// A folder this run created (undo removes it again if it's empty).
    Dir { path: PathBuf },
    /// A source folder this run emptied and removed (undo recreates it).
    Emptied { path: PathBuf },
    Move { from: PathBuf, to: PathBuf, size: u64, method: Method, sha256: Option<String> },
    /// A copy of a crux file taken before changing it; undo puts it back.
    Backup { path: PathBuf, backup: PathBuf },
    /// The flat library in `dir` as it was before this run, saved to its
    /// own database file; undo puts it back.
    Library { dir: PathBuf, snapshot: PathBuf },
    Finish { finished: String, moved: usize },
}

#[derive(Clone, Default)]
pub struct Options {
    /// Copy and verify even on one volume (tests the cross-drive path).
    pub force_copy: bool,
    /// Called after each file: (done, total, file just moved).
    pub progress: Option<std::sync::Arc<dyn Fn(usize, usize, &Path) + Send + Sync>>,
}

/// One apply in the history: its journal, when, how much, and whether it
/// has been undone.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub journal: PathBuf,
    pub plan: Option<PathBuf>,
    pub root: PathBuf,
    pub started: String,
    pub moves: usize,
    pub bytes: u64,
    pub finished: bool,
    /// The undo journal, when this apply was reversed.
    pub undone: Option<PathBuf>,
}

/// Undo journal path for a journal (`x.journal.jsonl` → `x.journal.undo.jsonl`).
pub fn undo_path(journal: &Path) -> PathBuf {
    let s = journal.to_string_lossy();
    PathBuf::from(format!("{}.undo.jsonl", s.strip_suffix(".jsonl").unwrap_or(&s)))
}

/// Every apply journal in a folder (the cache's `plans`), newest first.
pub fn history(dir: &Path) -> Vec<HistoryEntry> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    for e in rd.flatten() {
        let p = e.path();
        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if !name.ends_with(".journal.jsonl") {
            continue;
        }
        let Ok(recs) = read_journal(&p) else { continue };
        let Some(Record::Start { plan, root, started, undo_of: None, .. }) = recs.first().cloned() else { continue };
        let moves: Vec<u64> = recs.iter().filter_map(|r| if let Record::Move { size, .. } = r { Some(*size) } else { None }).collect();
        let finished = recs.iter().any(|r| matches!(r, Record::Finish { .. }));
        let up = undo_path(&p);
        let undone = read_journal(&up).ok().filter(|r| r.iter().any(|x| matches!(x, Record::Finish { .. }))).map(|_| up);
        out.push(HistoryEntry { journal: p, plan, root, started, moves: moves.len(), bytes: moves.iter().sum(), finished, undone });
    }
    out.sort_by(|a, b| b.started.cmp(&a.started));
    out
}

#[derive(Debug, Clone, Default)]
pub struct Outcome {
    pub moved: usize,
    pub copied: usize,
    pub bytes: u64,
    pub journal: PathBuf,
    pub notes: Vec<String>,
}

fn now() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn stamp() -> String {
    chrono::Local::now().format("%Y%m%d-%H%M%S").to_string()
}

fn key(p: &Path) -> String {
    crate::cache::key(p)
}

struct Journal {
    f: std::fs::File,
    path: PathBuf,
}

impl Journal {
    fn create(path: &Path) -> Result<Journal> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))?;
        }
        let f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("create journal {} (it must not exist yet)", path.display()))?;
        Ok(Journal { f, path: path.to_path_buf() })
    }

    fn write(&mut self, r: &Record) -> Result<()> {
        let mut line = serde_json::to_vec(r)?;
        line.push(b'\n');
        self.f.write_all(&line).and_then(|_| self.f.sync_data()).with_context(|| format!("write journal {}", self.path.display()))
    }
}

pub fn read_journal(path: &Path) -> Result<Vec<Record>> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line).with_context(|| format!("{} line {}", path.display(), n + 1))?);
    }
    Ok(out)
}

fn sha256(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Folders that would be created for `dir`, outermost first.
fn missing_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    let mut d = Some(dir);
    while let Some(p) = d {
        if p.as_os_str().is_empty() || p.exists() {
            break;
        }
        v.push(p.to_path_buf());
        d = p.parent();
    }
    v.reverse();
    v
}

/// Move one file without overwriting. Returns how, and the hash when copied.
fn move_file(from: &Path, to: &Path, force_copy: bool) -> Result<(Method, Option<String>)> {
    if to.exists() {
        bail!("{} already exists; not overwriting it", to.display());
    }
    if !force_copy {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok((Method::Rename, None)),
            Err(e) if crate::paths::cross_device(&e) => {}
            Err(e) => return Err(e).with_context(|| format!("move {} to {}", from.display(), to.display())),
        }
    }
    let partial = PathBuf::from(format!("{}.crux-partial", to.display()));
    if partial.exists() {
        bail!("{} exists from an earlier interrupted copy; check it and remove it by hand", partial.display());
    }
    let src_meta = std::fs::metadata(from).with_context(|| format!("read {}", from.display()))?;
    std::fs::copy(from, &partial).with_context(|| format!("copy {} to {}", from.display(), partial.display()))?;
    {
        let f = std::fs::OpenOptions::new().write(true).open(&partial)?;
        if let Ok(t) = src_meta.modified() {
            let _ = f.set_modified(t);
        }
        f.sync_all()?;
    }
    let dst_size = std::fs::metadata(&partial)?.len();
    let (hs, hd) = (sha256(from)?, sha256(&partial)?);
    if dst_size != src_meta.len() || hs != hd {
        let _ = std::fs::remove_file(&partial); // our own incomplete copy
        bail!("copy of {} didn't verify (size {} vs {}, hash differs: {}); source left in place", from.display(), dst_size, src_meta.len(), hs != hd);
    }
    if to.exists() {
        let _ = std::fs::remove_file(&partial);
        bail!("{} appeared during the copy; not overwriting it", to.display());
    }
    std::fs::rename(&partial, to).with_context(|| format!("rename {} to {}", partial.display(), to.display()))?;
    std::fs::remove_file(from).with_context(|| format!("remove {} after a verified copy (the copy is at {})", from.display(), to.display()))?;
    Ok((Method::Copy, Some(hs)))
}

/// Everything that would stop the plan: clashes, and sources that are
/// missing or changed since planning.
pub fn check(plan: &Plan) -> Vec<String> {
    let mut out = plan.conflicts.clone();
    for c in find_conflicts(&plan.moves) {
        if !out.contains(&c) {
            out.push(c);
        }
    }
    for m in &plan.moves {
        match crate::meta::file_stamp(&m.from) {
            Err(_) => out.push(format!("{} is missing (moved or renamed since the plan was made?)", m.from.display())),
            Ok((size, mtime)) if size != m.size || mtime != m.mtime_ns => {
                out.push(format!("{} changed since the plan was made; make a new plan", m.from.display()))
            }
            _ => {}
        }
    }
    out
}

/// Keep the scan cache in step with moved files (so nothing is re-measured).
fn follow_cache(dir: Option<&Path>, moves: &[(PathBuf, PathBuf)]) -> Option<String> {
    let dir = dir?;
    let mut cache = Cache::load(dir);
    for (from, to) in moves {
        cache.rename(from, to);
    }
    let mut notes = Vec::new();
    if let Err(e) = cache.save() {
        notes.push(format!("couldn't update the cache: {e:#}"));
    }
    // The library index follows too.
    let ixp = crate::cachedir::index_for(dir);
    if crate::index::Index::exists(&ixp) {
        match crate::index::Index::load(&ixp) {
            Ok(mut ix) => {
                ix.rename(moves);
                if let Err(e) = ix.save() {
                    notes.push(format!("couldn't update the library index: {e:#}"));
                }
            }
            Err(e) => notes.push(format!("couldn't read the library index: {e:#}")),
        }
    }
    // Your overrides go with the files.
    match crate::decisions::Decisions::load(dir) {
        Ok(mut d) => {
            let mut changed = false;
            for (from, to) in moves {
                changed |= d.rename(from, to);
            }
            if changed {
                if let Err(e) = d.save() {
                    notes.push(format!("couldn't update your decisions: {e:#}"));
                }
            }
        }
        Err(e) => notes.push(format!("couldn't read your decisions: {e:#}")),
    }
    (!notes.is_empty()).then(|| notes.join("; "))
}

/// Apply a checked plan. Call `check` first; this checks again and refuses
/// on any problem before moving anything.
pub fn apply(plan: &Plan, plan_path: Option<&Path>, journal_path: &Path, opts: &Options) -> Result<Outcome> {
    let problems = check(plan);
    if !problems.is_empty() {
        bail!("the plan can't be applied:\n  {}", problems.join("\n  "));
    }
    let mut j = Journal::create(journal_path)?;
    j.write(&Record::Start {
        version: JOURNAL_VERSION,
        plan: plan_path.map(Path::to_path_buf),
        root: plan.root.clone(),
        started: now(),
        cache_dir: plan.cache_dir.clone(),
        library_root: plan.library_root.clone(),
        undo_of: None,
    })?;
    let mut out = Outcome { journal: journal_path.to_path_buf(), ..Default::default() };
    let lib = match (&plan.library_dir, &plan.library_root) {
        (Some(d), Some(r)) => Library::load_at(d, r).ok().flatten(),
        _ => None,
    };
    let touches_library = lib.as_ref().is_some_and(|l| {
        let files: std::collections::HashSet<String> = l.sets.iter().flat_map(|s| s.files.iter().map(|f| key(f))).collect();
        !plan.combine.is_empty() || plan.moves.iter().any(|m| files.contains(&key(&m.from)))
    });
    if let Some(l) = lib.as_ref().filter(|_| touches_library) {
        let snapshot = journal_path.with_extension("library.db");
        l.snapshot(&snapshot).context("save the flat library before changing it")?;
        j.write(&Record::Library { dir: l.dir.clone(), snapshot })?;
    }

    let mut done: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut failure = None;
    for m in &plan.moves {
        let step = (|| -> Result<()> {
            if let Some(parent) = m.to.parent() {
                for d in missing_dirs(parent) {
                    std::fs::create_dir(&d).with_context(|| format!("create {}", d.display()))?;
                    j.write(&Record::Dir { path: d })?;
                }
            }
            let (method, sha) = move_file(&m.from, &m.to, opts.force_copy)?;
            j.write(&Record::Move { from: m.from.clone(), to: m.to.clone(), size: m.size, method, sha256: sha })?;
            if let Some(p) = &opts.progress {
                p(out.moved + 1, plan.moves.len(), &m.to);
            }
            out.moved += 1;
            out.bytes += m.size;
            out.copied += (method == Method::Copy) as usize;
            done.push((m.from.clone(), m.to.clone()));
            Ok(())
        })();
        if let Err(e) = step {
            failure = Some(e);
            break;
        }
    }

    // Source folders left empty, and their parents up to the scanned folder
    // (never the scanned folder itself).
    let mut parents: Vec<PathBuf> = Vec::new();
    for (f, _) in &done {
        let mut p = f.parent();
        while let Some(d) = p {
            if key(d) == key(&plan.root) || !d.starts_with(&plan.root) {
                break;
            }
            parents.push(d.to_path_buf());
            p = d.parent();
        }
    }
    parents.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    parents.dedup();
    for p in parents {
        if key(&p) != key(&plan.root) && std::fs::read_dir(&p).is_ok_and(|mut d| d.next().is_none()) && std::fs::remove_dir(&p).is_ok() {
            j.write(&Record::Emptied { path: p.clone() })?;
            out.notes.push(format!("removed the now-empty folder {}", p.display()));
        }
    }
    if let Some(n) = follow_cache(plan.cache_dir.as_deref(), &done) {
        out.notes.push(n);
    }
    if plan.target.is_some() && failure.is_none() {
        out.notes.extend(crate::targets::adopt_caches(plan));
    }
    if let Some(lib) = lib.filter(|_| touches_library) {
        let retired: std::collections::HashSet<String> =
            plan.moves.iter().filter(|m| m.kind == crate::plan::MoveKind::Retire).map(|m| key(&m.from)).collect();
        match update_library(lib, &done, &retired, if failure.is_none() { &plan.combine } else { &[] }) {
            Ok(n) => out.notes.extend(n),
            Err(e) => out.notes.push(format!("couldn't update the library index: {e:#}")),
        }
    }
    if let Some(e) = failure {
        bail!(
            "stopped after {} of {} moves: {e:#}\nNothing else was moved. Undo what was done from Organise → History (journal {}).",
            out.moved,
            plan.moves.len(),
            journal_path.display()
        );
    }
    j.write(&Record::Finish { finished: now(), moved: out.moved })?;
    Ok(out)
}

/// Point the index at the files' new places, and merge combined sets.
fn update_library(
    mut lib: Library,
    done: &[(PathBuf, PathBuf)],
    retired: &std::collections::HashSet<String>,
    combine: &[Vec<String>],
) -> Result<Vec<String>> {
    let root = lib.root.clone();
    let map: HashMap<String, &PathBuf> = done.iter().map(|(f, t)| (key(f), t)).collect();
    let mut notes = Vec::new();
    for s in lib.sets.iter_mut() {
        let mut was_retired = false;
        for f in s.files.iter_mut() {
            was_retired |= retired.contains(&key(f)) && map.contains_key(&key(f));
            if let Some(t) = map.get(&key(f)) {
                *f = (*t).clone();
            }
        }
        if was_retired && !s.retired {
            s.retired = true;
            s.home = crate::library::home_for(&root, &s.filter, s.night, true);
            notes.push(format!("library: {} marked retired", s.label()));
        }
    }
    for g in combine {
        notes.push(lib.merge(g)?);
    }
    lib.save()?;
    Ok(notes)
}

/// Everything that would stop an undo.
pub fn check_undo(records: &[Record]) -> Vec<String> {
    let mut out = Vec::new();
    for r in records {
        if let Record::Move { from, to, size, .. } = r {
            match std::fs::metadata(to) {
                Err(_) => out.push(format!("{} is missing (moved again since?)", to.display())),
                Ok(m) if m.len() != *size => out.push(format!("{} has changed size since it was moved", to.display())),
                _ => {}
            }
            if from.exists() {
                out.push(format!("{} exists again; not overwriting it", from.display()));
            }
        }
    }
    out
}

/// Reverse a journal: moves back in reverse order, empty folders it made
/// removed, crux's backed-up files restored. Writes its own journal.
pub fn undo(journal: &Path, undo_journal: &Path, opts: &Options) -> Result<Outcome> {
    let records = read_journal(journal)?;
    let Some(Record::Start { root, cache_dir, library_root, .. }) = records.first().cloned() else {
        bail!("{} isn't a crux journal (no start record)", journal.display());
    };
    let problems = check_undo(&records);
    if !problems.is_empty() {
        bail!("can't undo:\n  {}", problems.join("\n  "));
    }
    let mut j = Journal::create(undo_journal)?;
    j.write(&Record::Start {
        version: JOURNAL_VERSION,
        plan: None,
        root,
        started: now(),
        cache_dir: cache_dir.clone(),
        library_root,
        undo_of: Some(journal.to_path_buf()),
    })?;
    let mut out = Outcome { journal: undo_journal.to_path_buf(), ..Default::default() };
    let mut done = Vec::new();
    let total = records.iter().filter(|r| matches!(r, Record::Move { .. })).count();
    for r in records.iter().rev() {
        if let Record::Move { from, to, size, .. } = r {
            if let Some(parent) = from.parent() {
                for d in missing_dirs(parent) {
                    std::fs::create_dir(&d).with_context(|| format!("create {}", d.display()))?;
                    j.write(&Record::Dir { path: d })?;
                }
            }
            let (method, sha) = move_file(to, from, opts.force_copy)
                .with_context(|| format!("after {} moves back; this undo's own journal is {}", out.moved, undo_journal.display()))?;
            j.write(&Record::Move { from: to.clone(), to: from.clone(), size: *size, method, sha256: sha })?;
            if let Some(p) = &opts.progress {
                p(out.moved + 1, total, from);
            }
            out.moved += 1;
            out.bytes += size;
            out.copied += (method == Method::Copy) as usize;
            done.push((to.clone(), from.clone()));
        }
    }
    // Folders the run created, innermost first; only if now empty.
    for r in records.iter().rev() {
        if let Record::Dir { path } = r {
            let _ = std::fs::remove_dir(path);
        }
    }
    for r in records.iter().rev() {
        if let Record::Library { dir, snapshot } = r {
            if let Some(now) = Library::load_at(dir, dir)? {
                let aside = undo_journal.with_extension("library.db");
                now.snapshot(&aside)?;
                j.write(&Record::Library { dir: dir.clone(), snapshot: aside })?;
            }
            Library::restore(snapshot, dir).with_context(|| format!("restore the flat library from {}", snapshot.display()))?;
            out.notes.push(format!("restored the flat library in {}", Library::db_path(dir).display()));
        }
        if let Record::Backup { path, backup } = r {
            if path.exists() {
                let aside = PathBuf::from(format!("{}.{}.before-undo", path.display(), stamp()));
                std::fs::rename(path, &aside).with_context(|| format!("set aside {}", path.display()))?;
                j.write(&Record::Backup { path: path.clone(), backup: aside })?;
            }
            std::fs::copy(backup, path).with_context(|| format!("restore {} from {}", path.display(), backup.display()))?;
            out.notes.push(format!("restored {}", path.display()));
        }
    }
    if let Some(n) = follow_cache(cache_dir.as_deref(), &done) {
        out.notes.push(n);
    }
    j.write(&Record::Finish { finished: now(), moved: out.moved })?;
    Ok(out)
}
