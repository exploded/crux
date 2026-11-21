//! The triage pipeline: scan headers, classify leftovers from pixels, group,
//! measure lights (cached, a bounded number of frames at a time), grade.

use crate::analysis::{analyze_light, analyze_light_map, LightMetrics, ANALYSIS_VERSION};
use crate::cache::Cache;
use crate::calcheck::{self, CalCheck};
use crate::calib::{analyze_cal, CalMetrics, Map, MAP_BIN};
use crate::library::Library;
use crate::classify::{FrameType, Role, TypeSource};
use crate::config::Config;
use crate::grade::{grade_lights, FrameReport};
use crate::group::{self, Grouping};
use crate::scan::{self, Frame, ScanResult};
use crate::stats::median_f64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Headers,
    Pixels,
    Calibration,
}

/// What's stored in the cache per file, besides the header metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Cached {
    pub v: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub light: Option<LightMetrics>,
    /// Bias, dark and flat measurements.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cal: Option<CalMetrics>,
    /// `calib::CAL_VERSION` the calibration metrics were measured with.
    #[serde(default)]
    pub cal_v: u32,
    /// Bias read noise from this frame and the next one in its set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pair: Option<PairNoise>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixel_type: Option<FrameType>,
}

fn cached(cache: &Cache, f: &Frame) -> Option<Cached> {
    let e = cache.get(&f.meta.path, f.meta.file_size, f.meta.mtime_ns)?;
    let c: Cached = serde_json::from_value(e.analysis.clone()?).ok()?;
    (c.v == ANALYSIS_VERSION).then_some(c)
}

fn store(cache: &mut Cache, f: &Frame, update: impl FnOnce(&mut Cached)) {
    let mut c = cached(cache, f).unwrap_or(Cached { v: ANALYSIS_VERSION, ..Default::default() });
    update(&mut c);
    if let Ok(v) = serde_json::to_value(&c) {
        cache.set_analysis(&f.meta.path, v);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairNoise {
    /// Cache key of the partner frame, with its size and mtime.
    pub partner: String,
    pub noise_adu: f32,
}

pub struct Triage {
    pub scan: ScanResult,
    pub grouping: Grouping,
    pub metrics: HashMap<usize, LightMetrics>,
    pub errors: HashMap<usize, String>,
    /// Bias level (ADU) used for each light.
    pub pedestal: HashMap<usize, f64>,
    pub reports: Vec<FrameReport>,
    /// Each light's transparency within its field (the star matching
    /// behind the grades), kept so the GUI can regrade without redoing it.
    pub transparency: HashMap<usize, crate::transparency::FrameTransparency>,
    /// Bias, dark and flat measurements and read errors.
    pub cal_metrics: HashMap<usize, CalMetrics>,
    pub cal_errors: HashMap<usize, String>,
    /// Calibration frame and set checks.
    pub cal: CalCheck,
    /// Read noise per raw bias group (frame pairs), ADU.
    pub pair_noise: HashMap<usize, f64>,
    /// The flat library index, when one exists at `library.root`.
    pub library: Option<Library>,
    pub library_error: Option<String>,
    /// Calibration frames measured this run, and how long it took.
    pub cal_analysed: usize,
    pub cal_seconds: f64,
    /// Lights measured this run (not from cache).
    pub analysed: usize,
    pub analysis_seconds: f64,
}

impl Triage {
    pub fn report_for(&self, frame: usize) -> Option<&FrameReport> {
        self.reports.iter().find(|r| r.frame == frame)
    }
}

/// Run the whole triage. With `analyse = false` only headers are read.
pub fn run(dir: &Path, cfg: &Config, cache: &mut Cache, analyse: bool, progress: &(dyn Fn(Stage, usize, usize) + Sync)) -> Triage {
    run_with(dir, cfg, cache, analyse, progress, &AtomicBool::new(false))
}

/// [`run`] that stops reading pixels once `cancel` is set (between chunks;
/// what was measured is cached). The result is then incomplete.
pub fn run_with(
    dir: &Path,
    cfg: &Config,
    cache: &mut Cache,
    analyse: bool,
    progress: &(dyn Fn(Stage, usize, usize) + Sync),
    cancel: &AtomicBool,
) -> Triage {
    let timing = analyse && std::env::var_os("CRUX_TIMING").is_some();
    let tt = Instant::now();
    let lap = |what: &str| {
        if timing {
            eprintln!("triage: {what} at {:.2} s", tt.elapsed().as_secs_f64());
        }
    };
    let mut scan = scan::scan(dir, cfg, cache, &|d, t| progress(Stage::Headers, d, t));
    if analyse {
        classify_unknowns(&mut scan, cache, cfg);
    }
    lap("scanned");
    let grouping = group::build(&scan.frames, cfg);
    lap("grouped");
    let mut t = Triage {
        scan,
        grouping,
        metrics: HashMap::new(),
        errors: HashMap::new(),
        pedestal: HashMap::new(),
        reports: Vec::new(),
        transparency: HashMap::new(),
        cal_metrics: HashMap::new(),
        cal_errors: HashMap::new(),
        cal: CalCheck::default(),
        pair_noise: HashMap::new(),
        library: None,
        library_error: None,
        cal_analysed: 0,
        cal_seconds: 0.0,
        analysed: 0,
        analysis_seconds: 0.0,
    };
    if !analyse {
        return t;
    }
    let t0 = Instant::now();
    let (cm, ce, n) = analyse_calibration(&t.scan.frames, cache, cfg, progress, cancel);
    t.cal_metrics = cm;
    t.cal_errors = ce;
    t.cal_analysed = n;
    t.cal_seconds = t0.elapsed().as_secs_f64();
    t.pair_noise = bias_pair_noise(&t.scan.frames, &t.grouping, cache, cfg);
    t.cal_seconds = t0.elapsed().as_secs_f64();
    t.pedestal = pedestals(&t.grouping, &t.cal_metrics);
    lap("calibration frames");
    match Library::load(&cfg.library) {
        Ok(l) => t.library = l,
        Err(e) => t.library_error = Some(format!("{e:#}")),
    }
    lap("library loaded");
    let t0 = Instant::now();
    let (metrics, errors, analysed) = analyse_lights(&t.scan.frames, cache, cfg, progress, cancel);
    t.metrics = metrics;
    t.errors = errors;
    t.analysed = analysed;
    t.analysis_seconds = t0.elapsed().as_secs_f64();
    let ped = |i: usize| t.pedestal.get(&i).copied().unwrap_or(cfg.camera.bias_level_adu);
    lap("lights");
    t.transparency = crate::grade::transparency_all(&t.scan.frames, &t.metrics, cfg);
    lap("transparency");
    t.reports = crate::grade::grade_lights_with(&t.scan.frames, &t.metrics, &t.errors, &ped, cfg, &t.transparency);
    lap("graded");
    t.cal = calcheck::check(
        &t.scan.frames,
        &t.grouping,
        &t.cal_metrics,
        &t.cal_errors,
        &t.reports,
        &t.pair_noise,
        t.library.as_ref(),
        cache,
        cfg,
    );
    lap("calibration checked");
    t
}

/// The calibration half of a triage, for the calibration library: headers,
/// then every bias, dark and flat measured (cached) and checked. No light
/// is read: lights already measured in the cache are graded so flats can
/// be checked against them; the rest are left out.
pub fn run_calibration(dir: &Path, cfg: &Config, cache: &mut Cache, progress: &(dyn Fn(Stage, usize, usize) + Sync), cancel: &AtomicBool) -> Triage {
    let mut t = run(dir, cfg, cache, false, progress);
    classify_unknowns(&mut t.scan, cache, cfg);
    t.grouping = group::build(&t.scan.frames, cfg);
    let t0 = Instant::now();
    let (cm, ce, n) = analyse_calibration(&t.scan.frames, cache, cfg, progress, cancel);
    t.cal_metrics = cm;
    t.cal_errors = ce;
    t.cal_analysed = n;
    t.pair_noise = bias_pair_noise(&t.scan.frames, &t.grouping, cache, cfg);
    t.cal_seconds = t0.elapsed().as_secs_f64();
    t.pedestal = pedestals(&t.grouping, &t.cal_metrics);
    // No library: comparing every set with it is the library's own job
    // (consensus, when a folder chooses flats).
    for (i, f) in t.scan.frames.iter().enumerate() {
        if f.class.frame_type == FrameType::Light && f.class.role == Role::Raw {
            if let Some(m) = cached(cache, f).and_then(|c| c.light) {
                t.metrics.insert(i, m);
            }
        }
    }
    let ped = |i: usize| t.pedestal.get(&i).copied().unwrap_or(cfg.camera.bias_level_adu);
    t.reports = grade_lights(&t.scan.frames, &t.metrics, &t.errors, &ped, cfg);
    t.cal = calcheck::check(&t.scan.frames, &t.grouping, &t.cal_metrics, &t.cal_errors, &t.reports, &t.pair_noise, None, cache, cfg);
    t
}

/// Frames no header, file name or folder could type: decide from pixels.
fn classify_unknowns(scan: &mut ScanResult, cache: &mut Cache, cfg: &Config) {
    let idx: Vec<usize> = (0..scan.frames.len()).filter(|&i| scan.frames[i].class.frame_type == FrameType::Unknown).collect();
    for i in idx {
        let f = &scan.frames[i];
        let t = match cached(cache, f).and_then(|c| c.pixel_type) {
            Some(t) => t,
            None => match crate::read_for_measuring(&f.meta) {
                Ok(img) => {
                    let m = analyze_light(&img, cfg);
                    let t = m.pixel_type(cfg.camera.bias_level_adu as f32, f.meta.exptime);
                    store(cache, f, |c| c.pixel_type = Some(t));
                    t
                }
                Err(_) => continue,
            },
        };
        let c = &mut scan.frames[i].class;
        c.votes.push((TypeSource::Pixels, t));
        if t != FrameType::Unknown {
            c.frame_type = t;
            c.source = Some(TypeSource::Pixels);
        }
    }
}

/// Bias level for each light: the median level of its matched bias (else
/// dark) set, falling back to the configured level.
fn pedestals(g: &Grouping, cal: &HashMap<usize, CalMetrics>) -> HashMap<usize, f64> {
    let mut out = HashMap::new();
    for c in &g.light_calibration {
        let Some(&cg) = c.bias.first().or(c.darks.first()) else { continue };
        let lv: Vec<f64> = g.groups[cg].members.iter().filter_map(|i| cal.get(i)).map(|m| m.level_adu as f64).collect();
        if let Some(l) = median_f64(&lv) {
            for &m in &g.groups[c.group].members {
                out.insert(m, l);
            }
        }
    }
    out
}

/// Read noise of each raw bias set from up to 4 pairs of consecutive
/// frames (fixed pattern cancels in the difference). Cached per pair.
fn bias_pair_noise(frames: &[Frame], g: &Grouping, cache: &mut Cache, cfg: &Config) -> HashMap<usize, f64> {
    const PAIRS: usize = 4;
    let stamp = |f: &Frame| format!("{}|{}|{}", crate::cache::key(&f.meta.path), f.meta.file_size, f.meta.mtime_ns);
    let mut todo: Vec<(usize, usize, usize)> = Vec::new(); // group, a, b
    let mut have: HashMap<usize, Vec<f64>> = HashMap::new();
    for gr in g.groups.iter().filter(|g| g.kind == FrameType::Bias && !g.master) {
        let m: Vec<usize> = gr.members.iter().copied().filter(|i| !frames[*i].class.in_reject_dir).collect();
        for p in m.chunks_exact(2).take(PAIRS) {
            let (a, b) = (p[0], p[1]);
            match cached(cache, &frames[a]).and_then(|c| c.pair) {
                Some(pn) if pn.partner == stamp(&frames[b]) => have.entry(gr.id).or_default().push(pn.noise_adu as f64),
                _ => todo.push((gr.id, a, b)),
            }
        }
    }
    if !todo.is_empty() {
        let threads = cfg.analysis.thread_count().min(8);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("thread pool");
        let results: Vec<(usize, usize, usize, Option<f32>)> = pool.install(|| {
            todo.par_iter()
                .map(|&(gid, a, b)| {
                    let n = (|| {
                        let ia = crate::read_image(&frames[a].meta.path).ok()?;
                        let ib = crate::read_image(&frames[b].meta.path).ok()?;
                        crate::calib::pair_noise(&ia, &ib).map(|n| n * ia.adu_scale)
                    })();
                    (gid, a, b, n)
                })
                .collect()
        });
        for (gid, a, b, n) in results {
            if let Some(n) = n {
                have.entry(gid).or_default().push(n as f64);
                let partner = stamp(&frames[b]);
                store(cache, &frames[a], |c| c.pair = Some(PairNoise { partner, noise_adu: n }));
            }
        }
        let _ = cache.save();
    }
    have.into_iter().filter_map(|(k, v)| median_f64(&v).map(|m| (k, m))).collect()
}

/// Read frames in parallel, a bounded number at a time, handing each chunk's
/// results to `done` (which stores them and saves the cache).
fn run_chunked<T: Send>(
    frames: &[Frame],
    todo: &[usize],
    cfg: &Config,
    tick: &(dyn Fn() + Sync),
    work: &(dyn Fn(&Frame) -> anyhow::Result<T> + Sync),
    cancel: &AtomicBool,
    mut done: impl FnMut(Vec<(usize, Result<T, String>)>),
) {
    let threads = cfg.analysis.thread_count();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("thread pool");
    for chunk in todo.chunks(threads * 4) {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let results = pool.install(|| {
            chunk
                .par_iter()
                .map(|&i| {
                    let r = work(&frames[i]).map_err(|e| format!("{e:#}"));
                    tick();
                    (i, r)
                })
                .collect()
        });
        done(results);
    }
}

fn has_map(cache: &Cache, f: &Frame) -> bool {
    cache.get_map(&f.meta.path, f.meta.file_size, f.meta.mtime_ns).is_some()
}

type CalResults = (HashMap<usize, CalMetrics>, HashMap<usize, String>, usize);

/// Measure every bias, dark and flat (raw and master), from cache where possible.
pub fn analyse_calibration(
    frames: &[Frame],
    cache: &mut Cache,
    cfg: &Config,
    progress: &(dyn Fn(Stage, usize, usize) + Sync),
    cancel: &AtomicBool,
) -> CalResults {
    let ids: Vec<usize> = (0..frames.len())
        .filter(|&i| {
            let c = &frames[i].class;
            c.role != Role::Processed
                && matches!(c.frame_type, FrameType::Bias | FrameType::Dark | FrameType::DarkFlat | FrameType::Flat)
        })
        .collect();
    let mut metrics = HashMap::new();
    let mut todo = Vec::new();
    for &i in &ids {
        match cached(cache, &frames[i]).filter(|c| c.cal_v == crate::calib::CAL_VERSION).and_then(|c| c.cal) {
            Some(m) if has_map(cache, &frames[i]) => {
                metrics.insert(i, m);
            }
            _ => todo.push(i),
        }
    }
    let total = ids.len();
    let done = AtomicUsize::new(metrics.len());
    progress(Stage::Calibration, done.load(Ordering::Relaxed), total);
    let tick = || progress(Stage::Calibration, done.fetch_add(1, Ordering::Relaxed) + 1, total);
    let work = |f: &Frame| crate::read_image(&f.meta.path).map(|img| analyze_cal(&img));
    let mut errors = HashMap::new();
    let n = todo.len();
    run_chunked(frames, &todo, cfg, &tick, &work, cancel, |results| {
        for (i, r) in results {
            match r {
                Ok((m, map)) => {
                    let f = &frames[i];
                    cache.put_map(&f.meta.path, f.meta.file_size, f.meta.mtime_ns, &map);
                    store(cache, f, |c| {
                        c.cal = Some(m.clone());
                        c.cal_v = crate::calib::CAL_VERSION;
                    });
                    metrics.insert(i, m);
                }
                Err(e) => {
                    errors.insert(i, e);
                }
            }
        }
        let _ = cache.save();
    });
    (metrics, errors, n)
}

type LightResults = (HashMap<usize, LightMetrics>, HashMap<usize, String>, usize);

enum LightWork {
    Full(LightMetrics, Map, crate::cache::Thumb),
    /// Measured before maps or thumbnails existed: just those.
    Extras(Option<Map>, Option<crate::cache::Thumb>),
}

/// Measure every raw light, from cache where possible. Frames are processed
/// in chunks so the cache is saved as we go and memory stays bounded. Lights
/// measured before maps or thumbnails existed are only re-binned.
pub fn analyse_lights(
    frames: &[Frame],
    cache: &mut Cache,
    cfg: &Config,
    progress: &(dyn Fn(Stage, usize, usize) + Sync),
    cancel: &AtomicBool,
) -> LightResults {
    let lights: Vec<usize> = (0..frames.len())
        .filter(|&i| frames[i].class.frame_type == FrameType::Light && frames[i].class.role == Role::Raw)
        .collect();
    let total = lights.len();
    let mut metrics = HashMap::new();
    let mut todo = Vec::new();
    let mut full = std::collections::HashSet::new();
    for &i in &lights {
        match cached(cache, &frames[i]).and_then(|c| c.light) {
            Some(m) => {
                let f = &frames[i];
                if !has_map(cache, f) || !cache.has_thumb(&f.meta.path, f.meta.file_size, f.meta.mtime_ns) {
                    todo.push(i);
                }
                metrics.insert(i, m);
            }
            None => {
                todo.push(i);
                full.insert(frames[i].meta.path.clone());
            }
        }
    }
    let have_map: std::collections::HashSet<std::path::PathBuf> =
        todo.iter().filter(|&&i| has_map(cache, &frames[i])).map(|&i| frames[i].meta.path.clone()).collect();
    let done = AtomicUsize::new(total - todo.len());
    progress(Stage::Pixels, done.load(Ordering::Relaxed), total);
    let tick = || progress(Stage::Pixels, done.fetch_add(1, Ordering::Relaxed) + 1, total);
    let work = |f: &Frame| {
        let img = crate::read_for_measuring(&f.meta)?;
        Ok(if full.contains(&f.meta.path) {
            let (m, map) = analyze_light_map(&img, cfg);
            LightWork::Full(m, map, crate::stretch::thumbnail(&img))
        } else {
            let map = (!have_map.contains(&f.meta.path)).then(|| Map::bin(&img, MAP_BIN));
            LightWork::Extras(map, Some(crate::stretch::thumbnail(&img)))
        })
    };
    let analysed = full.len();
    let mut errors = HashMap::new();
    run_chunked(frames, &todo, cfg, &tick, &work, cancel, |results| {
        for (i, r) in results {
            let f = &frames[i];
            match r {
                Ok(LightWork::Full(m, map, thumb)) => {
                    cache.put_map(&f.meta.path, f.meta.file_size, f.meta.mtime_ns, &map);
                    cache.put_thumb(&f.meta.path, f.meta.file_size, f.meta.mtime_ns, &thumb);
                    store(cache, f, |c| c.light = Some(m.clone()));
                    metrics.insert(i, m);
                }
                Ok(LightWork::Extras(map, thumb)) => {
                    if let Some(map) = map {
                        cache.put_map(&f.meta.path, f.meta.file_size, f.meta.mtime_ns, &map);
                    }
                    if let Some(t) = thumb {
                        cache.put_thumb(&f.meta.path, f.meta.file_size, f.meta.mtime_ns, &t);
                    }
                }
                Err(e) => {
                    metrics.remove(&i);
                    errors.insert(i, e);
                }
            }
        }
        let _ = cache.save();
    });
    (metrics, errors, analysed)
}
