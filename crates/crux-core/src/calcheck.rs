//! Bias, dark and flat checks: each frame against its own set, then each set
//! (flat stability, flats against the lights, flat sets against each other).
//! The flat checks follow FlatChecker.js; see `calib`.

use crate::cache::Cache;
use crate::calib::{self, CalMetrics, Edges, FlatError, Map};
use crate::classify::{FrameType, Role};
use crate::config::Config;
use crate::grade::{Check, Flag, FrameReport, Grade};
use crate::group::{self, Group, Grouping};
use crate::library::Library;
use crate::scan::Frame;
use crate::stats::{median_f64, median_sigma_f64};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VsLights {
    pub error: FlatError,
    pub n_lights: usize,
    pub first_night: Option<NaiveDate>,
    pub last_night: Option<NaiveDate>,
    /// False when so much was masked that the map probably holds the target.
    pub reliable: bool,
    pub grade: Grade,
    /// Dust fingerprint correlation between this flat and the lights.
    #[serde(default)]
    pub dust: Option<f64>,
    /// The lights split in two (by field when there are several), each half
    /// stacked: how far apart the halves' illumination is, % peak-to-peak.
    /// Large when the stack still holds the target (few lights or fields).
    #[serde(default)]
    pub halves_diff: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetReport {
    pub group: usize,
    pub grade: Grade,
    pub flags: Vec<Flag>,
    pub notes: Vec<String>,
    /// Members measured.
    pub measured: usize,
    pub level_adu: Option<f64>,
    pub noise_adu: Option<f64>,
    /// Hot-pixel fraction (darks).
    pub hot: Option<f64>,
    /// Flats: worst change of an unsaturated frame from the set's median, %.
    pub worst_change: Option<f64>,
    pub n_saturated: usize,
    /// Flats: median level after the offset, fraction of full scale.
    pub flat_level: Option<f64>,
    /// Flats: each frame's change from the set's median flat, %.
    pub changes: BTreeMap<usize, f64>,
    pub vs_lights: Option<VsLights>,
    /// What was subtracted from raw flats.
    pub offset: Option<String>,
    /// Bias: read noise from frame-pair differences.
    #[serde(default)]
    pub read_noise_adu: Option<f64>,
    #[serde(default)]
    pub read_noise_e: Option<f64>,
    /// Bias and darks: median row and column banding, ADU.
    #[serde(default)]
    pub row_noise_adu: Option<f64>,
    #[serde(default)]
    pub col_noise_adu: Option<f64>,
}

impl SetReport {
    fn new(group: usize) -> Self {
        SetReport {
            group,
            grade: Grade::Good,
            flags: Vec::new(),
            notes: Vec::new(),
            measured: 0,
            level_adu: None,
            noise_adu: None,
            hot: None,
            worst_change: None,
            n_saturated: 0,
            flat_level: None,
            changes: BTreeMap::new(),
            vs_lights: None,
            offset: None,
            read_noise_adu: None,
            read_noise_e: None,
            row_noise_adu: None,
            col_noise_adu: None,
        }
    }

    fn flag(&mut self, check: Check, grade: Grade, reason: String) {
        self.grade = self.grade.max(grade);
        self.flags.push(Flag { check, grade, reason });
    }

    pub fn summary(&self) -> String {
        let mut f: Vec<&Flag> = self.flags.iter().filter(|f| f.grade != Grade::Good).collect();
        f.sort_by(|a, b| b.grade.cmp(&a.grade));
        f.iter().map(|f| f.reason.as_str()).collect::<Vec<_>>().join("; ")
    }
}

/// Two flat sets of the same filter compared.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatPair {
    pub a: usize,
    pub b: usize,
    /// Smoothed ratio, % peak-to-peak.
    pub spread: f64,
    /// Edges of a / b relative to the centre.
    pub edges: Edges,
    /// Dust fingerprint correlation.
    #[serde(default)]
    pub dust: Option<f64>,
    /// Both good, agree within `combine_pct` and share their dust.
    #[serde(default)]
    pub combinable: bool,
}

/// A flat set that could calibrate a night's lights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatOption {
    pub label: String,
    /// A flat group in this folder.
    pub group: Option<usize>,
    /// A set in the flat library.
    pub library_id: Option<String>,
    pub night: NaiveDate,
    /// Flat night minus light night.
    pub days: i64,
    pub grade: Grade,
    /// Dust fingerprint correlation with that night's lights.
    pub dust: Option<f64>,
}

/// Which flats a night's lights (one filter) should use, and whether their
/// dust confirms it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatChoice {
    pub filter: String,
    pub night: NaiveDate,
    pub binning: u32,
    pub light_groups: Vec<usize>,
    pub n_lights: usize,
    /// Lights stacked for the dust check.
    pub n_stacked: usize,
    pub chosen: Option<FlatOption>,
    /// Other usable sets, nearest first (up to 4).
    pub others: Vec<FlatOption>,
    pub grade: Grade,
    pub reason: String,
    /// Your choice (decisions sidecar), not crux's.
    #[serde(default)]
    pub pinned: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalCheck {
    pub reports: Vec<FrameReport>,
    pub sets: Vec<SetReport>,
    pub pairs: Vec<FlatPair>,
    /// What was subtracted from the lights for the flats-against-lights check.
    pub light_offsets: Vec<String>,
    /// Flats for each night and filter of lights.
    #[serde(default)]
    pub choices: Vec<FlatChoice>,
    /// Each flat set's median flat (median 1) and map size, for the library.
    #[serde(skip)]
    pub references: HashMap<usize, (Vec<f32>, usize, usize)>,
}

impl CalCheck {
    pub fn set(&self, group: usize) -> Option<&SetReport> {
        self.sets.iter().find(|s| s.group == group)
    }
}

fn is_cal(t: FrameType) -> bool {
    matches!(t, FrameType::Bias | FrameType::Dark | FrameType::DarkFlat | FrameType::Flat)
}

fn same_filter(a: &Option<String>, b: &Option<String>) -> bool {
    a.as_deref().unwrap_or("").eq_ignore_ascii_case(b.as_deref().unwrap_or(""))
}

struct Ctx<'a> {
    frames: &'a [Frame],
    g: &'a Grouping,
    metrics: &'a HashMap<usize, CalMetrics>,
    cfg: &'a Config,
    maps: HashMap<usize, Map>,
    /// Offset map (ADU) per bias/dark group: the master, or the median of the set.
    offsets: HashMap<usize, Vec<f32>>,
    /// Read noise per bias group from frame pairs, ADU.
    pair_noise: &'a HashMap<usize, f64>,
    library: Option<&'a Library>,
}

impl Ctx<'_> {
    fn offset_of(&self, gid: Option<&usize>) -> Option<(&[f32], String)> {
        let gid = *gid?;
        self.offsets.get(&gid).map(|m| (m.as_slice(), self.g.groups[gid].label.clone()))
    }

    /// Map minus offset, normalised. Also returns the level after the offset (ADU).
    fn flattened(&self, i: usize, offset: Option<&[f32]>) -> Option<(Vec<f32>, f32)> {
        let m = self.maps.get(&i)?;
        let scalar = self.cfg.camera.bias_level_adu as f32;
        let mut v: Vec<f32> = match offset {
            Some(o) if o.len() == m.data.len() => m.data.iter().zip(o).map(|(a, b)| a - b).collect(),
            _ => m.data.iter().map(|a| a - scalar).collect(),
        };
        let level = calib::normalise(&mut v)?;
        Some((v, level))
    }
}

/// Check every calibration frame and set. `light_reports` are the light
/// grades: rejected lights are left out of the illumination stack.
pub fn check(
    frames: &[Frame],
    g: &Grouping,
    metrics: &HashMap<usize, CalMetrics>,
    errors: &HashMap<usize, String>,
    light_reports: &[FrameReport],
    pair_noise: &HashMap<usize, f64>,
    library: Option<&Library>,
    cache: &Cache,
    cfg: &Config,
) -> CalCheck {
    let th = &cfg.thresholds.calibration;
    let timing = std::env::var_os("CRUX_TIMING").is_some();
    let tc = std::time::Instant::now();
    let lap = |what: &str| {
        if timing {
            eprintln!("calcheck: {what} at {:.2} s", tc.elapsed().as_secs_f64());
        }
    };
    let light_grade: HashMap<usize, Grade> = light_reports.iter().map(|r| (r.frame, r.grade)).collect();

    // Load every map needed: calibration members and usable lights.
    let mut maps = HashMap::new();
    for (i, f) in frames.iter().enumerate() {
        let want = (is_cal(f.class.frame_type) && metrics.contains_key(&i))
            || (f.class.frame_type == FrameType::Light
                && f.class.role == Role::Raw
                && !f.class.in_reject_dir
                && light_grade.get(&i).is_some_and(|&gr| gr != Grade::Reject));
        if want {
            if let Some(m) = cache.get_map(&f.meta.path, f.meta.file_size, f.meta.mtime_ns) {
                maps.insert(i, m);
            }
        }
    }
    lap("maps loaded");
    let mut ctx = Ctx { frames, g, metrics, cfg, maps, offsets: HashMap::new(), pair_noise, library };
    for gr in &g.groups {
        if !matches!(gr.kind, FrameType::Bias | FrameType::Dark | FrameType::DarkFlat) {
            continue;
        }
        let active: Vec<usize> = gr.members.iter().copied().filter(|i| !frames[*i].class.in_reject_dir).collect();
        let use_: &[usize] = if active.is_empty() { &gr.members } else { &active };
        let ms: Vec<&[f32]> = use_.iter().filter_map(|i| ctx.maps.get(i)).map(|m| m.data.as_slice()).collect();
        if !ms.is_empty() {
            ctx.offsets.insert(gr.id, calib::median_stack(&ms));
        }
    }

    let mut out = CalCheck::default();
    let mut refs: HashMap<usize, Reference> = HashMap::new();
    let mut reports: BTreeMap<usize, FrameReport> = BTreeMap::new();
    for gr in &g.groups {
        if !is_cal(gr.kind) {
            continue;
        }
        for &i in &gr.members {
            let mut r = FrameReport::new(i);
            if let Some(e) = errors.get(&i) {
                r.flag(Check::Read, Grade::Reject, format!("couldn't read: {e}"));
            }
            reports.insert(i, r);
        }
        let (set, reference) = match (gr.kind, gr.master) {
            (FrameType::Flat, false) => check_flat_set(&ctx, gr, &mut reports, th),
            (FrameType::Flat, true) => check_master_flat(&ctx, gr, &mut reports),
            (_, false) => (check_offset_set(&ctx, gr, &mut reports), None),
            (_, true) => (check_master_offset(&ctx, gr, &mut reports), None),
        };
        if let Some(r) = reference {
            refs.insert(gr.id, r);
        }
        out.sets.push(set);
    }
    // A master integrated from a set crux rejects (cloudy flats) is rejected too.
    let set_grade: HashMap<usize, Grade> = out.sets.iter().map(|s| (s.group, s.grade)).collect();
    for s in out.sets.iter_mut() {
        let gr = &g.groups[s.group];
        if gr.kind != FrameType::Flat || !gr.master || gr.retired {
            continue;
        }
        if let Some(src) = gr.built_from.filter(|src| set_grade.get(src) == Some(&Grade::Reject)) {
            let reason = format!("integrated from {}, which crux rejects", g.groups[src].label);
            if let Some(r) = reports.get_mut(&gr.members[0]) {
                r.flag(Check::Source, Grade::Reject, reason.clone());
            }
            s.flag(Check::Source, Grade::Reject, reason);
        }
    }

    lap("sets checked");
    let offsets = light_offsets(&ctx);
    let mut norm = LightNorm::default();
    flats_against_lights(&ctx, &refs, &offsets, &mut norm, &mut out, &mut reports, &light_grade);
    lap("flats against lights");
    out.pairs = compare_sets(&ctx, &refs, &out);
    lap("set pairs");
    let lib_consensus = library_consensus(&ctx, &refs, &mut out, &mut reports);
    lap("library consensus");
    out.choices = choose_flats(&ctx, &refs, &offsets, &mut norm, &out, &light_grade, &lib_consensus);
    lap("flats chosen");
    out.light_offsets = norm.used.iter().map(|(k, n)| format!("{k} ({n} lights)")).collect();
    out.reports = reports.into_values().collect();
    out.references = refs;
    out
}

/// Light frame → the bias or dark group its offset comes from.
fn light_offsets(ctx: &Ctx) -> HashMap<usize, Option<usize>> {
    let mut out = HashMap::new();
    for c in &ctx.g.light_calibration {
        let off = c.darks.first().or(c.bias.first()).copied();
        for &m in &ctx.g.groups[c.group].members {
            out.insert(m, off);
        }
    }
    out
}

/// Offset-subtracted, normalised light maps, made once each.
#[derive(Default)]
struct LightNorm {
    maps: HashMap<usize, Option<Vec<f32>>>,
    used: BTreeMap<String, usize>,
}

impl LightNorm {
    fn get(&mut self, ctx: &Ctx, offsets: &HashMap<usize, Option<usize>>, i: usize) -> Option<&[f32]> {
        if !self.maps.contains_key(&i) {
            let off = offsets.get(&i).copied().flatten();
            let o = ctx.offset_of(off.as_ref());
            let label = o.as_ref().map(|x| x.1.clone()).unwrap_or_else(|| format!("configured bias level ({:.0} ADU)", ctx.cfg.camera.bias_level_adu));
            *self.used.entry(label).or_default() += 1;
            let v = ctx.flattened(i, o.map(|x| x.0)).map(|x| x.0);
            self.maps.insert(i, v);
        }
        self.maps[&i].as_deref()
    }

    /// Median stack of the given lights (those that normalise).
    fn stack(&mut self, ctx: &Ctx, offsets: &HashMap<usize, Option<usize>>, ids: &[usize]) -> (Vec<f32>, usize) {
        for &i in ids {
            self.get(ctx, offsets, i);
        }
        let maps: Vec<&[f32]> = ids.iter().filter_map(|i| self.maps.get(i).and_then(|v| v.as_deref())).collect();
        let n = maps.len();
        if n == 0 {
            return (Vec::new(), 0);
        }
        (calib::median_stack(&maps), n)
    }
}

fn stat(v: &[f64]) -> Option<(f64, f64)> {
    median_sigma_f64(v)
}

/// Bias, dark or dark-flat set: each frame against the set's median.
fn check_offset_set(ctx: &Ctx, gr: &Group, reports: &mut BTreeMap<usize, FrameReport>) -> SetReport {
    let cfg = ctx.cfg;
    let th = &cfg.thresholds.calibration;
    let min_z = cfg.thresholds.min_z;
    let mut set = SetReport::new(gr.id);
    let ids: Vec<usize> = gr.members.iter().copied().filter(|i| ctx.metrics.contains_key(i)).collect();
    set.measured = ids.len();
    let dark = gr.kind != FrameType::Bias;
    let col = |f: &dyn Fn(&CalMetrics) -> f32| -> Vec<f64> { ids.iter().map(|i| f(&ctx.metrics[i]) as f64).collect() };
    let levels = col(&|m| m.level_adu);
    let noises = col(&|m| m.noise_adu);
    let hots = col(&|m| m.hot);
    let rows = col(&|m| m.row_noise_adu);
    let cols = col(&|m| m.col_noise_adu);
    // Each frame minus the set's median frame, smoothed: any structure.
    let median_map = ctx.offsets.get(&gr.id);
    let resid: Vec<Option<f64>> = ids
        .iter()
        .map(|i| {
            let (m, med) = (ctx.maps.get(i)?, median_map?);
            if m.data.len() != med.len() {
                return None;
            }
            let d: Vec<f32> = m.data.iter().zip(med).map(|(a, b)| a - b).collect();
            let sm = calib::gaussian_blur(&d, m.w, m.h, calib::DRIFT_SIGMA_PX / calib::MAP_BIN as f32);
            Some(calib::percentile_spread(&sm) as f64)
        })
        .collect();
    let resid_vals: Vec<f64> = resid.iter().flatten().copied().collect();
    set.row_noise_adu = median_f64(&rows);
    set.col_noise_adu = median_f64(&cols);
    if !dark {
        if let Some(&rn) = ctx.pair_noise.get(&gr.id) {
            set.read_noise_adu = Some(rn);
            let egain = ctx.frames[gr.members[0]].meta.egain.unwrap_or(cfg.camera.egain);
            set.read_noise_e = Some(rn * egain);
        }
        if let Some(l) = set.level_adu {
            let d = l - cfg.camera.bias_level_adu;
            if d.abs() > th.bias_expected_tolerance_adu {
                set.flag(
                    Check::Level,
                    Grade::Warn,
                    format!("level {l:.0} ADU, expected about {:.0} (camera.bias_level_adu) — check the offset setting", cfg.camera.bias_level_adu),
                );
            }
        }
    }
    set.level_adu = median_f64(&levels);
    set.noise_adu = median_f64(&noises);
    set.hot = median_f64(&hots);
    let relative = ids.len() >= cfg.thresholds.min_cohort;

    for (k, &i) in ids.iter().enumerate() {
        let f = &ctx.frames[i];
        let r = reports.get_mut(&i).expect("report");
        if relative {
            if let Some((med, sig)) = stat(&levels) {
                let d = levels[k] - med;
                let z = d / sig.max(0.5);
                if z.abs() >= min_z && d.abs() >= th.level_warn_adu {
                    let g = if d.abs() >= th.level_reject_adu { Grade::Reject } else { Grade::Warn };
                    let hint = match (dark, d > 0.0) {
                        (true, true) => " — light leak or a longer exposure?",
                        (_, false) => " — offset setting changed?",
                        (false, true) => " — offset setting changed, or light reaching the sensor?",
                    };
                    r.flag(Check::Level, g, format!("level {:.0} ADU, {:+.0} from the set's {:.0}{hint}", levels[k], d, med));
                }
            }
            if let Some((med, sig)) = stat(&noises) {
                let ratio = noises[k] / med.max(1e-6);
                let z = (noises[k] - med) / sig.max(0.02 * med).max(1e-6);
                if z >= min_z && ratio >= th.noise_warn {
                    let g = if ratio >= th.noise_reject { Grade::Reject } else { Grade::Warn };
                    r.flag(Check::Noise, g, format!("noise {:.1} ADU, {ratio:.2}× the set's — interference or a different gain?", noises[k]));
                }
            }
            if let (Some(x), Some((med, sig))) = (resid[k], stat(&resid_vals)) {
                let d = x - med;
                let z = d / sig.max(0.3);
                if z >= min_z && d >= th.structure_warn_adu {
                    let g = if d >= th.structure_reject_adu { Grade::Reject } else { Grade::Warn };
                    r.flag(Check::Structure, g, format!("differs from the set's median frame by {x:.1} ADU (smoothed, set typical {med:.1}) — light leak?"));
                }
            }
            for (name, v) in [("row", &rows), ("column", &cols)] {
                if let Some((med, sig)) = stat(v) {
                    let ratio = v[k] / med.max(1e-6);
                    let z = (v[k] - med) / sig.max(0.05 * med).max(1e-6);
                    if med > 0.0 && z >= min_z && ratio >= th.banding_warn {
                        r.flag(Check::Banding, Grade::Warn, format!("{name} banding {:.2} ADU, {ratio:.1}× the set's", v[k]));
                    }
                }
            }
            if dark {
                if let Some((med, sig)) = stat(&hots) {
                    let ratio = hots[k] / med.max(1e-9);
                    let z = (hots[k] - med) / sig.max(0.05 * med).max(1e-9);
                    if med > 0.0 && z >= min_z && ratio >= th.hot_warn {
                        r.flag(
                            Check::HotPixels,
                            Grade::Warn,
                            format!("{:.3}% hot pixels, {ratio:.1}× the set's — sensor warmer than the rest?", 100.0 * hots[k]),
                        );
                    }
                }
            }
        }
        if let (Some(t), Some(sp)) = (f.meta.ccd_temp, f.meta.set_temp) {
            let d = t - sp;
            if d.abs() >= th.temp_warn_c {
                let g = if d.abs() >= th.temp_reject_c { Grade::Reject } else { Grade::Warn };
                r.flag(Check::Temperature, g, format!("sensor at {t:.1} °C, set point {sp:.0} °C"));
            }
        }
        if !dark && f.meta.exptime.is_some_and(|e| e > th.bias_max_exposure) {
            r.flag(Check::Exposure, Grade::Warn, format!("exposure {:.2} s is long for a bias", f.meta.exptime.unwrap_or(0.0)));
        }
    }
    if !relative && !ids.is_empty() {
        set.notes.push(format!("only {} frames: compared with each other only from {}", ids.len(), cfg.thresholds.min_cohort));
    }
    summarise_members(&mut set, gr, reports);
    set
}

/// Set grade from its members: reject when most are rejected, warn when any are.
fn summarise_members(set: &mut SetReport, gr: &Group, reports: &BTreeMap<usize, FrameReport>) {
    let n = gr.members.len();
    let rej = gr.members.iter().filter(|i| reports.get(i).is_some_and(|r| r.grade == Grade::Reject)).count();
    let warn = gr.members.iter().filter(|i| reports.get(i).is_some_and(|r| r.grade == Grade::Warn)).count();
    if rej * 2 > n {
        set.flag(Check::Frames, Grade::Reject, format!("{rej} of {n} frames rejected"));
    } else if rej > 0 {
        set.flag(Check::Frames, Grade::Warn, format!("{rej} of {n} frames rejected — move them out before integrating"));
    } else if warn > 0 {
        set.notes.push(format!("{warn} of {n} frames have warnings"));
    }
}

/// Master bias or dark: compared with the raw set it was built from.
fn check_master_offset(ctx: &Ctx, gr: &Group, reports: &mut BTreeMap<usize, FrameReport>) -> SetReport {
    let th = &ctx.cfg.thresholds.calibration;
    let mut set = SetReport::new(gr.id);
    let i = gr.members[0];
    let Some(m) = ctx.metrics.get(&i) else { return set };
    set.measured = 1;
    set.level_adu = Some(m.level_adu as f64);
    set.noise_adu = Some(m.noise_adu as f64);
    set.hot = Some(m.hot as f64);
    if let Some(src) = gr.built_from {
        let lv: Vec<f64> = ctx.g.groups[src].members.iter().filter_map(|j| ctx.metrics.get(j)).map(|m| m.level_adu as f64).collect();
        if let Some(med) = median_f64(&lv) {
            let d = m.level_adu as f64 - med;
            set.notes.push(format!("built from {} (level {:+.1} ADU from it)", ctx.g.groups[src].label, d));
            if d.abs() >= th.level_warn_adu {
                let r = reports.get_mut(&i).expect("report");
                r.flag(Check::Level, Grade::Warn, format!("level {:.0} ADU, {:+.0} from the raw set it was built from", m.level_adu, d));
                set.flag(Check::Level, Grade::Warn, format!("level differs from its raw set by {d:+.0} ADU"));
            }
        }
    }
    set
}

/// Raw flat set: saturation, level, and FlatChecker's stability check.
fn check_flat_set(
    ctx: &Ctx,
    gr: &Group,
    reports: &mut BTreeMap<usize, FrameReport>,
    th: &crate::config::Calibration,
) -> (SetReport, Option<Reference>) {
    let mut set = SetReport::new(gr.id);
    let cal = ctx
        .g
        .flat_calibration
        .iter()
        .find(|c| c.group == gr.id)
        .cloned()
        .unwrap_or_else(|| group::match_flat(gr, &ctx.g.groups, ctx.cfg));
    let off_gid = cal.dark_flats.first().or(cal.bias.first()).or(cal.darks.first());
    let offset = ctx.offset_of(off_gid);
    set.offset = Some(match &offset {
        Some((_, l)) => l.clone(),
        None => format!("configured bias level ({:.0} ADU): no bias or dark flats matched", ctx.cfg.camera.bias_level_adu),
    });
    let mut ids = Vec::new();
    let mut flats = Vec::new();
    let mut levels = Vec::new();
    for &i in &gr.members {
        let Some(m) = ctx.metrics.get(&i) else { continue };
        let Some((v, level)) = ctx.flattened(i, offset.as_ref().map(|o| o.0)) else {
            reports.get_mut(&i).expect("report").flag(Check::Level, Grade::Reject, "no signal above the offset".into());
            continue;
        };
        ids.push((i, m.saturated as f64 <= th.flat_saturated_fraction));
        flats.push(v);
        levels.push(level as f64 / 65535.0);
    }
    set.measured = ids.len();
    if ids.is_empty() {
        return (set, None);
    }
    let (w, h) = ctx.maps.get(&ids[0].0).map(|m| (m.w, m.h)).unwrap_or((0, 0));
    let refs: Vec<&[f32]> = flats.iter().map(|v| v.as_slice()).collect();
    let use_: Vec<bool> = ids.iter().map(|x| x.1).collect();
    let (reference, drift) = calib::flat_drift(&refs, w, h, &use_);
    set.n_saturated = use_.iter().filter(|u| !**u).count();
    let good_levels: Vec<f64> = levels.iter().zip(&use_).filter(|(_, u)| **u).map(|(l, _)| *l).collect();
    set.flat_level = median_f64(&good_levels).or(median_f64(&levels));
    set.worst_change = drift.iter().zip(&use_).filter(|(_, u)| **u).map(|(d, _)| *d).max_by(|a, b| a.total_cmp(b));

    for (k, &(i, ok)) in ids.iter().enumerate() {
        set.changes.insert(i, drift[k]);
        let r = reports.get_mut(&i).expect("report");
        let m = &ctx.metrics[&i];
        if !ok {
            r.flag(Check::Saturated, Grade::Reject, format!("{:.2}% of pixels clipped — saturated flat", 100.0 * m.saturated));
            continue;
        }
        if levels[k] < th.flat_min_level {
            r.flag(Check::Level, Grade::Warn, format!("dim — {:.0}% of full scale", 100.0 * levels[k]));
        } else if levels[k] > th.flat_max_level {
            r.flag(Check::Level, Grade::Warn, format!("bright — {:.0}% of full scale, close to saturation", 100.0 * levels[k]));
        }
        let d = drift[k];
        if d > th.flat_change_warn {
            let g = if d > th.flat_change_reject { Grade::Reject } else { Grade::Warn };
            r.flag(Check::FlatChange, g, format!("changed {d:.1}% from the set's median flat — cloud or shifting twilight"));
        }
    }
    if set.n_saturated > 0 {
        set.notes.push(format!(
            "{} saturated flat{}: with sky flats, let the exposure shorten as the sky brightens",
            set.n_saturated,
            if set.n_saturated == 1 { "" } else { "s" }
        ));
    }
    if ids.len() < 3 {
        set.notes.push("fewer than 3 flats: the stability check means little".into());
    }
    summarise_members(&mut set, gr, reports);
    (set, Some((reference, w, h)))
}

/// A set's reference flat (median 1) and its map size.
type Reference = (Vec<f32>, usize, usize);

/// Master flat: already calibrated, so just normalised.
fn check_master_flat(ctx: &Ctx, gr: &Group, reports: &mut BTreeMap<usize, FrameReport>) -> (SetReport, Option<Reference>) {
    let mut set = SetReport::new(gr.id);
    let i = gr.members[0];
    let Some(m) = ctx.maps.get(&i) else { return (set, None) };
    let mut v = m.data.clone();
    if calib::normalise(&mut v).is_none() {
        return (set, None);
    }
    set.measured = 1;
    if let Some(src) = gr.built_from {
        let label = &ctx.g.groups[src].label;
        if gr.retired {
            let reason = format!("integrated from {label}, which is in a reject folder");
            reports.get_mut(&i).expect("report").flag(Check::Source, Grade::Reject, reason.clone());
            set.flag(Check::Source, Grade::Reject, reason);
        } else {
            set.notes.push(format!("built from {label}"));
        }
    }
    (set, Some((v, m.w, m.h)))
}

/// FlatChecker's second check for every flat set and master flat.
fn flats_against_lights(
    ctx: &Ctx,
    refs: &HashMap<usize, Reference>,
    offsets: &HashMap<usize, Option<usize>>,
    norm: &mut LightNorm,
    out: &mut CalCheck,
    reports: &mut BTreeMap<usize, FrameReport>,
    light_grade: &HashMap<usize, Grade>,
) {
    let cfg = ctx.cfg;
    let th = &cfg.thresholds.calibration;
    let mut stacks: HashMap<Vec<usize>, Vec<f32>> = HashMap::new();
    let mut halves: HashMap<Vec<usize>, Option<f64>> = HashMap::new();

    for set in out.sets.iter_mut() {
        let gr = &ctx.g.groups[set.group];
        if gr.kind != FrameType::Flat {
            continue;
        }
        let Some((flat, w, h)) = refs.get(&gr.id) else { continue };
        let Some(night) = gr.night.or_else(|| gr.first.map(|t| crate::classify::night_of(t))) else { continue };
        // Usable lights of the same filter, binning and size, nearest first.
        let mut cand: Vec<(i64, usize)> = ctx
            .frames
            .iter()
            .enumerate()
            .filter(|(i, f)| {
                f.class.frame_type == FrameType::Light
                    && f.class.role == Role::Raw
                    && !f.class.in_reject_dir
                    && light_grade.get(i).is_some_and(|&g| g != Grade::Reject)
                    && ctx.maps.contains_key(i)
                    && same_filter(&f.meta.filter, &gr.filter)
                    && f.meta.binning() == gr.binning.unwrap_or(1)
                    && (f.meta.width, f.meta.height) == (gr.width, gr.height)
            })
            .filter_map(|(i, f)| {
                let d = (f.class.night? - night).num_days().abs();
                (d <= th.flat_lights_days).then_some((d, i))
            })
            .collect();
        cand.sort();
        cand.truncate(th.flat_max_lights);
        if cand.len() < th.flat_min_lights {
            set.notes.push(format!(
                "not checked against lights: {} usable lights within {} nights (needs {})",
                cand.len(),
                th.flat_lights_days,
                th.flat_min_lights
            ));
            continue;
        }
        let mut key: Vec<usize> = cand.iter().map(|c| c.1).collect();
        key.sort();
        if !stacks.contains_key(&key) {
            let (st, n) = norm.stack(ctx, offsets, &key);
            if n < th.flat_min_lights || st.len() != flat.len() {
                continue;
            }
            stacks.insert(key.clone(), st);
        }
        let illum = &stacks[&key];
        let err = calib::flat_against_lights(illum, flat, *w, *h);
        let dust = calib::correlate(&calib::dust_fingerprint(flat, *w, *h), &calib::dust_of_lights(illum, *w, *h), *w, *h);
        if !halves.contains_key(&key) {
            let (a, b) = split_halves(ctx, &key);
            let d = if a.len() >= th.flat_min_lights && b.len() >= th.flat_min_lights {
                let (sa, _) = norm.stack(ctx, offsets, &a);
                let (sb, _) = norm.stack(ctx, offsets, &b);
                (sa.len() == sb.len() && !sa.is_empty()).then(|| calib::flat_against_lights(&sa, &sb, *w, *h).spread)
            } else {
                None
            };
            halves.insert(key.clone(), d);
        }
        let halves_diff = halves[&key];
        // Each half's error is about half their difference: give the flat
        // that much benefit of the doubt, and don't judge at all when the
        // halves disagree by more than the warn level twice over.
        let doubt = halves_diff.map(|d| d / 2.0).unwrap_or(0.0);
        let too_alike = doubt > th.flat_error_warn;
        let reliable = err.masked <= th.flat_mask_warn && !too_alike;
        let grade = if !reliable {
            Grade::Good
        } else if err.spread > th.flat_error_reject + doubt && halves_diff.is_some() {
            Grade::Reject
        } else if err.spread > th.flat_error_reject + doubt {
            // Too few lights to check the stack's consistency: warn only.
            set.notes.push(format!("only {} lights: too few to confirm the error map, so it can only warn", key.len()));
            Grade::Warn
        } else if err.spread > th.flat_error_warn + doubt {
            Grade::Warn
        } else {
            Grade::Good
        };
        if too_alike {
            set.notes.push(format!(
                "flats-against-lights check unreliable: two halves of the lights differ by {:.1}% (too few lights or fields to remove the target)",
                halves_diff.unwrap_or(0.0)
            ));
        }
        let nights: Vec<NaiveDate> = key.iter().filter_map(|&i| ctx.frames[i].class.night).collect();
        let shape = err.edges.describe(1.0);
        let reason = format!(
            "leaves {:.1}% error in the lights{}",
            err.spread,
            if shape.is_empty() { String::new() } else { format!(" ({shape})") }
        );
        if err.masked > th.flat_mask_warn {
            set.notes.push(format!(
                "flats-against-lights check unreliable: {:.0}% of the frame masked as target",
                100.0 * err.masked
            ));
        }
        if grade != Grade::Good {
            set.flag(Check::FlatMatch, grade, reason.clone());
            for &m in &gr.members {
                if let Some(r) = reports.get_mut(&m) {
                    r.flag(Check::FlatMatch, grade, format!("set {reason}"));
                }
            }
        }
        set.vs_lights = Some(VsLights {
            n_lights: key.len(),
            first_night: nights.iter().min().copied(),
            last_night: nights.iter().max().copied(),
            reliable,
            grade,
            dust,
            halves_diff,
            error: err,
        });
    }
}

/// Every pair of flat sets with the same filter, binning and size.
fn compare_sets(ctx: &Ctx, refs: &HashMap<usize, Reference>, out: &CalCheck) -> Vec<FlatPair> {
    let mut by_key: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for s in &out.sets {
        let gr = &ctx.g.groups[s.group];
        if gr.kind == FrameType::Flat && refs.contains_key(&gr.id) {
            let k = format!("{}|{:?}|{}x{}", gr.filter.clone().unwrap_or_default().to_uppercase(), gr.binning, gr.width, gr.height);
            by_key.entry(k).or_default().push(gr.id);
        }
    }
    let mut pairs = Vec::new();
    for ids in by_key.values_mut() {
        ids.sort_by_key(|&i| (ctx.g.groups[i].first, ctx.g.groups[i].master));
        ids.truncate(8);
        for a in 0..ids.len() {
            for b in a + 1..ids.len() {
                let (fa, w, h) = &refs[&ids[a]];
                let (fb, _, _) = &refs[&ids[b]];
                if fa.len() != fb.len() {
                    continue;
                }
                let (spread, edges) = calib::compare_flats(fa, fb, *w, *h);
                let dust = calib::correlate(&calib::dust_fingerprint(fa, *w, *h), &calib::dust_fingerprint(fb, *w, *h), *w, *h);
                let th = &ctx.cfg.thresholds.calibration;
                let (ga, gb) = (&ctx.g.groups[ids[a]], &ctx.g.groups[ids[b]]);
                let good = |id: usize| out.set(id).is_some_and(|s| s.grade != Grade::Reject) && !ctx.g.groups[id].retired;
                let combinable = !ga.master
                    && !gb.master
                    && good(ids[a])
                    && good(ids[b])
                    && spread <= th.combine_pct
                    && dust.is_some_and(|d| d >= th.dust_same_train);
                pairs.push(FlatPair { a: ids[a], b: ids[b], spread, edges, dust, combinable });
            }
        }
    }
    pairs
}

/// For each night and filter of lights: the nearest good flat set (this
/// folder or the library), confirmed by correlating its dust fingerprint
/// with the night's lights. A mismatch means the optical train changed.
fn choose_flats(
    ctx: &Ctx,
    refs: &HashMap<usize, Reference>,
    offsets: &HashMap<usize, Option<usize>>,
    norm: &mut LightNorm,
    out: &CalCheck,
    light_grade: &HashMap<usize, Grade>,
    lib_consensus: &HashMap<String, crate::library::Consensus>,
) -> Vec<FlatChoice> {
    let th = &ctx.cfg.thresholds.calibration;
    // Light groups by (filter, binning, size, night).
    let mut cohorts: BTreeMap<(String, u32, u32, u32, NaiveDate), Vec<usize>> = BTreeMap::new();
    for gr in ctx.g.groups.iter().filter(|g| g.kind == FrameType::Light && !g.master) {
        let Some(night) = gr.night else { continue };
        let f = gr.filter.clone().unwrap_or_default().to_uppercase();
        cohorts.entry((f, gr.binning.unwrap_or(1), gr.width, gr.height, night)).or_default().push(gr.id);
    }
    // Local candidates: flat sets with a reference flat.
    struct Cand {
        opt: FlatOption,
        fp: Vec<f32>,
        w: usize,
        h: usize,
        filter: String,
        binning: u32,
        size: (u32, u32),
        local: bool,
    }
    let mut cands: Vec<Cand> = Vec::new();
    let mut local_ids = std::collections::HashSet::new();
    for s in &out.sets {
        let gr = &ctx.g.groups[s.group];
        if gr.kind != FrameType::Flat {
            continue;
        }
        let (Some((flat, w, h)), Some(night)) = (refs.get(&gr.id), gr.night) else { continue };
        let filter = gr.filter.clone().unwrap_or_default().to_uppercase();
        let master_file = gr.master.then(|| ctx.frames[gr.members[0]].meta.file_name().to_string());
        local_ids.insert(crate::library::set_id(&filter, night, gr.binning.unwrap_or(1), gr.width, gr.height, master_file.as_deref(), gr.retired));
        if gr.retired || s.grade == Grade::Reject {
            continue;
        }
        cands.push(Cand {
            opt: FlatOption { label: gr.label.clone(), group: Some(gr.id), library_id: None, night, days: 0, grade: s.grade, dust: None },
            fp: calib::dust_fingerprint(flat, *w, *h),
            w: *w,
            h: *h,
            filter,
            binning: gr.binning.unwrap_or(1),
            size: (gr.width, gr.height),
            local: true,
        });
    }
    if let Some(lib) = ctx.library {
        for s in lib
            .sets
            .iter()
            .filter(|s| s.usable() && !local_ids.contains(&s.id))
            .filter(|s| lib_consensus.get(&s.id).is_none_or(|c| c.grade != Grade::Reject))
        {
            let Some(fp) = lib.fingerprint(&s.id) else { continue };
            cands.push(Cand {
                opt: FlatOption {
                    label: format!("library {}", s.label()),
                    group: None,
                    library_id: Some(s.id.clone()),
                    night: s.night,
                    days: 0,
                    grade: s.grade,
                    dust: None,
                },
                fp: fp.data,
                w: fp.w,
                h: fp.h,
                filter: s.filter.to_uppercase(),
                binning: s.binning,
                size: (s.width, s.height),
                local: false,
            });
        }
    }

    let mut choices = Vec::new();
    for ((filter, binning, w, h, night), groups) in cohorts {
        let lights: Vec<usize> = groups.iter().flat_map(|&g| ctx.g.groups[g].members.iter().copied()).collect();
        let usable: Vec<usize> = lights
            .iter()
            .copied()
            .filter(|i| !ctx.frames[*i].class.in_reject_dir && light_grade.get(i).is_some_and(|&g| g != Grade::Reject) && ctx.maps.contains_key(i))
            .collect();
        let (stack, n_stacked) = if usable.len() >= th.flat_min_lights { norm.stack(ctx, offsets, &usable) } else { (Vec::new(), 0) };
        let mut opts: Vec<(i64, bool, FlatOption)> = Vec::new();
        // The lights' dust pattern depends only on the map size: work it
        // out once, not once per candidate (it was most of a cached open).
        let mut lights_dust: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        for c in cands.iter().filter(|c| c.filter == filter && c.binning == binning && c.size == (w, h)) {
            let mut o = c.opt.clone();
            o.days = (o.night - night).num_days();
            if n_stacked >= th.flat_min_lights && stack.len() == c.fp.len() {
                let ld = lights_dust.entry((c.w, c.h)).or_insert_with(|| calib::dust_of_lights(&stack, c.w, c.h));
                o.dust = calib::correlate(&c.fp, ld, c.w, c.h);
            }
            opts.push((o.days.abs(), !c.local, o));
        }
        // A nebula filling the field swamps the dust pattern in the lights.
        let target_frac = (n_stacked >= th.flat_min_lights)
            .then(|| cands.iter().find(|c| c.fp.len() == stack.len()).map(|c| calib::smooth_masked(&stack, c.w, c.h, calib::sigma_bins(calib::ERROR_SIGMA_PX, c.w)).1))
            .flatten();
        let target_fills = target_frac.is_some_and(|f| f > th.flat_mask_warn);
        opts.sort_by(|a, b| (a.0, a.1, a.2.grade).cmp(&(b.0, b.1, b.2.grade)));
        let mut opts: Vec<FlatOption> = opts.into_iter().map(|x| x.2).collect();
        let chosen = (!opts.is_empty()).then(|| opts.remove(0));
        let better = opts.iter().filter(|o| o.dust.is_some_and(|d| d >= th.dust_match)).max_by(|a, b| a.dust.partial_cmp(&b.dust).unwrap_or(std::cmp::Ordering::Equal));
        let (grade, reason) = match &chosen {
            None => (Grade::Reject, format!("no usable {filter} flats at bin {binning}")),
            Some(c) => {
                let caveat = if c.grade == Grade::Warn { " — the set has warnings, see its frames" } else { "" };
                let when = match c.days {
                    0 => format!("same night{caveat}"),
                    d if d > 0 => format!("{d} night{} later{caveat}", if d == 1 { "" } else { "s" }),
                    d => format!("{} night{} earlier{caveat}", -d, if d == -1 { "" } else { "s" }),
                };
                match c.dust {
                    None => (Grade::Good, format!("{} ({when}); dust not checked: {} usable lights", c.label, n_stacked.max(usable.len()))),
                    Some(d) if d >= th.dust_match => (Grade::Good, format!("{} ({when}); dust matches the lights ({d:.2})", c.label)),
                    Some(d) if d >= th.dust_mismatch => (Grade::Good, format!("{} ({when}); dust matches the lights only weakly ({d:.2})", c.label)),
                    Some(d) if target_fills => (Grade::Good, format!("{} ({when}); dust not checked: the target fills {:.0}% of the lights (dust {d:.2})", c.label, 100.0 * target_frac.unwrap_or(0.0))),
                    Some(d) => (
                        Grade::Warn,
                        format!(
                            "{} ({when}); dust doesn't match the lights ({d:.2}) — the optical train has probably changed since these flats{}",
                            c.label,
                            better.map(|b| format!("; {} matches ({:.2})", b.label, b.dust.unwrap_or(0.0))).unwrap_or_default()
                        ),
                    ),
                }
            }
        };
        choices.push(FlatChoice {
            filter,
            night,
            binning,
            light_groups: groups,
            n_lights: lights.len(),
            n_stacked,
            chosen,
            others: opts,
            grade,
            reason,
            pinned: false,
        });
    }
    choices
}

/// With a library: each local flat set against the library's usable sets of
/// its filter that share its dust. A set far from all of them had cloud or a
/// twilight gradient in every frame, which its own stability check can't
/// see. Returns the library's own consensus, for flat matching.
fn library_consensus(
    ctx: &Ctx,
    refs: &HashMap<usize, Reference>,
    out: &mut CalCheck,
    reports: &mut BTreeMap<usize, FrameReport>,
) -> HashMap<String, crate::library::Consensus> {
    let Some(lib) = ctx.library else { return HashMap::new() };
    let th = &ctx.cfg.thresholds.calibration;
    // Library maps with their dust fingerprints, each worked out once.
    let lib_refs: Vec<(&crate::library::LibrarySet, Map, Vec<f32>)> = {
        use rayon::prelude::*;
        let all = lib.references();
        lib.sets
            .par_iter()
            .filter(|s| s.usable())
            .filter_map(|s| {
                let r = all.get(&s.id)?.clone();
                let fp = calib::dust_fingerprint(&r.data, r.w, r.h);
                Some((s, r, fp))
            })
            .collect()
    };
    for set in out.sets.iter_mut() {
        let gr = &ctx.g.groups[set.group];
        let (Some((flat, w, h)), Some(night)) = (refs.get(&gr.id), gr.night) else { continue };
        let filter = gr.filter.clone().unwrap_or_default().to_uppercase();
        let master_file = gr.master.then(|| ctx.frames[gr.members[0]].meta.file_name().to_string());
        let own = crate::library::set_id(&filter, night, gr.binning.unwrap_or(1), gr.width, gr.height, master_file.as_deref(), gr.retired);
        let fp = calib::dust_fingerprint(flat, *w, *h);
        let diffs: Vec<f64> = lib_refs
            .iter()
            .filter(|(s, r, _)| s.id != own && s.filter == filter && (s.binning, s.width, s.height) == (gr.binning.unwrap_or(1), gr.width, gr.height) && r.data.len() == flat.len())
            // Same night, other copy (raw vs master): not independent.
            .filter(|(s, _, _)| s.night != night)
            .filter(|(_, _, rf)| calib::correlate(&fp, rf, *w, *h).is_some_and(|d| d >= th.dust_same_train))
            .map(|(_, r, _)| calib::compare_flats(flat, &r.data, *w, *h).0)
            .collect();
        if let Some(c) = crate::library::consensus_of(&diffs, ctx.cfg) {
            if c.grade != Grade::Good {
                let reason = c.reason();
                set.flag(Check::Library, c.grade, reason.clone());
                for &m in &gr.members {
                    if let Some(r) = reports.get_mut(&m) {
                        r.flag(Check::Library, c.grade, format!("set {reason}"));
                    }
                }
            } else {
                set.notes.push(format!("within a median {:.1}% of {} library sets with the same dust", c.median, c.n));
            }
        }
    }
    let pairs = lib.pairs(ctx.cfg);
    lib.consensus(&pairs, ctx.cfg)
}

/// Split lights in two for a consistency check: by field (object) when
/// there are several, so each half sees different targets; else alternate
/// frames.
fn split_halves(ctx: &Ctx, ids: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut by_obj: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for &i in ids {
        by_obj.entry(ctx.frames[i].meta.object.clone().unwrap_or_default().to_lowercase()).or_default().push(i);
    }
    let (mut a, mut b) = (Vec::new(), Vec::new());
    if by_obj.len() >= 2 {
        let mut groups: Vec<Vec<usize>> = by_obj.into_values().collect();
        groups.sort_by_key(|g| std::cmp::Reverse(g.len()));
        for g in groups {
            if a.len() <= b.len() {
                a.extend(g);
            } else {
                b.extend(g);
            }
        }
    } else {
        for (k, &i) in ids.iter().enumerate() {
            if k % 2 == 0 {
                a.push(i);
            } else {
                b.push(i);
            }
        }
    }
    (a, b)
}
