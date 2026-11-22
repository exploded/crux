//! Process a folder: plan which lights, masters and reference to use, then
//! calibrate, fix hot pixels and register every light into the workspace.
//!
//! Inputs are read where they are; everything written goes under
//! `<workspace>\<folder name>\` (masters, registered, optionally
//! calibrated) with a manifest, `crux-process.json`, so a rerun skips
//! frames whose inputs and settings haven't changed.

use super::calibrate::{CalParams, CalStats, Calibrator, Master};
use super::combine;
use super::integrate::{self, Normalise, Rejection};
use super::register::{self, Registration};
use super::resample;
use crate::classify::{FrameType, Role};
use crate::config::Config;
use crate::decisions::Decisions;
use crate::grade::Grade;
use crate::header::Header;
use crate::image::Image;
use crate::pipeline::Triage;
use anyhow::{Context, Result};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const MANIFEST: &str = "crux-process.json";
/// Bump when outputs change for the same inputs and settings.
/// 2: masters of 3–7 frames are medians (2026-11-19).
pub const PROCESS_VERSION: u32 = 2;

/// Bump when masters change for the same inputs and settings.
pub const INTEGRATION_VERSION: u32 = 4;

/// Where a master comes from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Source {
    /// An existing master file.
    File { path: PathBuf },
    /// Built by crux from raw frames, minus `offset` (bias or dark flats) for flats.
    Build { frame_type: FrameType, files: Vec<PathBuf>, offset: Option<Box<Source>>, out: PathBuf },
}

impl Source {
    pub fn label(&self) -> String {
        match self {
            Source::File { path } => path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            Source::Build { frame_type, files, .. } => format!("{} raw {}, stacked by crux", files.len(), plural(frame_type.name(), files.len())),
        }
    }
}

/// Masters for one filter and night of lights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalSetup {
    pub label: String,
    pub filter: String,
    pub night: Option<NaiveDate>,
    pub exptime: Option<f64>,
    pub dark: Option<Source>,
    pub bias: Option<Source>,
    pub flat: Option<Source>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightItem {
    pub frame: usize,
    pub path: PathBuf,
    pub filter: String,
    pub night: Option<NaiveDate>,
    pub grade: Grade,
    /// Index into `Plan::cals`.
    pub cal: usize,
    pub reg: Registration,
    pub registered: PathBuf,
    pub calibrated: Option<PathBuf>,
    #[serde(default)]
    pub exptime: f64,
    #[serde(default)]
    pub fwhm: Option<f32>,
    /// Raw one-shot-colour light: calibrated as it is, then debayered and
    /// registered as three channels (R, G, B), each stacked like a filter.
    #[serde(default)]
    pub bayer: Option<crate::cfa::Bayer>,
}

/// A colour channel's registered frame: `<folder>\R\<name>` beside where a
/// mono frame would go.
fn channel_path(p: &Path, c: u8) -> PathBuf {
    let name = p.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    p.parent().unwrap_or(Path::new("")).join(crate::cfa::CHANNELS[c as usize]).join(name)
}

impl LightItem {
    /// The registered frame a stack of `channel` reads (mono lights have one).
    pub fn output(&self, channel: Option<u8>) -> PathBuf {
        match (self.bayer, channel) {
            (Some(_), Some(c)) => channel_path(&self.registered, c),
            _ => self.registered.clone(),
        }
    }

    /// Every registered frame this light writes.
    pub fn outputs(&self) -> Vec<PathBuf> {
        match self.bayer {
            Some(_) => (0..3).map(|c| channel_path(&self.registered, c)).collect(),
            None => vec![self.registered.clone()],
        }
    }

    fn outputs_exist(&self) -> bool {
        self.outputs().iter().all(|p| p.is_file())
    }
}

/// One master light: a field's lights of one filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackPlan {
    pub object: String,
    pub filter: String,
    /// Index into `Plan::fields`.
    pub field: usize,
    /// Indices into that field's lights.
    pub lights: Vec<usize>,
    /// The light (index into `lights`) everything is normalised to: the
    /// field's reference if it has this filter, else the best of them.
    pub reference: usize,
    /// The reference's unsaturated stars in the field's geometry.
    pub stars: Vec<(f32, f32)>,
    pub out: PathBuf,
    /// For one-shot-colour lights: the channel (0 R, 1 G, 2 B) this stacks.
    #[serde(default)]
    pub channel: Option<u8>,
    /// A second channel of every light in the same stack: OIII through a
    /// dual-band filter is in both green and blue (`cfa::dual_band`).
    #[serde(default)]
    pub also: Option<u8>,
}

impl StackPlan {
    /// The registered frame of light `i` (an index into the field's lights).
    pub fn input(&self, field: &FieldPlan, i: usize) -> PathBuf {
        field.lights[i].output(self.channel)
    }

    /// Every input: (index into `lights`, registered frame), the `also`
    /// channel's frames after the main ones.
    pub fn inputs(&self, field: &FieldPlan) -> Vec<(usize, PathBuf)> {
        let mut v: Vec<(usize, PathBuf)> = self.lights.iter().enumerate().map(|(k, &i)| (k, field.lights[i].output(self.channel))).collect();
        if let Some(c) = self.also {
            v.extend(self.lights.iter().enumerate().map(|(k, &i)| (k, field.lights[i].output(Some(c)))));
        }
        v
    }
}

/// One field (object or mosaic panel): a reference and its lights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldPlan {
    pub object: String,
    pub reference: usize,
    pub reference_path: PathBuf,
    pub lights: Vec<LightItem>,
    /// Lights left out: (path, why).
    pub skipped: Vec<(PathBuf, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Settings {
    pub workspace: PathBuf,
    pub include_warn: bool,
    pub keep_calibrated: bool,
    pub cal: CalParams,
    pub clamp: Option<f32>,
    /// File-name fragment of the reference to force (all fields).
    pub reference: Option<String>,
    /// Integrate each field and filter.
    #[serde(default = "yes")]
    pub integrate: bool,
    #[serde(default)]
    pub combine: combine::Settings,
    /// Also write a FITS copy beside each integrated master (`[process] fits`).
    #[serde(default)]
    pub fits: bool,
}

fn yes() -> bool {
    true
}

/// Parse the config's integration words.
pub fn combine_settings(p: &crate::config::Process) -> combine::Settings {
    combine::Settings {
        normalisation: if p.normalisation.eq_ignore_ascii_case("global") { combine::Normalisation::Global } else { combine::Normalisation::Local },
        weighting: match p.weights.to_ascii_lowercase().as_str() {
            "equal" => combine::Weighting::Equal,
            "exposure" => combine::Weighting::Exposure,
            _ => combine::Weighting::Snr,
        },
        rejection: match p.rejection.to_ascii_lowercase().as_str() {
            "esd" => Some(Rejection::Esd { fraction: 0.3, alpha: 0.05 }),
            "winsorized" | "winsorised" | "winsor" => Some(Rejection::Winsorized { low: 4.0, high: 3.0 }),
            "sigma" => Some(Rejection::Sigma { low: 4.0, high: 3.0 }),
            "percentile" => Some(Rejection::Percentile { low: 0.2, high: 0.1 }),
            "none" => Some(Rejection::None),
            _ => None,
        },
        ..Default::default()
    }
}

impl Settings {
    pub fn from_config(cfg: &Config) -> Settings {
        let p = &cfg.process;
        Settings {
            workspace: p.workspace.clone(),
            include_warn: p.include_warn,
            keep_calibrated: p.keep_calibrated,
            cal: CalParams {
                hot_sigma: (p.hot_sigma > 0.0).then_some(p.hot_sigma),
                pedestal_limit: (p.pedestal_limit > 0.0).then_some(p.pedestal_limit),
                ..Default::default()
            },
            clamp: (p.lanczos_clamp > 0.0).then_some(p.lanczos_clamp),
            reference: None,
            integrate: p.integrate,
            combine: combine_settings(p),
            fits: p.fits,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub root: PathBuf,
    pub out: PathBuf,
    pub settings: Settings,
    pub cals: Vec<CalSetup>,
    pub fields: Vec<FieldPlan>,
    /// Master lights to integrate (empty when integration is off).
    #[serde(default)]
    pub stacks: Vec<StackPlan>,
    /// Plain-English notes about choices and assumptions.
    pub notes: Vec<String>,
}

impl Plan {
    pub fn n_lights(&self) -> usize {
        self.fields.iter().map(|f| f.lights.len()).sum()
    }
    pub fn n_skipped(&self) -> usize {
        self.fields.iter().map(|f| f.skipped.len()).sum()
    }
}

/// "same night", "1 night earlier", "3 nights later".
fn days_text(d: i64) -> String {
    match d {
        0 => "same night".into(),
        _ => format!("{} night{} {}", d.abs(), if d.abs() == 1 { "" } else { "s" }, if d < 0 { "earlier" } else { "later" }),
    }
}

pub fn safe_name(s: &str) -> String {
    s.chars().map(|c| if c.is_alphanumeric() || " -_.".contains(c) { c } else { '_' }).collect::<String>().trim().to_string()
}

fn stem(p: &Path) -> String {
    p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()
}

/// The workspace folder for a data folder.
pub fn out_dir(root: &Path, workspace: &Path) -> PathBuf {
    workspace.join(safe_name(&root.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "folder".into())))
}

/// The fingerprint ranking's best usable dark or bias for a light group's
/// family, from any folder in the library.
fn library_pick(fp: Option<&crate::fingerprint::Fingerprints>, group: usize, kind: FrameType, out: &Path, notes: &mut Vec<String>) -> Option<Source> {
    let fp = fp?;
    let fam = fp.families.iter().find(|f| f.kind == FrameType::Light && f.groups.contains(&group))?;
    let list = if kind == FrameType::Dark { &fam.darks } else { &fam.bias };
    let r = list.first().filter(|r| r.usable)?;
    let c = fp.candidates.get(r.cand)?;
    let word = if kind == FrameType::Dark { "dark" } else { "bias" };
    notes.push(format!("{word}: {} from {} (the library's best match for these lights)", c.label, c.folder.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()));
    if c.master {
        return c.files.first().map(|p| Source::File { path: p.clone() });
    }
    let name = format!("master{}_{}.xisf", if kind == FrameType::Dark { "Dark" } else { "Bias" }, safe_name(&c.label));
    Some(Source::Build { frame_type: kind, files: c.files.clone(), offset: None, out: out.join("masters").join(name) })
}

/// Flats with no bias or dark flats of their own (library flats, or a
/// folder without them): the fingerprint ranking's best bias for the
/// lights they calibrate (same sensor, binning and size), from any folder.
fn fp_flat_bias(fp: Option<&crate::fingerprint::Fingerprints>, group: Option<usize>, out: &Path) -> Option<(Source, String)> {
    let (fp, group) = (fp?, group?);
    let fam = fp.families.iter().find(|f| f.kind == FrameType::Light && f.groups.contains(&group))?;
    let c = fp.candidates.get(fam.bias.first().filter(|r| r.usable)?.cand)?;
    let note = format!("offset with {} from {}", c.label, c.place());
    let src = if c.master {
        Source::File { path: c.files.first()?.clone() }
    } else {
        Source::Build { frame_type: FrameType::Bias, files: c.files.clone(), offset: None, out: out.join("masters").join(format!("masterBias_{}.xisf", safe_name(&c.label))) }
    };
    Some((src, note))
}

/// No bias or dark flats anywhere: the shortest dark in the library can
/// stand in when dark current is negligible (the 2012 STL-11000M: 559.6 ADU
/// at 300 s, 561.7 at 1200 s, so ~0.5 ADU of current against 18,500 ADU
/// flats, where leaving the 559 ADU bias in would be a 3% offset).
fn dark_as_flat_offset(fp: Option<&crate::fingerprint::Fingerprints>, w: u32, h: u32, out: &Path) -> Option<(Source, String)> {
    let fp = fp?;
    let darks: Vec<&crate::fingerprint::Candidate> = fp.candidates.iter().filter(|c| c.kind == FrameType::Dark && c.width == w && c.height == h && c.exptime.is_some() && c.level_adu.is_some() && c.error.is_none()).collect();
    let short = darks.iter().min_by(|a, b| a.exptime.unwrap().total_cmp(&b.exptime.unwrap()))?;
    let long = darks.iter().max_by(|a, b| a.exptime.unwrap().total_cmp(&b.exptime.unwrap()))?;
    let (es, el) = (short.exptime?, long.exptime?);
    if el < es * 1.5 {
        return None;
    }
    let rate = (long.level_adu? - short.level_adu?) / (el - es);
    let current = rate.max(0.0) * es;
    if current > 2.0 {
        return None;
    }
    let note = format!("flats offset with {} (no bias here; dark current {current:.1} ADU in {})", short.label, crate::group::fmt_exp(es));
    let src = if short.master {
        Source::File { path: short.files.first()?.clone() }
    } else {
        Source::Build { frame_type: FrameType::Dark, files: short.files.clone(), offset: None, out: out.join("masters").join(format!("masterDark_{}.xisf", safe_name(&short.label))) }
    };
    Some((src, note))
}

/// Decide what to process. Pure: reads nothing beyond the triage.
///
/// `fp` (the dark/bias fingerprint ranking across the library) supplies a
/// dark or bias from another folder when this one has none.
pub fn plan(t: &Triage, root: &Path, decisions: Option<&Decisions>, settings: &Settings, catalog: Option<&crate::catalog::Catalog>, fp: Option<&crate::fingerprint::Fingerprints>) -> Result<Plan> {
    let out = out_dir(root, &settings.workspace);
    let rk = crate::paths::key(root);
    let ok = crate::paths::key(&out);
    anyhow::ensure!(!ok.starts_with(&rk), "the workspace {} is inside the data folder {}: choose another workspace", out.display(), root.display());
    let frames = &t.scan.frames;
    let groups = &t.grouping.groups;
    let mut notes = Vec::new();
    let group_of = |i: usize| groups.iter().find(|g| g.kind == FrameType::Light && g.members.contains(&i));
    let cal_grade: HashMap<usize, Grade> = t.cal.reports.iter().map(|r| (r.frame, r.grade)).collect();
    // Flat sets chosen for some lights (by you, or by crux as better than no
    // flat): the set's verdict against the lights, copied to every frame,
    // was overridden, so only a frame's own problems leave it out. Without
    // this, choosing a rejected set left no frames. Sets crux chose for
    // being usable carry no rejecting verdict, so nothing changes for them.
    let pinned_flats: HashSet<usize> = t.cal.choices.iter().filter_map(|c| c.chosen.as_ref()?.group).collect();
    let own_reject: HashSet<usize> = t.cal.reports.iter().filter(|r| r.flags.iter().any(|f| f.grade == Grade::Reject && f.check != crate::grade::Check::FlatMatch)).map(|r| r.frame).collect();
    // A master or raw set → Source.
    let source_of_group = |gid: usize, offset: Option<Source>, out_name: String| -> Option<Source> {
        let g = groups.get(gid)?;
        if g.master {
            return Some(Source::File { path: frames[*g.members.first()?].meta.path.clone() });
        }
        let rejected = |i: usize| if pinned_flats.contains(&gid) { own_reject.contains(&i) } else { cal_grade.get(&i) == Some(&Grade::Reject) };
        let files: Vec<PathBuf> = g
            .members
            .iter()
            .filter(|&&i| !frames[i].class.in_reject_dir && !rejected(i))
            .map(|&i| frames[i].meta.path.clone())
            .collect();
        if files.is_empty() {
            return None;
        }
        Some(Source::Build { frame_type: g.kind, files, offset: offset.map(Box::new), out: out.join("masters").join(out_name) })
    };
    // Bias or dark flats for a raw flat group.
    let flat_offset = |gid: usize| -> Option<Source> {
        let fc = t.grouping.flat_calibration.iter().find(|c| c.group == gid)?;
        if let Some(&d) = fc.dark_flats.first() {
            return source_of_group(d, None, format!("masterDarkFlat_{}.xisf", safe_name(&groups[d].label)));
        }
        let &b = fc.bias.first()?;
        source_of_group(b, None, format!("masterBias_{}.xisf", safe_name(&groups[b].label)))
    };
    // Your dark/bias pick for a family of lights, as a source.
    let picked = |kind: &str, i: usize| -> Option<Source> {
        let d = decisions?;
        let m = &frames[i].meta;
        let k = crate::decisions::offset_key_of(root, kind, m.exptime, m.xbin.unwrap_or(1), m.width, m.height);
        let p = PathBuf::from(d.offsets.get(&k)?);
        let f = frames.iter().position(|f| crate::paths::key(&f.meta.path) == crate::paths::key(&p));
        match f {
            Some(fi) if frames[fi].class.role == Role::Master => Some(Source::File { path: p }),
            Some(fi) => {
                let g = groups.iter().find(|g| g.members.contains(&fi))?;
                source_of_group(g.id, None, format!("master{}_{}.xisf", if kind == "dark" { "Dark" } else { "Bias" }, safe_name(&g.label)))
            }
            None if p.is_file() => Some(Source::File { path: p }),
            None => None,
        }
    };

    // Lights to process. One field per object, spellings merged.
    let field_names = crate::targets::field_names(t.reports.iter().filter_map(|r| frames[r.frame].meta.object.as_deref()));
    let mut by_field: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut skipped_by_field: BTreeMap<String, Vec<(PathBuf, String)>> = BTreeMap::new();
    for r in &t.reports {
        let i = r.frame;
        let f = &frames[i];
        let object = f.meta.object.as_ref().map(|o| field_names.get(o).cloned().unwrap_or_else(|| o.clone())).unwrap_or_else(|| "unknown".into());
        // Filed as a reject (BAD folder or BAD_ name): processed only when
        // crux grades it good or you pinned Keep (both leave grade Good).
        let why = if f.class.in_reject_dir && r.grade != Grade::Good {
            Some(if f.class.reject_name_only { "named as a reject" } else { "in a reject folder" })
        } else if r.grade == Grade::Reject {
            Some("graded reject")
        } else if r.grade == Grade::Warn && !settings.include_warn {
            Some("graded warn (warn frames excluded)")
        } else if !t.metrics.contains_key(&i) {
            Some("not measured")
        } else {
            None
        };
        match why {
            Some(w) => skipped_by_field.entry(object).or_default().push((f.meta.path.clone(), w.to_string())),
            None => by_field.entry(object).or_default().push(i),
        }
    }

    // Calibration setups per filter × night (× exposure).
    let mut cals: Vec<CalSetup> = Vec::new();
    let mut cal_index: HashMap<(String, Option<NaiveDate>, String), usize> = HashMap::new();
    let mut cal_for = |i: usize, notes: &mut Vec<String>| -> usize {
        let m = &frames[i].meta;
        let filter = m.filter.clone().unwrap_or_default().to_uppercase();
        let night = frames[i].class.night;
        let exp = format!("{:.2}", m.exptime.unwrap_or(0.0));
        let k = (filter.clone(), night, exp);
        if let Some(&c) = cal_index.get(&k) {
            return c;
        }
        let mut cn = Vec::new();
        let lc = group_of(i).and_then(|g| t.grouping.light_calibration.iter().find(|c| c.group == g.id));
        let dark = picked("dark", i)
            .inspect(|_| cn.push("dark: your choice".into()))
            .or_else(|| {
                let &d = lc?.darks.first()?;
                source_of_group(d, None, format!("masterDark_{}.xisf", safe_name(&groups[d].label)))
            })
            .or_else(|| library_pick(fp, group_of(i)?.id, FrameType::Dark, &out, &mut cn));
        let bias = picked("bias", i)
            .or_else(|| {
                let &b = lc?.bias.first()?;
                source_of_group(b, None, format!("masterBias_{}.xisf", safe_name(&groups[b].label)))
            })
            .or_else(|| library_pick(fp, group_of(i)?.id, FrameType::Bias, &out, &mut cn));
        if dark.is_none() {
            cn.push("no dark: bias only".into());
        }
        let choice = night.and_then(|n| t.cal.choices.iter().find(|c| c.filter.eq_ignore_ascii_case(&filter) && c.night == n));
        let flat = choice.and_then(|c| c.chosen.as_ref()).and_then(|o| {
            if let Some(mut g) = o.group {
                // A master whose raw flats are here: restack them without the
                // frames crux rejects (WBPP integrated them all).
                if let Some(src) = groups.get(g).filter(|gr| gr.master).and_then(|gr| gr.built_from) {
                    let usable = !groups[src].retired && t.cal.set(src).is_some_and(|s| s.grade != Grade::Reject);
                    if usable {
                        let n_rej = groups[src].members.iter().filter(|i| cal_grade.get(i) == Some(&Grade::Reject)).count();
                        cn.push(format!("flats restacked from the raw {}{}", groups[src].label, if n_rej > 0 { format!(" without the {n_rej} frames crux rejects") } else { String::new() }));
                        g = src;
                    }
                }
                let mut off = if groups.get(g).is_some_and(|g| !g.master) { flat_offset(g) } else { None };
                if off.is_none() && groups.get(g).is_some_and(|g| !g.master) {
                    let bias = fp_flat_bias(fp, group_of(i).map(|lg| lg.id), &out).map(|(s, n)| (s, format!("flats {n}")));
                    if let Some((src, note)) = bias.or_else(|| dark_as_flat_offset(fp, m.width, m.height, &out)) {
                        cn.push(note);
                        off = Some(src);
                    }
                }
                source_of_group(g, off, format!("masterFlat_{}_{}.xisf", filter, o.night))
            } else if let (Some(id), Some(lib)) = (&o.library_id, &t.library) {
                let set = lib.sets.iter().find(|s| &s.id == id)?;
                if set.master {
                    let p = set.files.first()?;
                    let from = p.parent().and_then(|d| d.file_name()).map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                    cn.push(format!("library {} from {from} ({})", set.label(), days_text(o.days)));
                    Some(Source::File { path: p.clone() })
                } else {
                    // Library flats: offset with this folder's flat bias.
                    let mut off = t
                        .grouping
                        .flat_calibration
                        .iter()
                        .find_map(|c| c.bias.first().copied())
                        .or_else(|| lc.and_then(|c| c.bias.first().copied()))
                        .and_then(|b| source_of_group(b, None, format!("masterBias_{}.xisf", safe_name(&groups[b].label))));
                    let mut how = if off.is_some() { "offset with this folder's bias".to_string() } else { "not offset: no bias or dark flats".to_string() };
                    if off.is_none() {
                        if let Some((src, note)) = fp_flat_bias(fp, group_of(i).map(|lg| lg.id), &out) {
                            how = note;
                            off = Some(src);
                        } else if let Some((src, note)) = dark_as_flat_offset(fp, m.width, m.height, &out) {
                            how = note;
                            off = Some(src);
                        }
                    }
                    cn.push(format!("library flats {} ({}), {how}", set.label(), days_text(o.days)));
                    Some(Source::Build { frame_type: FrameType::Flat, files: set.files.clone(), offset: off.map(Box::new), out: out.join("masters").join(format!("masterFlat_{}_{}_library.xisf", filter, o.night)) })
                }
            } else {
                None
            }
        });
        match (choice, &flat) {
            (Some(c), Some(_)) => {
                if c.chosen.as_ref().is_some_and(|o| o.days != 0 && o.library_id.is_none()) {
                    cn.push(format!("flats from {} ({})", days_text(c.chosen.as_ref().unwrap().days), c.chosen.as_ref().unwrap().label));
                }
                if c.pinned {
                    cn.push("flats: your choice".into());
                }
            }
            _ => {
                cn.push("no usable flats: not flat-fielded".into());
                notes.push(format!("{filter} lights of {}: no usable flats", night.map(|n| n.to_string()).unwrap_or_default()));
            }
        }
        let label = format!("{} · {} · {}", filter, night.map(|n| n.to_string()).unwrap_or_else(|| "?".into()), crate::group::fmt_exp(m.exptime.unwrap_or(0.0)));
        cals.push(CalSetup { label, filter: filter.clone(), night, exptime: m.exptime, dark, bias, flat, notes: cn });
        cal_index.insert(k, cals.len() - 1);
        cals.len() - 1
    };

    let mut fields = Vec::new();
    let mut learned = crate::solve::Learned::default();
    let tpl = std::time::Instant::now();
    for (object, ids) in &by_field {
        let score = |i: usize| {
            let m = &t.metrics[&i];
            register::reference_score(m.n_stars, m.fwhm, m.ecc, t.report_for(i).and_then(|r| r.transparency.as_ref()).map(|x| x.value))
        };
        let forced = settings.reference.as_ref().and_then(|f| ids.iter().copied().find(|&i| frames[i].meta.file_name().contains(f.as_str())));
        let Some(r0) = forced.or_else(|| ids.iter().copied().max_by(|&a, &b| score(a).total_cmp(&score(b)).then(b.cmp(&a)))) else { continue };
        let rm = &t.metrics[&r0];
        let (w, h) = (rm.width as f64, rm.height as f64);
        let mut ref_solution: Option<Option<crate::solve::Solution>> = None;
        let mut lights = Vec::new();
        let mut skipped = skipped_by_field.remove(object).unwrap_or_default();
        // Star matching against the reference, every light at once (about
        // 30 ms each; 213 IC 434 lights took 6.8 s one after another).
        let mut matched: HashMap<usize, Option<Registration>> = {
            use rayon::prelude::*;
            ids.par_iter()
                .filter(|&&i| i != r0 && (t.metrics[&i].width, t.metrics[&i].height) == (rm.width, rm.height))
                .map(|&i| (i, register::register(&rm.stars, &t.metrics[&i].stars, w, h)))
                .collect()
        };
        for &i in ids {
            let f = &frames[i];
            let m = &t.metrics[&i];
            if (m.width, m.height) != (rm.width, rm.height) {
                skipped.push((f.meta.path.clone(), format!("{}×{} (the reference is {}×{})", m.width, m.height, rm.width, rm.height)));
                continue;
            }
            let mut reg = if i == r0 {
                Some(Registration { to_ref: resample::Homography::identity(), matched: m.stars.len(), rms: 0.0, shift: (0.0, 0.0), rotation: 0.0, scale: 1.0, flipped: false })
            } else {
                matched.remove(&i).flatten()
            };
            // Star matching failed: seed from plate solutions, then refine.
            if reg.is_none() {
                if let Some(cat) = catalog {
                    // The reference's measured scale hints the rest.
                    let solve = |k: usize, learned: &mut crate::solve::Learned| {
                        let mm = &t.metrics[&k];
                        crate::solve::Hint::from_meta(&frames[k].meta).and_then(|hh| learned.solve(cat, &mm.stars, mm.width as f64, mm.height as f64, &hh).ok().flatten())
                    };
                    if ref_solution.is_none() {
                        ref_solution = Some(solve(r0, &mut learned));
                    }
                    if let (Some(Some(rs)), Some(fs)) = (&ref_solution, solve(i, &mut learned)) {
                        if let Some(seed) = register::seed_from_solutions(rs, &fs, w, h) {
                            reg = register::refine(&rm.stars, &m.stars, seed).map(|(hm, matched, rms)| {
                                let (dx, dy, rotation, scale) = hm.summary();
                                Registration { to_ref: hm, matched, rms, shift: (dx, dy), rotation, scale, flipped: rotation.abs() > 90.0 }
                            });
                            if reg.is_some() {
                                notes.push(format!("{}: registered from plate solutions (star matching alone failed)", f.meta.file_name()));
                            }
                        }
                    }
                }
            }
            let Some(reg) = reg else {
                skipped.push((f.meta.path.clone(), "couldn't match its stars to the reference".into()));
                continue;
            };
            let filter = f.meta.filter.clone().unwrap_or_default().to_uppercase();
            let cal = cal_for(i, &mut notes);
            let sub = |kind: &str, suffix: &str| out.join(kind).join(safe_name(object)).join(if filter.is_empty() { "nofilter".to_string() } else { filter.clone() }).join(format!("{}{suffix}.xisf", stem(&f.meta.path)));
            lights.push(LightItem {
                frame: i,
                path: f.meta.path.clone(),
                filter: filter.clone(),
                night: f.class.night,
                grade: t.report_for(i).map(|r| r.grade).unwrap_or(Grade::Good),
                cal,
                reg,
                registered: sub("registered", "_r"),
                calibrated: settings.keep_calibrated.then(|| sub("calibrated", "_c")),
                exptime: f.meta.exptime.unwrap_or(0.0),
                fwhm: m.fwhm,
                bayer: f.meta.bayer(),
            });
        }
        unique_outputs(&mut lights, root);
        lights.sort_by(|a, b| a.cal.cmp(&b.cal).then(a.path.cmp(&b.path)));
        fields.push(FieldPlan { object: object.clone(), reference: r0, reference_path: frames[r0].meta.path.clone(), lights, skipped });
    }
    for (object, skipped) in skipped_by_field {
        fields.push(FieldPlan { object, reference: usize::MAX, reference_path: PathBuf::new(), lights: Vec::new(), skipped });
    }
    if std::env::var_os("CRUX_TIMING").is_some() {
        eprintln!("plan: lights registered in {:.2} s", tpl.elapsed().as_secs_f64());
    }
    let stacks = if settings.integrate { plan_stacks(t, &fields, &out) } else { Vec::new() };
    Ok(Plan { root: root.to_path_buf(), out, settings: settings.clone(), cals, fields, stacks, notes })
}

/// "dark" → "darks", "bias" → "bias" (crux's own usage), "dark flat" → "dark flats";
/// one frame keeps the singular.
fn plural(name: &str, n: usize) -> String {
    if name == "bias" || n == 1 {
        name.to_string()
    } else {
        format!("{name}s")
    }
}

/// One master per field and filter. The reference for normalisation is
/// the field's reference when it has the filter, else the filter's best
/// light; its stars are mapped into the field's geometry.
fn plan_stacks(t: &Triage, fields: &[FieldPlan], out: &Path) -> Vec<StackPlan> {
    let mut stacks = Vec::new();
    for (fi, f) in fields.iter().enumerate() {
        // Colour lights go into one stack per channel; through a dual-band
        // filter, one per line (H or S from red, O from green and blue).
        // Keyed by the filter in front too, so two filters never share a
        // stack; names clashing between filters get the filter added.
        let mut by_filter: BTreeMap<(String, String, Option<u8>, Option<u8>), Vec<usize>> = BTreeMap::new();
        for (k, l) in f.lights.iter().enumerate() {
            match (l.bayer, crate::cfa::dual_band(&l.filter)) {
                (Some(_), Some([red, gb])) => {
                    by_filter.entry((red.to_string(), l.filter.clone(), Some(0), None)).or_default().push(k);
                    by_filter.entry((gb.to_string(), l.filter.clone(), Some(1), Some(2))).or_default().push(k);
                }
                (Some(_), None) => (0..3u8).for_each(|c| by_filter.entry((crate::cfa::CHANNELS[c as usize].to_string(), l.filter.clone(), Some(c), None)).or_default().push(k)),
                (None, _) => by_filter.entry((l.filter.clone(), l.filter.clone(), None, None)).or_default().push(k),
            }
        }
        let mut sources: HashMap<String, BTreeSet<String>> = HashMap::new();
        for (name, front, _, _) in by_filter.keys() {
            sources.entry(name.clone()).or_default().insert(front.clone());
        }
        // The filter with most frames keeps the plain names.
        let mut frames_by_front: HashMap<&str, usize> = HashMap::new();
        for ((_, front, _, _), ids) in &by_filter {
            *frames_by_front.entry(front.as_str()).or_default() += ids.len();
        }
        for ((name, front, channel, also), ids) in by_filter.clone() {
            let clash = sources[&name].len() > 1 && sources[&name].iter().any(|o| o != &front && frames_by_front[o.as_str()] >= frames_by_front[front.as_str()]);
            let filter = if clash { format!("{name} {front}") } else { name };
            let score = |k: usize| {
                let l = &f.lights[k];
                t.metrics.get(&l.frame).map(|m| register::reference_score(m.n_stars, m.fwhm, m.ecc, t.report_for(l.frame).and_then(|r| r.transparency.as_ref()).map(|x| x.value))).unwrap_or(0.0)
            };
            let best = ids.iter().position(|&k| f.lights[k].frame == f.reference).unwrap_or_else(|| (0..ids.len()).max_by(|&a, &b| score(ids[a]).total_cmp(&score(ids[b]))).unwrap_or(0));
            let rl = &f.lights[ids[best]];
            let stars: Vec<(f32, f32)> = t
                .metrics
                .get(&rl.frame)
                .map(|m| {
                    m.stars
                        .iter()
                        .filter(|s| s.flags == 0)
                        .map(|s| {
                            let (x, y) = rl.reg.to_ref.apply(s.x as f64, s.y as f64);
                            (x as f32, y as f32)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let name = if filter.is_empty() { "nofilter".to_string() } else { filter.clone() };
            let out = out.join("integrated").join(safe_name(&f.object)).join(format!("masterLight_{}_{}.xisf", safe_name(&f.object), name));
            stacks.push(StackPlan { object: f.object.clone(), filter: filter.clone(), field: fi, lights: ids, reference: best, stars, out, channel, also });
        }
    }
    stacks
}

/// Lights with the same file name in different subfolders (MaxIm reuses
/// `NGC6188-002H.fit` night after night) would write the same output:
/// those get their subfolder as a prefix (`DAY_2_NGC6188-002H_r.xisf`).
fn unique_outputs(lights: &mut [LightItem], root: &Path) {
    let mut count: HashMap<String, usize> = HashMap::new();
    for l in lights.iter() {
        *count.entry(crate::paths::key(&l.registered)).or_default() += 1;
    }
    for l in lights.iter_mut() {
        if count[&crate::paths::key(&l.registered)] < 2 {
            continue;
        }
        let rel = l.path.parent().and_then(|p| p.strip_prefix(root).ok()).map(|p| p.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect::<Vec<_>>().join("_")).unwrap_or_default();
        if rel.is_empty() {
            continue;
        }
        let prefix = |p: &Path| p.with_file_name(format!("{}_{}", safe_name(&rel), p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()));
        l.registered = prefix(&l.registered);
        l.calibrated = l.calibrated.as_deref().map(prefix);
    }
}

/// Progress events from `run`.
#[derive(Debug, Clone)]
pub enum Event {
    /// Building or loading a master.
    Master { label: String },
    /// A light finished (or was up to date).
    Light { done: usize, total: usize, path: PathBuf, result: LightResult },
    /// Integrating a master: which (1-based) of how many, and progress.
    Integrate { index: usize, total: usize, label: String, step: combine::Step },
    /// A master finished (or was up to date: `report` is the earlier one).
    Integrated { index: usize, total: usize, label: String, report: Option<Box<combine::Report>>, error: Option<String> },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LightResult {
    pub skipped_up_to_date: bool,
    pub cal: CalStats,
    pub seconds: f64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunReport {
    pub processed: usize,
    pub up_to_date: usize,
    pub failed: usize,
    pub masters_built: Vec<String>,
    pub seconds: f64,
    pub cancelled: bool,
    /// Master lights integrated, and those already up to date.
    #[serde(default)]
    pub integrated: usize,
    #[serde(default)]
    pub integrations_up_to_date: usize,
    #[serde(default)]
    pub integration_failed: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    /// Output path key → input signature.
    products: BTreeMap<String, String>,
    /// Data folder this workspace belongs to.
    root: PathBuf,
}

fn file_sig(p: &Path) -> String {
    match std::fs::metadata(p) {
        Ok(m) => {
            let t = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos()).unwrap_or(0);
            format!("{}:{}:{t}", p.display(), m.len())
        }
        Err(_) => format!("{}:missing", p.display()),
    }
}

fn source_sig(s: &Source) -> String {
    match s {
        Source::File { path } => file_sig(path),
        Source::Build { files, offset, .. } => {
            let mut v: Vec<String> = files.iter().map(|f| file_sig(f)).collect();
            if let Some(o) = offset {
                v.push(source_sig(o));
            }
            format!("build[{}]", v.join(","))
        }
    }
}

fn load_manifest(out: &Path) -> Manifest {
    std::fs::read(out.join(MANIFEST)).ok().and_then(|b| serde_json::from_slice::<Manifest>(&b).ok()).filter(|m| m.version == PROCESS_VERSION).unwrap_or_default()
}

fn save_manifest(out: &Path, m: &Manifest) -> Result<()> {
    std::fs::create_dir_all(out)?;
    crate::paths::write_atomic(&out.join(MANIFEST), &serde_json::to_vec_pretty(m)?)?;
    Ok(())
}

/// Build (or reuse) a master; returns it loaded.
fn materialise(src: &Source, label: &str, manifest: &mut Manifest, loaded: &mut HashMap<PathBuf, Arc<Image>>, built: &mut Vec<String>, progress: &(dyn Fn(Event) + Sync)) -> Result<Master> {
    let (path, exptime) = match src {
        Source::File { path } => (path.clone(), None),
        Source::Build { frame_type, files, offset, out } => {
            let sig = format!("v{PROCESS_VERSION}|{}", source_sig(src));
            let key = crate::paths::key(out);
            if !(out.is_file() && manifest.products.get(&key) == Some(&sig)) {
                progress(Event::Master { label: format!("Stacking {} raw {} for {label}", files.len(), plural(frame_type.name(), files.len())) });
                let off = match offset {
                    Some(o) => Some(materialise(o, "offset", manifest, loaded, built, progress)?),
                    None => None,
                };
                let mut imgs = Vec::with_capacity(files.len());
                let mut exps = Vec::new();
                for f in files {
                    let (mut img, hdr) = super::read_with_header(f)?;
                    if let Some(o) = &off {
                        anyhow::ensure!(o.image.data.len() == img.data.len(), "{} differs in size from {}", f.display(), o.label);
                        for (v, b) in img.data.iter_mut().zip(&o.image.data) {
                            *v -= b;
                        }
                    }
                    if let Some(e) = hdr.f64("EXPTIME") {
                        exps.push(e);
                    }
                    imgs.push(img);
                }
                let norm = if *frame_type == FrameType::Flat { Normalise::Multiplicative } else { Normalise::None };
                let st = integrate::stack(&imgs, &integrate::Params { normalise: norm, rejection: Rejection::auto(imgs.len()) })?;
                drop(imgs);
                let mut hdr = Header::default();
                hdr.push("IMAGETYP", format!("Master {}", frame_type.name()), "Integrated by crux");
                exps.sort_by(|a, b| a.total_cmp(b));
                if let Some(e) = exps.get(exps.len() / 2) {
                    hdr.push("EXPTIME", format!("{e}"), "[s] Median exposure of the inputs");
                }
                hdr.push("NCOMBINE", format!("{}", files.len()), "Frames stacked");
                hdr.push("CRUX-REJ", Rejection::auto(files.len()).name(), "Rejection");
                if let Some(d) = out.parent() {
                    std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))?;
                }
                crate::xisf::write_image(out, &st.image, &hdr, None)?;
                manifest.products.insert(key, sig);
                built.push(out.display().to_string());
                loaded.insert(out.clone(), Arc::new(st.image));
            }
            (out.clone(), None::<f64>)
        }
    };
    let img = match loaded.get(&path) {
        Some(i) => i.clone(),
        None => {
            progress(Event::Master { label: format!("Loading {}", path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()) });
            let i = Arc::new(crate::read_image(&path).with_context(|| format!("read {}", path.display()))?);
            loaded.insert(path.clone(), i.clone());
            i
        }
    };
    let exptime = exptime.or_else(|| crate::meta::read_meta(&path).ok().and_then(|m| m.exptime));
    Ok(Master { path: path.clone(), label: path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), image: img, exptime })
}

/// Masters loaded so far (shared between runs and previews).
pub type Loaded = HashMap<PathBuf, Arc<Image>>;

fn build_calibrator(plan: &Plan, cal: usize, params: CalParams, manifest: &mut Manifest, loaded: &mut Loaded, built: &mut Vec<String>, progress: &(dyn Fn(Event) + Sync)) -> Result<Calibrator> {
    let setup = &plan.cals[cal];
    let mut ld = |s: &Option<Source>, what: &str| -> Result<Option<Master>> { s.as_ref().map(|s| materialise(s, &format!("{} {what}", setup.label), manifest, loaded, built, progress)).transpose() };
    let dark = ld(&setup.dark, "dark")?;
    let flat = ld(&setup.flat, "flat")?;
    // A dark of the light's exposure already contains the bias.
    let need_bias = dark.as_ref().is_none_or(|d| match (d.exptime, setup.exptime) {
        (Some(a), Some(b)) => a > 0.0 && ((a - b) / a).abs() > 0.005,
        _ => false,
    });
    let bias = if need_bias { ld(&setup.bias, "bias")? } else { None };
    Calibrator::new(bias, dark, flat, setup.exptime, params)
}

/// The calibrator for one setup, building any masters it needs (shared with
/// `run` through the manifest). For previews.
pub fn calibrator(plan: &Plan, cal: usize, params: CalParams, loaded: &mut Loaded, progress: &(dyn Fn(Event) + Sync)) -> Result<Calibrator> {
    std::fs::create_dir_all(&plan.out).with_context(|| format!("create {}", plan.out.display()))?;
    let mut manifest = load_manifest(&plan.out);
    manifest.version = PROCESS_VERSION;
    manifest.root = plan.root.clone();
    let mut built = Vec::new();
    let c = build_calibrator(plan, cal, params, &mut manifest, loaded, &mut built, progress)?;
    if !built.is_empty() {
        save_manifest(&plan.out, &manifest)?;
    }
    Ok(c)
}

/// One light through every stage, in memory: (raw, calibrated without the
/// hot-pixel fix, calibrated, registered). For previews; a one-shot-colour
/// light is shown as its luminance (`cfa::luminance`) at every stage.
pub fn stages(l: &LightItem, cal: &Calibrator, clamp: Option<f32>) -> Result<[Image; 4]> {
    let (raw, _) = super::read_with_header(&l.path)?;
    let mut cal = cal.clone();
    cal.cfa = l.bayer.is_some();
    let mut no_hot = cal.clone();
    no_hot.hot = Arc::new(Vec::new());
    let (plain, _) = no_hot.apply(&raw)?;
    let (img, _) = cal.apply(&raw)?;
    let [raw, plain, img] = match l.bayer {
        Some(_) => [crate::cfa::luminance(&raw), crate::cfa::luminance(&plain), crate::cfa::luminance(&img)],
        None => [raw, plain, img],
    };
    let inv = l.reg.to_ref.inverse().context("singular registration")?;
    let reg = resample::warp(&img, &inv, img.width, img.height, clamp);
    Ok([raw, plain, img, reg])
}

fn light_settings_sig(plan: &Plan) -> String {
    serde_json::to_string(&(&plan.settings.cal, plan.settings.clamp)).unwrap_or_default()
}

fn cal_setup_sig(setup: &CalSetup) -> String {
    format!(
        "{}|{}|{}",
        setup.dark.as_ref().map(source_sig).unwrap_or_default(),
        setup.bias.as_ref().map(source_sig).unwrap_or_default(),
        setup.flat.as_ref().map(source_sig).unwrap_or_default()
    )
}

/// What a registered light was made from: the raw file, its masters, the
/// settings, the reference and the registration.
fn light_sig(field: &FieldPlan, l: &LightItem, cal_sig: &str, settings_sig: &str) -> String {
    format!(
        "v{PROCESS_VERSION}|{}|{}|{}|{}|{:?}",
        file_sig(&l.path),
        cal_sig,
        settings_sig,
        file_sig(&field.reference_path),
        l.reg.to_ref.0.map(|v| (v * 1e9).round() / 1e9)
    )
}

fn light_up_to_date(l: &LightItem, manifest: &Manifest, sig: &str) -> bool {
    l.outputs_exist() && manifest.products.get(&crate::paths::key(&l.registered)).map(String::as_str) == Some(sig) && l.calibrated.as_ref().is_none_or(|c| c.is_file())
}

/// A master light's inputs: the integration settings and the registered
/// frames that exist (with the index of the reference among them), or
/// None when the reference frame doesn't exist.
/// `present` indexes `StackPlan::inputs`.
fn stack_sig(plan: &Plan, st: &StackPlan, settings_sig: &str) -> Option<(String, Vec<usize>, usize)> {
    let field = &plan.fields[st.field];
    let inputs = st.inputs(field);
    let present: Vec<usize> = (0..inputs.len()).filter(|&i| inputs[i].1.is_file()).collect();
    let reference = present.iter().position(|&i| i == st.reference)?;
    let sig = format!("v{INTEGRATION_VERSION}|{settings_sig}|{}", present.iter().map(|&i| file_sig(&inputs[i].1)).collect::<Vec<_>>().join("|"));
    Some((sig, present, reference))
}

/// What `run` would do now, worked out from file sizes and times only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pending {
    /// Lights to calibrate and register (the rest are done and unchanged).
    pub lights: usize,
    /// Master lights to integrate.
    pub stacks: usize,
}

impl Pending {
    pub fn is_empty(&self) -> bool {
        self.lights == 0 && self.stacks == 0
    }
}

pub fn pending(plan: &Plan) -> Pending {
    let manifest = load_manifest(&plan.out);
    let settings_sig = light_settings_sig(plan);
    let cal_sigs: Vec<String> = plan.cals.iter().map(cal_setup_sig).collect();
    let mut todo: HashSet<(usize, usize)> = HashSet::new();
    for (fi, f) in plan.fields.iter().enumerate() {
        for (li, l) in f.lights.iter().enumerate() {
            let sig = light_sig(f, l, &cal_sigs[l.cal], &settings_sig);
            if !light_up_to_date(l, &manifest, &sig) {
                todo.insert((fi, li));
            }
        }
    }
    let combine_sig = serde_json::to_string(&plan.settings.combine).unwrap_or_default();
    let stacks = plan
        .stacks
        .iter()
        .filter(|st| {
            st.lights.iter().any(|&i| todo.contains(&(st.field, i)))
                || !st.out.is_file()
                || stack_sig(plan, st, &combine_sig).is_none_or(|(sig, _, _)| manifest.products.get(&crate::paths::key(&st.out)) != Some(&sig))
        })
        .count();
    Pending { lights: todo.len(), stacks }
}

/// Why lights are out of date: which part of each light's signature no
/// longer matches what the last run recorded, with the lights.
pub fn pending_reasons(plan: &Plan) -> std::collections::BTreeMap<&'static str, Vec<PathBuf>> {
    let manifest = load_manifest(&plan.out);
    let settings_sig = light_settings_sig(plan);
    let mut why = std::collections::BTreeMap::new();
    for f in &plan.fields {
        let ref_sig = file_sig(&f.reference_path);
        for l in &f.lights {
            let cal_sig = cal_setup_sig(&plan.cals[l.cal]);
            let sig = light_sig(f, l, &cal_sig, &settings_sig);
            if light_up_to_date(l, &manifest, &sig) {
                continue;
            }
            let reason = match manifest.products.get(&crate::paths::key(&l.registered)) {
                _ if !l.outputs_exist() => "no registered file",
                None => "not in the manifest",
                Some(old) if !old.starts_with(&format!("v{PROCESS_VERSION}|")) => "older processing version",
                Some(old) if !old.contains(&file_sig(&l.path)) => "raw file changed",
                Some(old) if !old.contains(&cal_sig) => "masters changed",
                Some(old) if !old.contains(&settings_sig) => "settings changed",
                Some(old) if !old.contains(&ref_sig) => "reference changed",
                Some(_) if l.calibrated.as_ref().is_some_and(|c| !c.is_file()) => "calibrated copy missing",
                Some(_) => "registration changed",
            };
            why.entry(reason).or_insert_with(Vec::new).push(l.path.clone());
        }
    }
    why
}

/// Calibrate and register every planned light.
pub fn run(plan: &Plan, progress: &(dyn Fn(Event) + Sync), cancel: &AtomicBool) -> Result<RunReport> {
    run_with(plan, &mut HashMap::new(), progress, cancel)
}

/// `run`, reusing masters already in memory.
pub fn run_with(plan: &Plan, loaded: &mut Loaded, progress: &(dyn Fn(Event) + Sync), cancel: &AtomicBool) -> Result<RunReport> {
    let t0 = Instant::now();
    std::fs::create_dir_all(&plan.out).with_context(|| format!("create {}", plan.out.display()))?;
    let mut manifest = load_manifest(&plan.out);
    manifest.version = PROCESS_VERSION;
    manifest.root = plan.root.clone();
    let mut rep = RunReport::default();
    let loaded = loaded;
    let total = plan.n_lights();
    let mut done = 0;
    let settings_sig = light_settings_sig(plan);
    let mut cal_sigs: HashMap<usize, String> = HashMap::new();
    // Lights grouped by calibration setup, across fields.
    let mut order: Vec<(&FieldPlan, &LightItem)> = plan.fields.iter().flat_map(|f| f.lights.iter().map(move |l| (f, l))).collect();
    order.sort_by_key(|(_, l)| l.cal);
    let mut current: Option<(usize, Calibrator, String)> = None;
    // What each light needs, in order; up-to-date ones are only reported.
    struct Work<'a> {
        field: &'a FieldPlan,
        l: &'a LightItem,
        sig: String,
        key: String,
        cal_sig: String,
        up_to_date: bool,
    }
    let work: Vec<Work> = order
        .into_iter()
        .map(|(field, l)| {
            let cal_sig = cal_sigs.entry(l.cal).or_insert_with(|| cal_setup_sig(&plan.cals[l.cal])).clone();
            let sig = light_sig(field, l, &cal_sig, &settings_sig);
            let up_to_date = light_up_to_date(l, &manifest, &sig);
            Work { field, l, key: crate::paths::key(&l.registered), sig, cal_sig, up_to_date }
        })
        .collect();
    // Three stages overlap: a reader thread fetches the next raw frame,
    // this thread calibrates and warps, a writer thread saves the outputs.
    // A light counts as processed (and goes in the manifest) only once its
    // files are written.
    type Read = (usize, Result<(Image, crate::header::Header)>, f64);
    struct Written {
        idx: usize,
        result: Result<()>,
        seconds: f64,
    }
    let timing = crate::timing::on();
    let (mut t_cal, mut t_warp, mut t_read, mut t_write) = (0f64, 0f64, 0f64, 0f64);
    std::thread::scope(|scope| -> Result<()> {
        let (read_tx, read_rx) = std::sync::mpsc::sync_channel::<Read>(1);
        let todo: Vec<(usize, &Path)> = work.iter().enumerate().filter(|(_, w)| !w.up_to_date).map(|(i, w)| (i, w.l.path.as_path())).collect();
        scope.spawn(move || {
            for (i, p) in todo {
                let t = Instant::now();
                let r = super::read_with_header(p);
                if read_tx.send((i, r, t.elapsed().as_secs_f64())).is_err() {
                    break;
                }
            }
        });
        let (write_tx, write_rx) = std::sync::mpsc::sync_channel::<(usize, Vec<(PathBuf, Image, crate::header::Header)>)>(2);
        let (done_tx, done_rx) = std::sync::mpsc::channel::<Written>();
        scope.spawn(move || {
            for (idx, files) in write_rx {
                let t = Instant::now();
                let result = files.iter().try_for_each(|(p, img, h)| write_xisf(p, img, h));
                if done_tx.send(Written { idx, result, seconds: t.elapsed().as_secs_f64() }).is_err() {
                    break;
                }
            }
        });
        let mut compute_secs: HashMap<usize, (f64, CalStats)> = HashMap::new();
        let mut finished = 0usize;
        let mut finish = |w: Written, manifest: &mut Manifest, rep: &mut RunReport, done: &mut usize, compute_secs: &mut HashMap<usize, (f64, CalStats)>| {
            let (secs, st) = compute_secs.remove(&w.idx).unwrap_or_default();
            let item = &work[w.idx];
            let mut res = LightResult { seconds: secs + w.seconds, ..Default::default() };
            match w.result {
                Ok(()) => {
                    res.cal = st;
                    manifest.products.insert(item.key.clone(), item.sig.clone());
                    rep.processed += 1;
                }
                Err(e) => {
                    res.error = Some(format!("{e:#}"));
                    rep.failed += 1;
                }
            }
            *done += 1;
            finished += 1;
            // Save as we go, so a crash or cancel keeps finished work.
            if finished % 8 == 0 {
                let _ = save_manifest(&plan.out, manifest);
            }
            t_write += w.seconds;
            progress(Event::Light { done: *done, total, path: item.l.path.clone(), result: res });
        };
        let mut pending = Some(write_tx);
        for (idx, w) in work.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                rep.cancelled = true;
                break;
            }
            while let Ok(wr) = done_rx.try_recv() {
                finish(wr, &mut manifest, &mut rep, &mut done, &mut compute_secs);
            }
            if w.up_to_date {
                rep.up_to_date += 1;
                done += 1;
                progress(Event::Light { done, total, path: w.l.path.clone(), result: LightResult { skipped_up_to_date: true, ..Default::default() } });
                continue;
            }
            let (field, l) = (w.field, w.l);
            let t1 = Instant::now();
            let r: Result<(CalStats, Vec<(PathBuf, Image, crate::header::Header)>)> = (|| {
                let (ri, raw, read_secs) = read_rx.recv().context("the reader stopped")?;
                debug_assert_eq!(ri, idx);
                t_read += read_secs;
                let (raw, hdr) = raw?;
                if current.as_ref().is_none_or(|c| c.0 != l.cal) {
                    let c = build_calibrator(plan, l.cal, plan.settings.cal.clone(), &mut manifest, loaded, &mut rep.masters_built, progress)?;
                    current = Some((l.cal, c, w.cal_sig.clone()));
                }
                let tc = Instant::now();
                let cal = &mut current.as_mut().unwrap().1;
                cal.cfa = l.bayer.is_some();
                let cal = &*cal;
                let (img, st) = cal.apply(&raw)?;
                drop(raw);
                t_cal += tc.elapsed().as_secs_f64();
                let mut files = Vec::new();
                let mut h = hdr.without_commentary();
                h.cards.retain(|c| !c.key.starts_with("CRUX-") && c.key != "PEDESTAL");
                h.push("CRUX-CAL", cal.notes.iter().filter(|n| n.starts_with("dark") || n.starts_with("flat") || n.starts_with("bias")).cloned().collect::<Vec<_>>().join(" — "), "crux calibration");
                h.push("CRUX-HOT", format!("{}", st.hot_fixed), "Hot pixels replaced");
                if st.pedestal > 0.0 {
                    h.push("PEDESTAL", format!("{}", (st.pedestal * raw_scale(&img)).round()), "[DN] Output pedestal added by crux");
                }
                let inv = l.reg.to_ref.inverse().context("singular registration")?;
                let tw = Instant::now();
                let mut hr = h.clone();
                hr.push("CRUX-REF", field.reference_path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), "Registration reference");
                hr.push("CRUX-RMS", format!("{:.3}", l.reg.rms), "[px] Registration residual");
                hr.push("CRUX-MAT", l.reg.to_ref.0.iter().map(|v| format!("{v:.10e}")).collect::<Vec<_>>().join(","), "Frame to reference homography");
                match l.bayer {
                    None => {
                        let reg = resample::warp(&img, &inv, img.width, img.height, plan.settings.clamp);
                        files.push((l.registered.clone(), reg, hr));
                    }
                    Some(b) => {
                        // Debayer, then each channel is a mono frame of its
                        // own "filter": no Bayer keywords any more.
                        let planes = crate::cfa::debayer(&img, b);
                        hr.cards.retain(|c| !matches!(c.key.as_str(), "BAYERPAT" | "XBAYROFF" | "YBAYROFF" | "COLORTYP" | "CFA-PAT" | "FILTER"));
                        hr.push("CRUX-CFA", b.name(), "Debayered by crux (bilinear) from this pattern");
                        if let Some(f) = Some(l.filter.clone()).filter(|f| !f.is_empty()) {
                            hr.push("CRUX-OSF", f, "Filter in front of the colour sensor");
                        }
                        for (c, p) in planes.iter().enumerate() {
                            let reg = resample::warp(p, &inv, p.width, p.height, plan.settings.clamp);
                            let mut hc = hr.clone();
                            hc.push("FILTER", crate::cfa::CHANNELS[c], "Colour channel of a one-shot-colour sensor");
                            files.push((channel_path(&l.registered, c as u8), reg, hc));
                        }
                    }
                }
                t_warp += tw.elapsed().as_secs_f64();
                if let Some(c) = &l.calibrated {
                    files.push((c.clone(), img, h));
                }
                Ok((st, files))
            })();
            match r {
                Ok((st, files)) => {
                    compute_secs.insert(idx, (t1.elapsed().as_secs_f64(), st));
                    if let Some(tx) = &pending {
                        if tx.send((idx, files)).is_err() {
                            anyhow::bail!("the writer stopped");
                        }
                    }
                }
                Err(e) => {
                    rep.failed += 1;
                    done += 1;
                    let res = LightResult { error: Some(format!("{e:#}")), seconds: t1.elapsed().as_secs_f64(), ..Default::default() };
                    progress(Event::Light { done, total, path: l.path.clone(), result: res });
                }
            }
        }
        // Let the reader stop, then wait for the last writes.
        drop(read_rx);
        pending.take();
        for wr in done_rx {
            finish(wr, &mut manifest, &mut rep, &mut done, &mut compute_secs);
        }
        Ok(())
    })?;
    if timing {
        eprintln!("process: read {t_read:.1} s, calibrate {t_cal:.1} s, warp {t_warp:.1} s, write {t_write:.1} s (overlapping), lights done at {:.1} s", t0.elapsed().as_secs_f64());
    }
    // Earlier versions wrote the plan here too; nothing read it.
    let _ = std::fs::remove_file(plan.out.join("crux-plan.json"));
    save_manifest(&plan.out, &manifest)?;
    if !rep.cancelled {
        integrate_all(plan, &mut manifest, &mut rep, progress, cancel)?;
        save_manifest(&plan.out, &manifest)?;
    }
    rep.seconds = t0.elapsed().as_secs_f64();
    Ok(rep)
}

/// Earlier integration reports, by master path.
pub fn load_integrations(out: &Path) -> BTreeMap<String, combine::Report> {
    std::fs::read(out.join("crux-integration.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

fn integrate_all(plan: &Plan, manifest: &mut Manifest, rep: &mut RunReport, progress: &(dyn Fn(Event) + Sync), cancel: &AtomicBool) -> Result<()> {
    let mut reports = load_integrations(&plan.out);
    let total = plan.stacks.len();
    let settings_sig = serde_json::to_string(&plan.settings.combine).unwrap_or_default();
    for (k, st) in plan.stacks.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            rep.cancelled = true;
            break;
        }
        let field = &plan.fields[st.field];
        let label = format!("{} · {}", st.object, if st.filter.is_empty() { "no filter" } else { &st.filter });
        let lights: Vec<&LightItem> = st.lights.iter().map(|&i| &field.lights[i]).collect();
        // Only lights whose registered frame exists (a failed light is left out).
        let Some((sig, present, reference)) = stack_sig(plan, st, &settings_sig) else {
            rep.integration_failed += 1;
            progress(Event::Integrated { index: k + 1, total, label, report: None, error: Some("the reference frame wasn't processed".into()) });
            continue;
        };
        let key = crate::paths::key(&st.out);
        if st.out.is_file() && manifest.products.get(&key) == Some(&sig) {
            // A FITS copy asked for after the master was made.
            if let Some(e) = fits_copy(plan, &st.out) {
                rep.integration_failed += 1;
                progress(Event::Integrated { index: k + 1, total, label, report: None, error: Some(e) });
                continue;
            }
            rep.integrations_up_to_date += 1;
            progress(Event::Integrated { index: k + 1, total, label, report: reports.get(&key).cloned().map(Box::new), error: None });
            continue;
        }
        let all = st.inputs(field);
        let inputs: Vec<combine::Input> = present
            .iter()
            .map(|&i| {
                let (k, ref path) = all[i];
                let l = lights[k];
                let mut name = l.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                if st.also.is_some() {
                    name = format!("{name} ({})", crate::meta::read_meta(path).ok().and_then(|m| m.filter).unwrap_or_default());
                }
                combine::Input { path: path.clone(), name, exptime: l.exptime, fwhm: l.fwhm, night: l.night, grade: l.grade }
            })
            .collect();
        let extra = crate::meta::read_meta(&all[present[reference]].1).map(|m| m.header).unwrap_or_default();
        let mut extra = extra;
        extra.cards.retain(|c| !matches!(c.key.as_str(), "CRUX-REF" | "CRUX-RMS" | "CRUX-MAT" | "CRUX-HOT" | "PEDESTAL"));
        let r = combine::integrate(&inputs, reference, &st.stars, &plan.settings.combine, &st.object, &st.filter, &st.out, &extra, &|step| progress(Event::Integrate { index: k + 1, total, label: label.clone(), step }), cancel);
        match r {
            Ok(r) => {
                manifest.products.insert(key.clone(), sig);
                reports.insert(key, r.clone());
                let _ = crate::paths::write_atomic(&plan.out.join("crux-integration.json"), &serde_json::to_vec_pretty(&reports)?);
                if let Some(e) = fits_copy(plan, &st.out) {
                    rep.integration_failed += 1;
                    progress(Event::Integrated { index: k + 1, total, label, report: None, error: Some(e) });
                    continue;
                }
                rep.integrated += 1;
                progress(Event::Integrated { index: k + 1, total, label, report: Some(Box::new(r)), error: None });
            }
            Err(e) if cancel.load(Ordering::Relaxed) => {
                let _ = e;
                rep.cancelled = true;
                break;
            }
            Err(e) => {
                rep.integration_failed += 1;
                progress(Event::Integrated { index: k + 1, total, label, report: None, error: Some(format!("{e:#}")) });
            }
        }
    }
    Ok(())
}

/// The master's FITS copy when `[process] fits` is on; the error, if any.
fn fits_copy(plan: &Plan, master: &Path) -> Option<String> {
    if !plan.settings.fits {
        return None;
    }
    crate::fits::ensure_copy(master).err().map(|e| format!("the master was written, but not its FITS copy: {e:#}"))
}

fn raw_scale(img: &Image) -> f32 {
    img.adu_scale.max(1.0)
}

fn write_xisf(p: &Path, img: &Image, h: &Header) -> Result<()> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))?;
    }
    let tmp = p.with_extension("xisf.tmp");
    crate::xisf::write_image(&tmp, img, h, None)?;
    std::fs::rename(&tmp, p).with_context(|| format!("replace {}", p.display()))?;
    Ok(())
}
