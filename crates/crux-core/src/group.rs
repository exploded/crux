//! Group frames into calibration sets and light groups, then match each
//! light group to darks, bias and flats and say what's missing.

use crate::classify::{FrameType, Role};
use crate::config::Config;
use crate::scan::Frame;
use chrono::{NaiveDate, NaiveDateTime};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Group {
    pub id: usize,
    pub kind: FrameType,
    /// A single integrated master file (WBPP masterDark etc.).
    pub master: bool,
    /// Every member is in a reject folder (FLAT.bad, BAD).
    pub retired: bool,
    pub label: String,
    /// Indices into `ScanResult::frames`.
    pub members: Vec<usize>,
    /// Members already sitting in a reject folder.
    pub n_rejected: usize,
    pub object: Option<String>,
    pub filter: Option<String>,
    /// Exposure in seconds (median for flats).
    pub exptime: Option<f64>,
    pub gain: Option<f64>,
    pub offset: Option<f64>,
    pub binning: Option<u32>,
    pub temp: Option<f64>,
    pub width: u32,
    pub height: u32,
    pub night: Option<NaiveDate>,
    pub first: Option<NaiveDateTime>,
    pub last: Option<NaiveDateTime>,
    /// For a master: the raw set it was integrated from, found by matching
    /// the master's observation time span against raw sets.
    pub built_from: Option<usize>,
}

impl Group {
    pub fn active_count(&self) -> usize {
        self.members.len() - self.n_rejected
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatCandidate {
    pub group: usize,
    /// Flat night minus light night.
    pub nights_apart: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Calibration {
    /// Light or flat group being calibrated.
    pub group: usize,
    pub darks: Vec<usize>,
    pub bias: Vec<usize>,
    pub dark_flats: Vec<usize>,
    /// Lights only; nearest night first.
    pub flats: Vec<FlatCandidate>,
    /// Assumptions made while matching.
    pub notes: Vec<String>,
    pub missing: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Grouping {
    pub groups: Vec<Group>,
    pub light_calibration: Vec<Calibration>,
    pub flat_calibration: Vec<Calibration>,
    /// Night → light group ids.
    pub sessions: BTreeMap<NaiveDate, Vec<usize>>,
    /// Frames not grouped: processed products and unknown types.
    pub ungrouped: Vec<usize>,
    /// Problems found while grouping, in plain English.
    pub warnings: Vec<String>,
}

impl Grouping {
    pub fn by_kind(&self, kind: FrameType) -> impl Iterator<Item = &Group> {
        self.groups.iter().filter(move |g| g.kind == kind)
    }
}

pub fn fmt_exp(e: f64) -> String {
    if e >= 10.0 {
        format!("{e:.0} s")
    } else if e >= 1.0 {
        format!("{e:.1} s")
    } else {
        format!("{e:.2} s")
    }
}

fn opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x}")).unwrap_or_else(|| "?".into())
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

/// Most common value among members.
fn mode<T: Clone + PartialEq>(vals: impl Iterator<Item = Option<T>>) -> Option<T> {
    let mut counts: Vec<(T, usize)> = Vec::new();
    for v in vals.flatten() {
        match counts.iter_mut().find(|(x, _)| *x == v) {
            Some(c) => c.1 += 1,
            None => counts.push((v, 1)),
        }
    }
    counts.into_iter().max_by_key(|c| c.1).map(|c| c.0)
}

/// Exposures within `rel` of their neighbour (sorted, chained) share one
/// value, the cluster's middle one: a DSLR's bulb exposures through an
/// ASIAIR are recorded as 239.8, 239.9, 240.1 s and split one night into
/// three stacks. Returns (exposure, shared value) pairs, sorted.
pub fn exposure_clusters(mut exps: Vec<f64>, rel: f64) -> Vec<(f64, f64)> {
    exps.retain(|e| e.is_finite());
    exps.sort_by(f64::total_cmp);
    exps.dedup();
    let mut out = Vec::with_capacity(exps.len());
    let mut start = 0;
    for i in 1..=exps.len() {
        if i == exps.len() || exps[i] - exps[i - 1] > rel * exps[i] {
            let mid = exps[start + (i - start) / 2];
            out.extend(exps[start..i].iter().map(|&e| (e, mid)));
            start = i;
        }
    }
    out
}

pub fn build(frames: &[Frame], cfg: &Config) -> Grouping {
    let mut buckets: HashMap<String, Vec<usize>> = HashMap::new();
    let mut ungrouped = Vec::new();
    let shared = exposure_clusters(frames.iter().filter_map(|f| f.meta.exptime).collect(), cfg.grouping.exposure_tolerance);
    let shared_exp = |e: f64| shared.binary_search_by(|p| p.0.total_cmp(&e)).map(|k| shared[k].1).unwrap_or(e);
    for (i, f) in frames.iter().enumerate() {
        let m = &f.meta;
        let c = &f.class;
        if c.role == Role::Processed || c.frame_type == FrameType::Unknown {
            ungrouped.push(i);
            continue;
        }
        let exp = m.exptime.map(|e| format!("{:.2}", shared_exp(e))).unwrap_or_default();
        let filt = m.filter.clone().unwrap_or_default().to_uppercase();
        let size = m.size_key();
        let bin = m.binning();
        let night = c.night.map(|n| n.to_string()).unwrap_or_default();
        let key = if c.role == Role::Master {
            format!("master|{}", m.path.display())
        } else {
            match c.frame_type {
                FrameType::Light => format!(
                    "L|{}|{filt}|{exp}|{night}|{bin}|{size}",
                    m.object.clone().unwrap_or_default().to_lowercase()
                ),
                FrameType::Flat => format!("F|{filt}|{night}|{bin}|{size}|{}", c.in_reject_dir),
                // By set point only: an uncooled camera (a DSLR's 27, 28,
                // 29 °C in one night) drifts, and splitting by its sensor
                // temperature left each set a few frames.
                FrameType::Dark => format!(
                    "D|{exp}|{}|{}|{bin}|{}|{size}",
                    opt(m.gain),
                    opt(m.offset),
                    m.set_temp.map(|t| format!("{:.0}", t)).unwrap_or_default()
                ),
                FrameType::Bias => format!("B|{}|{}|{bin}|{size}", opt(m.gain), opt(m.offset)),
                FrameType::DarkFlat => format!("DF|{exp}|{}|{}|{bin}|{size}", opt(m.gain), opt(m.offset)),
                FrameType::Unknown => unreachable!(),
            }
        };
        buckets.entry(key).or_default().push(i);
    }

    let mut groups: Vec<Group> = buckets
        .into_values()
        .map(|mut members| {
            members.sort_by_key(|&i| (frames[i].class.local_time, i));
            make_group(frames, members)
        })
        .collect();
    groups.sort_by(|a, b| {
        (a.kind, a.master, a.night, a.object.clone(), a.filter.clone(), a.first)
            .partial_cmp(&(b.kind, b.master, b.night, b.object.clone(), b.filter.clone(), b.first))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, g) in groups.iter_mut().enumerate() {
        g.id = i;
    }
    let warnings = trace_masters(frames, &mut groups);

    let mut sessions: BTreeMap<NaiveDate, Vec<usize>> = BTreeMap::new();
    for g in groups.iter().filter(|g| g.kind == FrameType::Light && !g.master) {
        if let Some(n) = g.night {
            sessions.entry(n).or_default().push(g.id);
        }
    }

    let light_calibration = groups
        .iter()
        .filter(|g| g.kind == FrameType::Light && !g.master)
        .map(|g| match_light(g, &groups, cfg))
        .collect();
    let flat_calibration = groups
        .iter()
        .filter(|g| g.kind == FrameType::Flat && !g.master && !g.retired)
        .map(|g| match_flat(g, &groups, cfg))
        .collect();

    Grouping { groups, light_calibration, flat_calibration, sessions, ungrouped, warnings }
}

/// Link each master to the raw set it was integrated from, by overlapping
/// its observation span (XISF Observation:Time:Start/End) with raw sets of
/// the same type, filter and size. A master built from a retired set is
/// retired too: on the LMC data, WBPP built master\masterFlat_*.xisf from
/// the cloudy 2026-11-04 flats after they'd been moved to FLAT.bad.
fn trace_masters(frames: &[Frame], groups: &mut [Group]) -> Vec<String> {
    let mut warnings = Vec::new();
    let slack = chrono::Duration::minutes(1);
    for gi in 0..groups.len() {
        if !groups[gi].master {
            continue;
        }
        let f = &frames[groups[gi].members[0]];
        let (Some(start), Some(obs)) = (f.class.local_time, f.meta.date_obs) else { continue };
        let end = f.meta.date_end.map(|e| e + (start - obs)).unwrap_or(start).max(start);
        let m = &groups[gi];
        let filt = m.filter.clone().unwrap_or_default().to_uppercase();
        let src = groups
            .iter()
            .filter(|r| !r.master && r.kind == m.kind && (r.width, r.height) == (m.width, m.height))
            .filter(|r| r.filter.clone().unwrap_or_default().to_uppercase() == filt)
            .find(|r| matches!((r.first, r.last), (Some(a), Some(b)) if start <= b + slack && end >= a - slack))
            .map(|r| r.id);
        groups[gi].last = Some(end);
        if let Some(s) = src {
            groups[gi].built_from = Some(s);
            if groups[s].retired {
                groups[gi].retired = true;
                warnings.push(format!(
                    "{} was integrated from {}, which is in a reject folder; it won't be used, and anything stacked with it carries the problem",
                    groups[gi].label, groups[s].label
                ));
                groups[gi].label.push_str(" [from retired set]");
            }
        }
    }
    warnings
}

fn make_group(frames: &[Frame], members: Vec<usize>) -> Group {
    let f0 = &frames[members[0]];
    let kind = f0.class.frame_type;
    let master = f0.class.role == Role::Master;
    let ms = || members.iter().map(|&i| &frames[i].meta);
    let n_rejected = members.iter().filter(|&&i| frames[i].class.in_reject_dir).count();
    let exptime = median(ms().filter_map(|m| m.exptime).collect());
    let filter = mode(ms().map(|m| m.filter.clone()));
    let object = mode(ms().map(|m| m.object.clone()));
    let gain = mode(ms().map(|m| m.gain.map(|v| v.to_bits()))).map(f64::from_bits);
    let offset = mode(ms().map(|m| m.offset.map(|v| v.to_bits()))).map(f64::from_bits);
    let binning = mode(ms().map(|m| m.xbin));
    let temp = median(ms().filter_map(|m| m.temp()).collect());
    let times: Vec<NaiveDateTime> = members.iter().filter_map(|&i| frames[i].class.local_time).collect();
    let night = mode(members.iter().map(|&i| frames[i].class.night));
    let retired = n_rejected == members.len();

    let filt = filter.clone().unwrap_or_else(|| "no filter".into());
    let exp_s = exptime.map(fmt_exp).unwrap_or_else(|| "? s".into());
    let night_s = night.map(|n| n.to_string()).unwrap_or_else(|| "no date".into());
    let label = if master {
        format!(
            "master {} {}({})",
            kind.name(),
            match kind {
                FrameType::Dark | FrameType::DarkFlat => format!("{exp_s} "),
                FrameType::Flat => format!("{filt} "),
                _ => String::new(),
            },
            f0.meta.file_name()
        )
    } else {
        match kind {
            FrameType::Light => format!("{} · {filt} · {exp_s} · {night_s}", object.clone().unwrap_or_else(|| "no object".into())),
            FrameType::Flat => format!("{filt} flats · {night_s}{}", if retired { " (retired)" } else { "" }),
            FrameType::Dark => format!(
                "{exp_s} darks · gain {} · offset {} · {}",
                opt(gain),
                opt(offset),
                temp.map(|t| format!("{t:.0} °C")).unwrap_or_else(|| "? °C".into())
            ),
            FrameType::Bias => format!("bias · gain {} · offset {}", opt(gain), opt(offset)),
            FrameType::DarkFlat => format!("{exp_s} dark flats · gain {} · offset {}", opt(gain), opt(offset)),
            FrameType::Unknown => "unknown".into(),
        }
    };
    Group {
        id: 0,
        kind,
        master,
        retired,
        label,
        n_rejected,
        object,
        filter,
        exptime,
        gain,
        offset,
        binning,
        temp,
        width: f0.meta.width,
        height: f0.meta.height,
        night,
        first: times.iter().min().copied(),
        last: times.iter().max().copied(),
        built_from: None,
        members,
    }
}

#[derive(PartialEq)]
enum Fit {
    Yes,
    Unknown,
    No,
}

fn fit_f(a: Option<f64>, b: Option<f64>, tol: f64) -> Fit {
    match (a, b) {
        (Some(x), Some(y)) if (x - y).abs() <= tol => Fit::Yes,
        (Some(_), Some(_)) => Fit::No,
        _ => Fit::Unknown,
    }
}

fn fit_bin(a: Option<u32>, b: Option<u32>) -> Fit {
    fit_f(a.map(f64::from), b.map(f64::from), 0.0)
}

/// Compare a calibration group against target settings. Returns None on a
/// mismatch, else the list of fields that couldn't be checked.
fn compatible(target: &Group, cal: &Group, temp_tol: Option<f64>) -> Option<Vec<&'static str>> {
    if (target.width, target.height) != (cal.width, cal.height) {
        return None;
    }
    let mut unknown = Vec::new();
    for (name, fit) in [
        ("gain", fit_f(target.gain, cal.gain, 0.0)),
        ("offset", fit_f(target.offset, cal.offset, 0.0)),
        ("binning", fit_bin(target.binning, cal.binning)),
    ] {
        match fit {
            Fit::No => return None,
            Fit::Unknown => unknown.push(name),
            Fit::Yes => {}
        }
    }
    if let Some(tol) = temp_tol {
        match fit_f(target.temp, cal.temp, tol) {
            Fit::No => return None,
            Fit::Unknown => unknown.push("temperature"),
            Fit::Yes => {}
        }
    }
    Some(unknown)
}

fn exposure_matches(a: Option<f64>, b: Option<f64>, rel: f64) -> bool {
    matches!((a, b), (Some(x), Some(y)) if (x - y).abs() <= (rel * x.max(y)).max(0.005))
}

fn settings(g: &Group, with_temp: bool) -> String {
    let mut s = format!("gain {}, offset {}, bin {}", opt(g.gain), opt(g.offset), g.binning.map(|b| b.to_string()).unwrap_or("?".into()));
    if with_temp {
        s.push_str(&format!(", {}", g.temp.map(|t| format!("{t:.0} °C")).unwrap_or("? °C".into())));
    }
    s
}

fn note_unknown(notes: &mut Vec<String>, cal: &Group, unknown: &[&str]) {
    if !unknown.is_empty() {
        notes.push(format!("{}: {} not recorded, assumed to match", cal.label, unknown.join(", ")));
    }
}

/// Pick calibration groups of `kind` compatible with `target`, ranked by
/// fewest unknown fields then most frames.
fn pick(target: &Group, groups: &[Group], kind: FrameType, exp: Option<(Option<f64>, f64)>, temp_tol: Option<f64>, notes: &mut Vec<String>) -> Vec<usize> {
    let mut found: Vec<(usize, Vec<&str>)> = groups
        .iter()
        .filter(|c| c.kind == kind && !c.retired)
        .filter(|c| exp.is_none_or(|(e, rel)| exposure_matches(e, c.exptime, rel)))
        .filter_map(|c| compatible(target, c, temp_tol).map(|u| (c.id, u)))
        .collect();
    found.sort_by_key(|(id, u)| (u.len(), usize::MAX - groups[*id].active_count()));
    if let Some((id, u)) = found.first() {
        note_unknown(notes, &groups[*id], u);
    }
    found.into_iter().map(|(id, _)| id).collect()
}

fn match_light(g: &Group, groups: &[Group], cfg: &Config) -> Calibration {
    let mut c = Calibration { group: g.id, ..Default::default() };
    let tol = cfg.grouping.dark_temp_tolerance_c;
    c.darks = pick(g, groups, FrameType::Dark, Some((g.exptime, cfg.grouping.exposure_tolerance)), Some(tol), &mut c.notes);
    if c.darks.is_empty() {
        let mut msg = format!("no {} darks at {}", g.exptime.map(fmt_exp).unwrap_or("?".into()), settings(g, true));
        let near: Vec<String> = groups
            .iter()
            .filter(|d| d.kind == FrameType::Dark && exposure_matches(g.exptime, d.exptime, cfg.grouping.exposure_tolerance))
            .map(|d| d.label.clone())
            .collect();
        if !near.is_empty() {
            msg.push_str(&format!(" (same exposure, other settings: {})", near.join("; ")));
        }
        c.missing.push(msg);
    }
    c.bias = pick(g, groups, FrameType::Bias, None, None, &mut c.notes);
    if c.bias.is_empty() && c.darks.is_empty() {
        c.missing.push(format!("no bias at {}", settings(g, false)));
    }

    let filt = g.filter.clone().unwrap_or_default().to_uppercase();
    let flat_ok = |f: &&Group| {
        f.kind == FrameType::Flat
            && f.filter.clone().unwrap_or_default().to_uppercase() == filt
            && (f.width, f.height) == (g.width, g.height)
            && fit_bin(g.binning, f.binning) != Fit::No
    };
    let mut flats: Vec<FlatCandidate> = groups
        .iter()
        .filter(flat_ok)
        .filter(|f| !f.retired)
        .map(|f| FlatCandidate {
            group: f.id,
            nights_apart: match (f.night, g.night) {
                (Some(a), Some(b)) => (a - b).num_days(),
                _ => i64::MAX / 2,
            },
        })
        .collect();
    // Raw sets before masters at equal distance; masters often lack a date.
    flats.sort_by_key(|f| (f.nights_apart.abs(), groups[f.group].master));
    c.flats = flats;
    if c.flats.is_empty() {
        let retired = groups.iter().filter(flat_ok).filter(|f| f.retired).count();
        if retired > 0 {
            c.missing.push(format!("no usable {filt} flats; only {retired} retired set(s)"));
        } else {
            c.missing.push(format!("no {} flats at bin {}", if filt.is_empty() { "unfiltered".into() } else { filt }, g.binning.unwrap_or(1)));
        }
    }
    c
}

/// Bias and dark flats for a flat set (also used for retired sets, which
/// aren't in `Grouping::flat_calibration`).
pub fn match_flat(g: &Group, groups: &[Group], cfg: &Config) -> Calibration {
    let mut c = Calibration { group: g.id, ..Default::default() };
    c.bias = pick(g, groups, FrameType::Bias, None, None, &mut c.notes);
    c.dark_flats = pick(g, groups, FrameType::DarkFlat, Some((g.exptime, 0.05)), None, &mut c.notes);
    if c.dark_flats.is_empty() {
        c.darks = pick(g, groups, FrameType::Dark, Some((g.exptime, 0.05)), None, &mut c.notes);
    }
    if c.bias.is_empty() && c.dark_flats.is_empty() && c.darks.is_empty() {
        c.missing.push(format!(
            "no bias or dark flats for {} ({} flats at {})",
            g.label,
            g.exptime.map(fmt_exp).unwrap_or("?".into()),
            settings(g, false)
        ));
    }
    let _ = cfg;
    c
}

#[cfg(test)]
mod tests {
    #[test]
    fn exposure_clusters_merge_bulb_jitter() {
        let c = super::exposure_clusters(vec![240.1, 239.8, 0.00125, 239.9, 300.0, 239.9, 0.1], 0.005);
        let get = |e: f64| c.iter().find(|p| p.0 == e).unwrap().1;
        assert_eq!(get(239.8), get(240.1));
        assert_eq!(get(239.9), 239.9);
        assert_eq!(get(300.0), 300.0);
        assert_eq!(get(0.1), 0.1);
    }
}
