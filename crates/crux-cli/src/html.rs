//! Standalone HTML report: one file, no external resources. The page is a
//! template (`report_template.html`) with the triage data embedded as JSON.

use anyhow::{Context, Result};
use crux_core::grade::Grade;
use crux_core::pipeline::Triage;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

const TEMPLATE: &str = include_str!("report_template.html");

fn grade_key(g: Grade) -> &'static str {
    match g {
        Grade::Good => "good",
        Grade::Warn => "warn",
        Grade::Reject => "reject",
    }
}

pub fn write(t: &Triage, path: &Path) -> Result<()> {
    let g = &t.grouping;
    let rep: BTreeMap<usize, &crux_core::grade::FrameReport> = t.reports.iter().map(|r| (r.frame, r)).collect();
    let mut nights = Vec::new();
    for (night, ids) in &g.sessions {
        let noon = night.and_hms_opt(12, 0, 0).expect("valid time");
        let mut frames: Vec<Value> = Vec::new();
        for &id in ids {
            for i in &g.groups[id].members {
                let Some(r) = rep.get(i) else { continue };
                let f = &t.scan.frames[*i];
                let m = t.metrics.get(i);
                let Some(lt) = f.class.local_time else { continue };
                let reasons: Vec<String> = {
                    let mut fl: Vec<_> = r.flags.iter().filter(|x| x.grade != Grade::Good).collect();
                    fl.sort_by(|a, b| b.grade.cmp(&a.grade));
                    fl.iter().map(|x| x.reason.clone()).collect()
                };
                frames.push(json!({
                    "x": (lt - noon).num_seconds() as f64 / 60.0,
                    "time": lt.format("%H:%M").to_string(),
                    "object": f.meta.object.clone().unwrap_or_default(),
                    "grade": grade_key(r.grade),
                    "inReject": f.class.in_reject_dir,
                    "transparency": r.transparency.as_ref().map(|x| x.value),
                    "fwhm": m.and_then(|m| m.fwhm),
                    "ecc": m.and_then(|m| m.ecc),
                    "stars": m.map(|m| m.n_stars),
                    "sky": r.sky_e_s,
                    "trails": m.map(|m| m.trails.len()).filter(|&n| n > 0),
                    "reasons": reasons,
                    "notes": r.notes,
                    "file": f.path().strip_prefix(&t.scan.root).unwrap_or(f.path()).display().to_string(),
                }));
            }
        }
        frames.sort_by(|a, b| a["x"].as_f64().partial_cmp(&b["x"].as_f64()).unwrap_or(std::cmp::Ordering::Equal));
        nights.push(json!({ "label": format!("Night of {} ({})", night, night.format("%a")), "frames": frames }));
    }

    let mut missing: BTreeMap<String, usize> = BTreeMap::new();
    for c in g.light_calibration.iter().chain(&g.flat_calibration) {
        for m in &c.missing {
            *missing.entry(m.clone()).or_default() += 1;
        }
    }
    let mut used: BTreeMap<String, usize> = BTreeMap::new();
    for c in &g.light_calibration {
        if let Some(&d) = c.darks.first() {
            *used.entry(format!("Darks: {}", g.groups[d].label)).or_default() += 1;
        }
        if let Some(&b) = c.bias.first() {
            *used.entry(format!("Bias: {}", g.groups[b].label)).or_default() += 1;
        }
        if let Some(f) = c.flats.first() {
            *used.entry(format!("Flats: {}", g.groups[f.group].label)).or_default() += 1;
        }
    }
    let plural = |n: usize| if n == 1 { String::new() } else { format!(" (×{n})") };
    let folder = t.scan.root.file_name().and_then(|n| n.to_str()).unwrap_or("folder").to_string();
    let data = json!({
        "title": format!("{folder} · crux triage"),
        "root": t.scan.root.display().to_string(),
        "generated": chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(),
        "warnings": g.warnings,
        "missing": missing.iter().map(|(k, n)| format!("{k}{}", plural(*n))).collect::<Vec<_>>(),
        "calibration": used.iter().map(|(k, n)| format!("{k} — {n} light group{}", if *n == 1 { "" } else { "s" })).collect::<Vec<_>>(),
        "nights": nights,
        "cal": cal_json(t),
        "footer": format!(
            "crux {} · {} image files scanned, {} lights measured. Grades compare each frame with its night (focus, tracking, stars) and its field (sky, gradient, transparency).",
            env!("CARGO_PKG_VERSION"),
            t.scan.stats.files,
            t.metrics.len()
        ),
    });
    // Keep the JSON from closing the <script> element early.
    let json = serde_json::to_string(&data)?.replace("</", "<\\/");
    let html = TEMPLATE.replace("/*DATA*/", &json);
    std::fs::write(path, html).with_context(|| format!("write {}", path.display()))
}

/// Calibration checks for the report: sets, flagged frames, flat error maps
/// (downsampled 3× to keep the file small) and flat-set comparisons.
fn cal_json(t: &Triage) -> Value {
    use crux_core::classify::FrameType;
    let g = &t.grouping;
    let rep: BTreeMap<usize, &crux_core::grade::FrameReport> = t.cal.reports.iter().map(|r| (r.frame, r)).collect();
    let order = |k: FrameType| match k {
        FrameType::Flat => 0,
        FrameType::Dark => 1,
        FrameType::DarkFlat => 2,
        _ => 3,
    };
    let mut sets: Vec<&crux_core::calcheck::SetReport> = t.cal.sets.iter().collect();
    sets.sort_by_key(|s| {
        let gr = &g.groups[s.group];
        (order(gr.kind), gr.filter.clone(), gr.master, gr.first)
    });
    let sets: Vec<Value> = sets
        .into_iter()
        .map(|s| {
            let gr = &g.groups[s.group];
            let mut facts = Vec::new();
            if !gr.master {
                facts.push(format!("{} frames", s.measured));
            }
            if gr.kind == FrameType::Flat {
                if let Some(l) = s.flat_level {
                    facts.push(format!("level {:.0}% of full scale", 100.0 * l));
                }
                if let Some(w) = s.worst_change {
                    facts.push(format!("worst change {w:.1}%"));
                }
                if s.n_saturated > 0 {
                    facts.push(format!("{} saturated", s.n_saturated));
                }
            } else {
                if let Some(l) = s.level_adu {
                    facts.push(format!("level {l:.1} ADU"));
                }
                if let Some(n) = s.noise_adu {
                    facts.push(format!("noise {n:.2} ADU"));
                }
                if gr.kind != FrameType::Bias {
                    if let Some(h) = s.hot {
                        facts.push(format!("{:.3}% hot pixels", 100.0 * h));
                    }
                }
            }
            // A master is one file: its reasons are the set's.
            let frames: Vec<Value> = if gr.master { &[][..] } else { &gr.members[..] }
                .iter()
                .filter_map(|i| rep.get(i))
                .filter_map(|r| {
                    let own: Vec<&crux_core::grade::Flag> =
                        r.flags.iter().filter(|f| f.grade != Grade::Good && !matches!(f.check, crux_core::grade::Check::FlatMatch | crux_core::grade::Check::Library)).collect();
                    let worst = own.iter().map(|f| f.grade).max()?;
                    Some(json!({
                        "file": t.scan.frames[r.frame].meta.file_name(),
                        "grade": grade_key(worst),
                        "reasons": own.iter().map(|f| f.reason.clone()).collect::<Vec<_>>(),
                    }))
                })
                .collect();
            let vs = s.vs_lights.as_ref().map(|v| {
                let e = &v.error;
                let f = 3;
                let (w2, h2) = (e.w / f, e.h / f);
                let mut cells = Vec::with_capacity(w2 * h2);
                for y in 0..h2 {
                    for x in 0..w2 {
                        let mut acc = 0f32;
                        for dy in 0..f {
                            for dx in 0..f {
                                acc += e.map[(y * f + dy) * e.w + x * f + dx];
                            }
                        }
                        // Tenths of a percent from 1.
                        cells.push(((acc / (f * f) as f32 - 1.0) * 1000.0).round() as i32);
                    }
                }
                json!({
                    "spread": e.spread,
                    "grade": grade_key(v.grade),
                    "reliable": v.reliable,
                    "masked": e.masked,
                    "lights": v.n_lights,
                    "nights": match (v.first_night, v.last_night) {
                        (Some(a), Some(b)) if a == b => format!("night of {a}"),
                        (Some(a), Some(b)) => format!("nights {a} to {b}"),
                        _ => String::new(),
                    },
                    "edges": e.edges,
                    "map": { "w": w2, "h": h2, "v": cells },
                })
            });
            json!({
                "label": gr.label,
                "kind": gr.kind.name(),
                "grade": grade_key(s.grade),
                "facts": facts.join(" · "),
                "offset": s.offset,
                "reasons": s.flags.iter().filter(|f| f.grade != Grade::Good).map(|f| f.reason.clone()).collect::<Vec<_>>(),
                "notes": s.notes,
                "frames": frames,
                "vs": vs,
            })
        })
        .collect();
    let short = |id: usize| {
        let gr = &g.groups[id];
        if gr.master {
            format!("master ({})", t.scan.frames[gr.members[0]].meta.file_name())
        } else {
            format!("{}{}", gr.night.map(|n| n.to_string()).unwrap_or_default(), if gr.retired { " (retired)" } else { "" })
        }
    };
    let pairs: Vec<Value> = t
        .cal
        .pairs
        .iter()
        .map(|p| {
            json!({
                "filter": g.groups[p.a].filter.clone().unwrap_or_default(),
                "a": short(p.a),
                "b": short(p.b),
                "spread": p.spread,
                "dust": p.dust,
                "combinable": p.combinable,
                "shape": p.edges.describe(0.5),
            })
        })
        .collect();
    let choices: Vec<Value> = t
        .cal
        .choices
        .iter()
        .map(|c| {
            json!({
                "night": c.night.to_string(),
                "filter": c.filter,
                "lights": c.n_lights,
                "grade": grade_key(c.grade),
                "reason": c.reason,
                "chosen": c.chosen.as_ref().map(|o| json!({ "label": o.label, "days": o.days, "dust": o.dust })),
                "others": c.others.iter().take(4).map(|o| json!({ "label": o.label, "days": o.days, "dust": o.dust })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let library = match (&t.library, &t.library_error) {
        (Some(l), _) => format!("Flat library: {} sets", l.sets.len()),
        (None, Some(e)) => format!("Flat library unreadable: {e}"),
        (None, None) => "No flat library: only this folder's flats were considered.".into(),
    };
    json!({ "sets": sets, "pairs": pairs, "lightOffsets": t.cal.light_offsets, "choices": choices, "library": library })
}
