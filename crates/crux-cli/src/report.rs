//! Plain-text terminal reports and CSV export.

use anyhow::Result;
use crux_core::classify::Role;
use crux_core::config::Config;
use crux_core::grade::{FrameReport, Grade};
use crux_core::group::Group;
use crux_core::pipeline::{Stage, Triage};
use crux_core::scan::ScanResult;
use crux_core::FrameType;
use std::collections::BTreeMap;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

static LAST: AtomicUsize = AtomicUsize::new(usize::MAX);

pub fn progress(stage: Stage, done: usize, total: usize) {
    if !std::io::stderr().is_terminal() || total == 0 {
        return;
    }
    let step = (total / 200).max(1);
    let key = (stage as usize) << 32 | (done / step);
    if done == total || LAST.swap(key, Ordering::Relaxed) != key {
        let what = match stage {
            Stage::Headers => "Reading headers",
            Stage::Calibration => "Measuring calibration frames",
            Stage::Pixels => "Measuring lights",
        };
        eprint!("\r{what} {done}/{total}      ");
        let _ = std::io::stderr().flush();
    }
}

/// A one-line progress message (terminal only, overwritten in place).
pub fn progress_text(text: &str) {
    if std::io::stderr().is_terminal() {
        eprint!("\r{text:70}");
        let _ = std::io::stderr().flush();
    }
}

pub fn clear_progress() {
    if std::io::stderr().is_terminal() {
        eprint!("\r{:70}\r", "");
    }
}

fn rel(r: &ScanResult, p: &std::path::Path) -> String {
    p.strip_prefix(&r.root).unwrap_or(p).display().to_string()
}

const TYPES: [FrameType; 6] = [FrameType::Light, FrameType::Flat, FrameType::Dark, FrameType::Bias, FrameType::DarkFlat, FrameType::Unknown];

pub fn print_scan(r: &ScanResult, list: bool) {
    let s = &r.stats;
    println!("Scanned {}", r.root.display());
    println!("  {} image files in {:.2} s ({} from cache), {} unreadable", s.files, s.seconds, s.cache_hits, s.errors);
    if !s.skipped_dirs.is_empty() {
        let names: Vec<String> = s.skipped_dirs.iter().map(|d| rel(r, d)).collect();
        println!("  Skipped folders: {}", names.join(", "));
    }

    println!("\nFrames by type");
    println!("  {:<10} {:>6} {:>7} {:>10} {:>16}", "type", "raw", "master", "processed", "in reject folder");
    for t in TYPES {
        let of = |role: Role| r.frames.iter().filter(|f| f.class.frame_type == t && f.class.role == role).count();
        let rej = r.frames.iter().filter(|f| f.class.frame_type == t && f.class.in_reject_dir).count();
        let (a, b, c) = (of(Role::Raw), of(Role::Master), of(Role::Processed));
        if a + b + c > 0 {
            println!("  {:<10} {:>6} {:>7} {:>10} {:>16}", t.name(), a, b, c, rej);
        }
    }

    let conflicts: Vec<_> = r.frames.iter().filter(|f| f.class.has_conflict()).collect();
    if !conflicts.is_empty() {
        println!("\nType sources disagree ({})", conflicts.len());
        for f in conflicts {
            let votes: Vec<String> = f.class.votes.iter().map(|(s, t)| format!("{} says {}", s.name(), t.name())).collect();
            println!("  {}: {} (using {})", rel(r, f.path()), votes.join(", "), f.class.frame_type.name());
        }
    }
    let untyped: Vec<_> = r.frames.iter().filter(|f| f.class.frame_type == FrameType::Unknown).collect();
    if !untyped.is_empty() {
        println!("\nType unknown ({})", untyped.len());
        for f in untyped.iter().take(20) {
            println!("  {}", rel(r, f.path()));
        }
    }
    if !r.errors.is_empty() {
        println!("\nUnreadable files ({})", r.errors.len());
        for (p, e) in &r.errors {
            println!("  {}: {e}", rel(r, p));
        }
    }

    if list {
        println!("\n{:<70} {:<6} {:<9} {:<6} {:>8} {:<19} {:<10}", "file", "type", "role", "filter", "exp", "local time", "night");
        for f in &r.frames {
            let m = &f.meta;
            println!(
                "{:<70} {:<6} {:<9} {:<6} {:>8} {:<19} {:<10}{}",
                rel(r, f.path()),
                f.class.frame_type.name(),
                format!("{:?}", f.class.role).to_lowercase(),
                m.filter.clone().unwrap_or_default(),
                m.exptime.map(|e| format!("{e}")).unwrap_or_default(),
                f.class.local_time.map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()).unwrap_or_default(),
                f.class.night.map(|n| n.to_string()).unwrap_or_default(),
                if f.class.in_reject_dir { "  [reject folder]" } else { "" }
            );
        }
    }
}

pub fn print_analysis_stats(t: &Triage) {
    if t.metrics.is_empty() && t.errors.is_empty() && t.cal_metrics.is_empty() {
        return;
    }
    let per = if t.analysed > 0 {
        let cpu: f32 = t.metrics.values().map(|m| m.seconds).sum::<f32>() / t.metrics.len().max(1) as f32;
        format!(
            "; {} measured in {:.1} s ({:.2} s per frame wall clock, {:.2} s per frame per thread)",
            t.analysed,
            t.analysis_seconds,
            t.analysis_seconds / t.analysed as f64,
            cpu
        )
    } else {
        String::new()
    };
    println!(
        "\nMeasured {} lights ({} from cache){per}",
        t.metrics.len(),
        t.metrics.len() - t.analysed.min(t.metrics.len())
    );
    for (i, e) in &t.errors {
        println!("  couldn't measure {}: {e}", rel(&t.scan, t.scan.frames[*i].path()));
    }
    if !t.cal_metrics.is_empty() || !t.cal_errors.is_empty() {
        let fresh = if t.cal_analysed > 0 { format!("; {} measured in {:.1} s", t.cal_analysed, t.cal_seconds) } else { String::new() };
        println!(
            "Measured {} calibration frames ({} from cache){fresh}",
            t.cal_metrics.len(),
            t.cal_metrics.len().saturating_sub(t.cal_analysed)
        );
        for (i, e) in &t.cal_errors {
            println!("  couldn't measure {}: {e}", rel(&t.scan, t.scan.frames[*i].path()));
        }
    }
}

fn span(g: &Group) -> String {
    match (g.first, g.last) {
        (Some(a), Some(b)) if a.date() == b.date() => format!("{}–{}", a.format("%H:%M"), b.format("%H:%M")),
        (Some(a), Some(b)) => format!("{}–{}", a.format("%d %b %H:%M"), b.format("%d %b %H:%M")),
        _ => String::new(),
    }
}

fn count(g: &Group) -> String {
    let n = g.active_count();
    let mut s = format!("{n} frame{}", if n == 1 { "" } else { "s" });
    if g.n_rejected > 0 {
        s.push_str(&format!(" (+{} in reject folder)", g.n_rejected));
    }
    s
}

fn grade_tag(g: Grade) -> &'static str {
    match g {
        Grade::Good => "good  ",
        Grade::Warn => "WARN  ",
        Grade::Reject => "REJECT",
    }
}

fn opt(v: Option<f32>, prec: usize) -> String {
    v.map(|x| format!("{x:.prec$}")).unwrap_or_else(|| "-".into())
}

fn frame_line(t: &Triage, r: &FrameReport) -> String {
    let f = &t.scan.frames[r.frame];
    let time = f.class.local_time.map(|t| t.format("%H:%M").to_string()).unwrap_or_default();
    let Some(m) = t.metrics.get(&r.frame) else {
        return format!("{time}  {}  (not measured)", grade_tag(r.grade));
    };
    let moon = r
        .sky
        .filter(|s| s.moon_alt > 0.0)
        .map(|s| format!("Moon {:.0}%/{:.0}°", s.moon_illum * 100.0, s.moon_alt))
        .unwrap_or_else(|| "        ".into());
    let tr = r
        .transparency
        .as_ref()
        .map(|t| format!("{:.2}{}", t.value, if t.patchiness.is_some_and(|p| p >= 0.35) { "*" } else { " " }))
        .unwrap_or_else(|| "  -  ".into());
    let trails = if m.trails.is_empty() { "  ".to_string() } else { format!("{}✈", m.trails.len()) };
    format!(
        "{time}  {}  T {tr} {trails} FWHM {:>4} HFR {:>4} ecc {:>4}  stars {:>5}  sky {:>5} e/s ({:>4})  grad {:>3}%  {moon}{}",
        grade_tag(r.grade),
        opt(m.fwhm, 1),
        opt(m.hfr, 1),
        opt(m.ecc, 2),
        m.n_stars,
        r.sky_e_s.map(|v| format!("{v:.3}")).unwrap_or("-".into()),
        r.sky_ratio.map(|v| format!("{v:.1}×")).unwrap_or("-".into()),
        r.gradient_pct.map(|v| format!("{v:.0}")).unwrap_or("-".into()),
        if f.class.in_reject_dir { "  [in reject folder]" } else { "" }
    )
}

pub fn print_report(t: &Triage, _cfg: &Config) {
    let g = &t.grouping;
    let r = &t.scan;
    print_scan(r, false);
    print_analysis_stats(t);

    println!("\nCalibration sets");
    for kind in [FrameType::Flat, FrameType::Dark, FrameType::Bias, FrameType::DarkFlat] {
        let sets: Vec<&Group> = g.by_kind(kind).collect();
        if sets.is_empty() {
            println!("  {}: none", kind.name());
            continue;
        }
        for s in sets {
            let exp = match kind {
                FrameType::Flat if !s.master => s.exptime.map(|e| format!("  median {e:.2} s")).unwrap_or_default(),
                _ => String::new(),
            };
            let n = if s.master { String::new() } else { format!("  {}", count(s)) };
            println!("  {:<62}{n}{exp}  {}", s.label, span(s));
            if let Some(src) = s.built_from {
                println!("      integrated from {}", g.groups[src].label);
            }
        }
    }

    if !t.reports.is_empty() {
        print_grades(t);
    } else if t.cal.sets.is_empty() {
        print_sessions_headers_only(t);
    }
    print_calibration_checks(t);

    println!("\nCalibration matching ({} light groups)", g.light_calibration.len());
    let mut outcomes: BTreeMap<String, usize> = BTreeMap::new();
    for c in &g.light_calibration {
        let darks = c.darks.first().map(|&d| g.groups[d].label.clone()).unwrap_or_else(|| "NONE".into());
        let bias = c.bias.first().map(|&d| g.groups[d].label.clone()).unwrap_or_else(|| "none".into());
        *outcomes.entry(format!("darks: {darks}\n      bias:  {bias}")).or_default() += 1;
    }
    for (k, n) in &outcomes {
        println!("  {n} group(s):\n      {k}");
    }
    let mut flat_use: BTreeMap<String, usize> = BTreeMap::new();
    for c in &g.light_calibration {
        if let Some(f) = c.flats.first() {
            *flat_use.entry(g.groups[f.group].label.clone()).or_default() += 1;
        }
    }
    for (k, n) in &flat_use {
        println!("  flats {k}: nearest set for {n} light group(s)");
    }

    let mut missing: BTreeMap<String, usize> = BTreeMap::new();
    let mut notes: BTreeMap<String, usize> = BTreeMap::new();
    for c in g.light_calibration.iter().chain(&g.flat_calibration) {
        for m in &c.missing {
            *missing.entry(m.clone()).or_default() += 1;
        }
        for m in &c.notes {
            *notes.entry(m.clone()).or_default() += 1;
        }
    }
    if !g.warnings.is_empty() {
        println!("\nWarnings");
        for w in &g.warnings {
            println!("  ⚠ {w}");
        }
    }
    if !missing.is_empty() {
        println!("\nMissing calibration");
        for (m, n) in &missing {
            println!("  ⚠ {m}{}", if *n > 1 { format!("  (×{n})") } else { String::new() });
        }
    }
    if !notes.is_empty() {
        println!("\nAssumptions");
        for (m, n) in &notes {
            println!("  • {m}{}", if *n > 1 { format!("  (×{n})") } else { String::new() });
        }
    }
    if !g.ungrouped.is_empty() {
        let processed = g.ungrouped.iter().filter(|&&i| r.frames[i].class.role == Role::Processed).count();
        println!("\nNot grouped: {} processed products (master lights, references), {} unknown type", processed, g.ungrouped.len() - processed);
    }
    if !t.reports.is_empty() {
        print_versus_rejects(t);
    }
}

fn print_sessions_headers_only(t: &Triage) {
    let g = &t.grouping;
    println!("\nSessions");
    for (night, ids) in &g.sessions {
        let frames: usize = ids.iter().map(|&i| g.groups[i].active_count()).sum();
        println!("  Night of {} ({}): {} light groups, {} frames", night, night.format("%a"), ids.len(), frames);
        for &id in ids {
            let lg = &g.groups[id];
            println!("    {:<44} {:<32} {}", lg.label, count(lg), span(lg));
        }
    }
}

fn tally(rs: &[&FrameReport]) -> String {
    let n = |g: Grade| rs.iter().filter(|r| r.grade == g).count();
    format!("{} good, {} warn, {} reject", n(Grade::Good), n(Grade::Warn), n(Grade::Reject))
}

fn print_grades(t: &Triage) {
    let g = &t.grouping;
    let rep: BTreeMap<usize, &FrameReport> = t.reports.iter().map(|r| (r.frame, r)).collect();
    println!("\nGrades by night");
    for (night, ids) in &g.sessions {
        let all: Vec<&FrameReport> = ids.iter().flat_map(|&id| g.groups[id].members.iter()).filter_map(|i| rep.get(i).copied()).collect();
        let fw: Vec<f64> = all.iter().filter_map(|r| t.metrics.get(&r.frame)?.fwhm.map(f64::from)).collect();
        let ec: Vec<f64> = all.iter().filter_map(|r| t.metrics.get(&r.frame)?.ecc.map(f64::from)).collect();
        println!(
            "\n  Night of {} ({}): {} frames, {}; median FWHM {} px, ecc {}",
            night,
            night.format("%a"),
            all.len(),
            tally(&all),
            crux_core::stats::median_f64(&fw).map(|v| format!("{v:.2}")).unwrap_or("-".into()),
            crux_core::stats::median_f64(&ec).map(|v| format!("{v:.2}")).unwrap_or("-".into()),
        );
        for &id in ids {
            let lg = &g.groups[id];
            println!("    {}", lg.label.rsplit_once(" · ").map(|(a, _)| a).unwrap_or(&lg.label));
            for i in &lg.members {
                let Some(r) = rep.get(i) else { continue };
                println!("      {}", frame_line(t, r));
                for fl in r.flags.iter().filter(|f| f.grade != Grade::Good) {
                    println!("             ↳ {}: {}", fl.check.name(), fl.reason);
                }
                for n in &r.notes {
                    println!("             · {n}");
                }
            }
        }
    }
}

fn print_calibration_checks(t: &Triage) {
    let g = &t.grouping;
    let c = &t.cal;
    if c.sets.is_empty() {
        // Flats can still come from the library.
        print_flat_choices(t);
        return;
    }
    let rep: BTreeMap<usize, &FrameReport> = c.reports.iter().map(|r| (r.frame, r)).collect();
    println!("\nCalibration checks");
    for kind in [FrameType::Bias, FrameType::Dark, FrameType::DarkFlat, FrameType::Flat] {
        for s in c.sets.iter().filter(|s| g.groups[s.group].kind == kind) {
            let gr = &g.groups[s.group];
            let mut facts: Vec<String> = Vec::new();
            if !gr.master {
                facts.push(format!("{} measured", s.measured));
            }
            match kind {
                FrameType::Flat => {
                    if let Some(l) = s.flat_level {
                        facts.push(format!("level {:.0}%", 100.0 * l));
                    }
                    if let Some(w) = s.worst_change {
                        facts.push(format!("worst change {w:.1}%"));
                    }
                    if s.n_saturated > 0 {
                        facts.push(format!("{} saturated", s.n_saturated));
                    }
                }
                _ => {
                    if let Some(l) = s.level_adu {
                        facts.push(format!("level {l:.1} ADU"));
                    }
                    if let Some(n) = s.noise_adu {
                        facts.push(format!("noise {n:.2} ADU"));
                    }
                    if let (Some(a), Some(e)) = (s.read_noise_adu, s.read_noise_e) {
                        facts.push(format!("read noise {a:.2} ADU ({e:.2} e-, frame pairs)"));
                    }
                    if let (Some(r), Some(c)) = (s.row_noise_adu, s.col_noise_adu) {
                        facts.push(format!("banding rows {r:.2} / columns {c:.2} ADU"));
                    }
                    if kind != FrameType::Bias {
                        if let Some(h) = s.hot {
                            facts.push(format!("hot {:.3}%", 100.0 * h));
                        }
                    }
                }
            }
            println!("\n  {:<62} {}  {}", gr.label, grade_tag(s.grade), facts.join(", "));
            if let Some(v) = &s.vs_lights {
                let nights = match (v.first_night, v.last_night) {
                    (Some(a), Some(b)) if a == b => format!("night of {a}"),
                    (Some(a), Some(b)) => format!("nights {a} to {b}"),
                    _ => String::new(),
                };
                println!(
                    "      vs {} lights ({nights}): {:.1}% error{}  [{}]",
                    v.n_lights,
                    v.error.spread,
                    format!(
                        "{}{}",
                        v.dust.map(|d| format!(", dust {d:.2}")).unwrap_or_default(),
                        v.halves_diff.map(|d| format!(", halves differ {d:.1}%")).unwrap_or_default()
                    ),
                    if v.reliable { v.grade.name().to_uppercase() } else { "UNRELIABLE".into() }
                );
                let e = &v.error.edges;
                println!(
                    "        edges: left {}  right {}  top {}  bottom {};  corners {} {} {} {}",
                    crux_core::calib::pct(e.left),
                    crux_core::calib::pct(e.right),
                    crux_core::calib::pct(e.top),
                    crux_core::calib::pct(e.bottom),
                    crux_core::calib::pct(e.tl),
                    crux_core::calib::pct(e.tr),
                    crux_core::calib::pct(e.bl),
                    crux_core::calib::pct(e.br)
                );
                if let (Some(u), Some(n), Some(nu)) = (v.untilted, v.no_flat, v.no_flat_untilted) {
                    println!("        without the tilt {u:.1}%;  with no flat {n:.1}% ({nu:.1}% without the tilt)");
                }
                if let Some(e) = &v.no_flat_edges {
                    println!(
                        "        no flat: left {}  right {}  top {}  bottom {};  corners {} {} {} {}",
                        crux_core::calib::pct(e.left),
                        crux_core::calib::pct(e.right),
                        crux_core::calib::pct(e.top),
                        crux_core::calib::pct(e.bottom),
                        crux_core::calib::pct(e.tl),
                        crux_core::calib::pct(e.tr),
                        crux_core::calib::pct(e.bl),
                        crux_core::calib::pct(e.br)
                    );
                }
            }
            if let Some(o) = &s.offset {
                println!("      offset: {o}");
            }
            for f in s.flags.iter().filter(|f| f.grade != Grade::Good) {
                println!("      ↳ {}: {}", f.check.name(), f.reason);
            }
            for n in &s.notes {
                println!("      · {n}");
            }
            // Frames with their own problems (set-level reasons are shown above).
            let own: Vec<(&FrameReport, Vec<&crux_core::grade::Flag>)> = gr
                .members
                .iter()
                .filter_map(|i| rep.get(i).copied())
                .map(|r| (r, r.flags.iter().filter(|f| f.grade != Grade::Good && !matches!(f.check, crux_core::grade::Check::FlatMatch | crux_core::grade::Check::Library)).collect::<Vec<_>>()))
                .filter(|(_, f)| !f.is_empty())
                .collect();
            for (r, fl) in own.iter().take(12) {
                let f = &t.scan.frames[r.frame];
                let why: Vec<&str> = fl.iter().map(|f| f.reason.as_str()).collect();
                println!("        {}  {}  {}", f.meta.file_name(), grade_tag(fl.iter().map(|f| f.grade).max().unwrap_or(Grade::Good)), why.join("; "));
            }
            if own.len() > 12 {
                println!("        … and {} more", own.len() - 12);
            }
        }
    }
    if !c.pairs.is_empty() {
        println!("\n  Flat sets compared (smoothed ratio, % peak-to-peak)");
        let short = |id: usize| {
            let gr = &g.groups[id];
            if gr.master {
                format!("master {}", t.scan.frames[gr.members[0]].meta.file_name())
            } else {
                format!("{}{}", gr.night.map(|n| n.to_string()).unwrap_or_default(), if gr.retired { " (retired)" } else { "" })
            }
        };
        for p in &c.pairs {
            let shape = p.edges.describe(0.5);
            println!(
                "    {} {:<22} vs {:<52} {:>5.1}%  dust {}{}{}",
                g.groups[p.a].filter.clone().unwrap_or_default(),
                short(p.a),
                short(p.b),
                p.spread,
                p.dust.map(|d| format!("{d:.2}")).unwrap_or("-".into()),
                if p.combinable { "  can combine" } else { "" },
                if shape.is_empty() { String::new() } else { format!("  ({shape})") }
            );
        }
    }
    if !c.light_offsets.is_empty() {
        println!("\n  Lights offset for the flat checks: {}", c.light_offsets.join(", "));
    }
    print_flat_choices(t);
}

fn print_flat_choices(t: &Triage) {
    let c = &t.cal;
    if c.choices.is_empty() {
        return;
    }
    let lib = match (&t.library, &t.library_error) {
        (Some(l), _) => format!("library {} sets", l.sets.len()),
        (None, Some(e)) => format!("library unreadable: {e}"),
        (None, None) => "no flat library".into(),
    };
    println!("\nFlat matching (nearest good set, confirmed by dust; {lib})");
    for ch in &c.choices {
        println!(
            "  {} {} · {} lights  {}  {}",
            ch.night,
            ch.filter,
            ch.n_lights,
            grade_tag(ch.grade),
            ch.reason
        );
        let others: Vec<String> = ch
            .others
            .iter()
            .take(4)
            .map(|o| format!("{} ({:+} nights{})", o.label, o.days, o.dust.map(|d| format!(", dust {d:.2}")).unwrap_or_default()))
            .collect();
        if !others.is_empty() {
            println!("      also: {}", others.join("; "));
        }
    }
}

/// The library index: sets, optical trains, and sets that can be combined.
/// `crux library targets`: one line per target, then calibration-only
/// folders, then what could be combined.
pub fn print_targets(lt: &crux_core::targets::LibraryTargets) {
    println!("Library folders: {}", lt.roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", "));
    for n in &lt.notes {
        println!("  note: {n}");
    }
    println!("\n{} targets in {} folders", lt.targets.len(), lt.units.len());
    for t in &lt.targets {
        let filters: Vec<String> = t.secs.iter().map(|(f, s)| format!("{f} {:.1}h", s / 3600.0)).collect();
        let span = match (t.nights.first(), t.nights.last()) {
            (Some(a), Some(b)) if a != b => format!("{a} … {b}"),
            (Some(a), _) => a.to_string(),
            _ => String::new(),
        };
        let panels = t.panels();
        println!(
            "  {:<20} {:>5.1} h  {:>4} lights  {:>2} night{}  {}{}  [{}]",
            t.name,
            t.hours(),
            t.lights,
            t.nights.len(),
            if t.nights.len() == 1 { " " } else { "s" },
            span,
            if panels > 1 { format!("  {panels} panels") } else { String::new() },
            filters.join(" ")
        );
        if let Some((ra, dec)) = t.centre {
            println!("  {:<20} RA {:.3}° Dec {:+.3}°", "", ra, dec);
        }
        for &u in &t.units {
            let unit = &lt.units[u];
            println!("  {:<20} in {}", "", unit.name);
        }
        for n in &t.notes {
            println!("  {:<20} note: {n}", "");
        }
    }
    let cal: Vec<&crux_core::targets::Unit> = lt.units.iter().filter(|u| u.lights() == 0 && u.calibration() > 0).collect();
    if !cal.is_empty() {
        println!("\nCalibration only ({} folders):", cal.len());
        for u in cal {
            let mut what = Vec::new();
            for (n, w) in [(u.flats, "flats"), (u.darks, "darks"), (u.bias, "bias"), (u.dark_flats, "dark flats")] {
                if n > 0 {
                    what.push(format!("{n} {w}"));
                }
            }
            println!("  {:<36} {}", u.name, what.join(", "));
        }
    }
    let comb = lt.combinable();
    if !comb.is_empty() {
        println!("\nIn several folders (crux library combine \"<target>\"):");
        for (ti, c) in comb {
            println!("  {:<20} {} folders → {}", lt.targets[ti].name, c.moving.len() + c.existing.is_some() as usize, c.dest.display());
            for (u, why) in &c.blocked {
                println!("  {:<20}   {} stays: {why}", "", lt.units[*u].name);
            }
        }
    }
}

pub fn print_library(lib: &crux_core::library::Library, cfg: &Config) {
    let pairs = lib.pairs(cfg);
    let cons = lib.consensus(&pairs, cfg);
    println!("\nFlat library in {} ({} sets)", crux_core::library::Library::db_path(&lib.dir).display(), lib.sets.len());
    for s in &lib.sets {
        let mut facts = vec![format!("{} frame{}", s.frames, if s.frames == 1 { "" } else { "s" })];
        if let Some(e) = s.exposure_s {
            facts.push(format!("{e:.2} s"));
        }
        if let Some(l) = s.level {
            facts.push(format!("level {:.0}%", 100.0 * l));
        }
        if let Some(w) = s.worst_change {
            facts.push(format!("change {w:.1}%"));
        }
        if let Some(e) = s.error_vs_lights {
            facts.push(format!("{e:.1}% vs lights"));
        }
        let c = cons.get(&s.id);
        if let Some(c) = c {
            facts.push(format!("{:.1}% from {} same-dust sets", c.median, c.n));
        }
        let grade = s.grade.max(c.map(|c| c.grade).unwrap_or(Grade::Good));
        println!("  {:<30} {}  {}", s.label(), grade_tag(grade), facts.join(", "));
        for r in &s.reasons {
            println!("      ↳ {r}");
        }
        if let Some(c) = c.filter(|c| c.grade != Grade::Good) {
            println!("      ↳ {}", c.reason());
        }
    }
    let label = |id: &str| lib.sets.iter().find(|s| s.id == id).map(|s| s.label()).unwrap_or_else(|| id.to_string());
    let groups = lib.combinable_groups(&pairs);
    if !groups.is_empty() {
        println!("\n  Can be combined (every pair within {:.1}% and the same dust)", cfg.thresholds.calibration.combine_pct);
        for g in &groups {
            println!("    {}", g.iter().map(|id| label(id)).collect::<Vec<_>>().join(", "));
        }
    }
    let trains = lib.trains(&pairs, cfg);
    println!("\n  Optical trains by dust (correlation ≥ {:.2})", cfg.thresholds.calibration.dust_same_train);
    for tr in &trains {
        println!("    {}", tr.iter().map(|id| label(id)).collect::<Vec<_>>().join(", "));
    }
    let homes: Vec<&crux_core::library::LibrarySet> =
        lib.sets.iter().filter(|s| s.files.first().and_then(|p| p.parent()) != Some(s.home.as_path())).collect();
    if !homes.is_empty() {
        println!("\n  {} sets are used where they are (`crux plan <dir> --to-library` can file them under {})", homes.len(), lib.root.display());
    }
    let offsets: Vec<&crux_core::library::CalFolder> = lib.folders.iter().filter(|f| f.has_offsets()).collect();
    if !offsets.is_empty() {
        println!("\n  Darks, bias and dark flats are looked for in {} folder{}", offsets.len(), if offsets.len() == 1 { "" } else { "s" });
        for f in offsets {
            println!("    {} ({} darks, {} bias, {} dark flats)", f.folder.display(), f.darks, f.bias, f.dark_flats);
        }
    }
    if let Some(r) = &lib.refreshed {
        println!("\n  Refreshed {r} from {}", lib.searched.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", "));
    }
}

fn print_versus_rejects(t: &Triage) {
    let in_rej: Vec<&FrameReport> = t.reports.iter().filter(|r| t.scan.frames[r.frame].class.in_reject_dir).collect();
    let kept: Vec<&FrameReport> = t.reports.iter().filter(|r| !t.scan.frames[r.frame].class.in_reject_dir).collect();
    println!("\nYour rejects vs crux");
    println!("  In a reject folder ({}): {}", in_rej.len(), tally(&in_rej));
    println!("  Everywhere else ({}): {}", kept.len(), tally(&kept));
    let show = |title: &str, rs: Vec<&&FrameReport>| {
        if rs.is_empty() {
            return;
        }
        println!("\n  {title} ({})", rs.len());
        for r in rs {
            let f = &t.scan.frames[r.frame];
            let when = f.class.local_time.map(|t| t.format("%Y-%m-%d %H:%M").to_string()).unwrap_or_default();
            let why = if r.summary().is_empty() { r.notes.join("; ") } else { r.summary() };
            println!("    {when}  {:<14} {}  {}", f.meta.object.clone().unwrap_or_default(), grade_tag(r.grade), why);
        }
    };
    show("In a reject folder but crux grades good", in_rej.iter().filter(|r| r.grade == Grade::Good).collect());
    show("Not in a reject folder but crux rejects", kept.iter().filter(|r| r.grade == Grade::Reject).collect());
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

pub fn write_csv(t: &Triage, _cfg: &Config, path: &std::path::Path) -> Result<()> {
    let mut out = String::from(
        "file,night,local_time,object,filter,exposure,in_reject_folder,grade,transparency,patchiness,matched,flipped,trails,trail_max_adu,fwhm_px,hfr_px,ecc,pa_deg,coherence,radial,stars,sky_adu,sky_e_s,sky_ratio,gradient_pct,fwhm_ratio,sun_alt,moon_alt,moon_illum,moon_sep,cloudcvr,mpsas,airmass,focpos,foctemp,reasons,notes\n",
    );
    let o = |v: Option<f64>, p: usize| v.map(|x| format!("{x:.p$}")).unwrap_or_default();
    for r in &t.reports {
        let f = &t.scan.frames[r.frame];
        let m = t.metrics.get(&r.frame);
        let mf = |g: &dyn Fn(&crux_core::analysis::LightMetrics) -> Option<f32>, p: usize| o(m.and_then(g).map(f64::from), p);
        let row = [
            csv_field(&rel(&t.scan, f.path())),
            f.class.night.map(|n| n.to_string()).unwrap_or_default(),
            f.class.local_time.map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()).unwrap_or_default(),
            csv_field(&f.meta.object.clone().unwrap_or_default()),
            f.meta.filter.clone().unwrap_or_default(),
            o(f.meta.exptime, 2),
            f.class.in_reject_dir.to_string(),
            r.grade.name().into(),
            o(r.transparency.as_ref().map(|t| t.value), 3),
            o(r.transparency.as_ref().and_then(|t| t.patchiness), 3),
            r.transparency.as_ref().map(|t| t.n_matched.to_string()).unwrap_or_default(),
            r.transparency.as_ref().map(|t| t.flipped.to_string()).unwrap_or_default(),
            m.map(|m| m.trails.len().to_string()).unwrap_or_default(),
            mf(&|m| m.trails.iter().map(|t| t.brightness_adu).reduce(f32::max), 1),
            mf(&|m| m.fwhm, 2),
            mf(&|m| m.hfr, 2),
            mf(&|m| m.ecc, 3),
            mf(&|m| m.pa, 1),
            mf(&|m| m.coherence, 3),
            mf(&|m| m.radial, 3),
            m.map(|m| m.n_stars.to_string()).unwrap_or_default(),
            mf(&|m| Some(m.sky_adu), 1),
            o(r.sky_e_s, 3),
            o(r.sky_ratio, 2),
            o(r.gradient_pct, 1),
            o(r.fwhm_ratio, 2),
            o(r.sky.map(|s| s.sun_alt), 1),
            o(r.sky.map(|s| s.moon_alt), 1),
            o(r.sky.map(|s| s.moon_illum), 3),
            o(r.sky.and_then(|s| s.moon_sep), 1),
            o(f.meta.cloudcvr, 0),
            o(f.meta.mpsas, 2),
            o(f.meta.airmass, 3),
            o(f.meta.focpos, 0),
            o(f.meta.foctemp, 2),
            csv_field(&r.summary()),
            csv_field(&r.notes.join("; ")),
        ];
        out.push_str(&row.join(","));
        out.push('\n');
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// One row per bias, dark and flat (raw and master).
pub fn write_cal_csv(t: &Triage, path: &std::path::Path) -> Result<()> {
    let g = &t.grouping;
    let mut out = String::from(
        "file,type,master,set,night,local_time,filter,exposure,in_reject_folder,grade,set_grade,level_adu,noise_adu,hot_pct,structure_adu,saturated_pct,flat_change_pct,ccd_temp,set_temp,reasons\n",
    );
    let o = |v: Option<f64>, p: usize| v.map(|x| format!("{x:.p$}")).unwrap_or_default();
    let set_of: BTreeMap<usize, usize> = g.groups.iter().flat_map(|gr| gr.members.iter().map(move |&m| (m, gr.id))).collect();
    for r in &t.cal.reports {
        let f = &t.scan.frames[r.frame];
        let gid = set_of.get(&r.frame).copied();
        let set = gid.and_then(|id| t.cal.set(id));
        let m = t.cal_metrics.get(&r.frame);
        let mf = |get: &dyn Fn(&crux_core::calib::CalMetrics) -> f32, k: f64, p: usize| o(m.map(|m| get(m) as f64 * k), p);
        let row = [
            csv_field(&rel(&t.scan, f.path())),
            f.class.frame_type.name().into(),
            (f.class.role == Role::Master).to_string(),
            csv_field(&gid.map(|id| g.groups[id].label.clone()).unwrap_or_default()),
            f.class.night.map(|n| n.to_string()).unwrap_or_default(),
            f.class.local_time.map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()).unwrap_or_default(),
            f.meta.filter.clone().unwrap_or_default(),
            o(f.meta.exptime, 3),
            f.class.in_reject_dir.to_string(),
            r.grade.name().into(),
            set.map(|s| s.grade.name().to_string()).unwrap_or_default(),
            mf(&|m| m.level_adu, 1.0, 2),
            mf(&|m| m.noise_adu, 1.0, 3),
            mf(&|m| m.hot, 100.0, 4),
            mf(&|m| m.structure_adu, 1.0, 2),
            mf(&|m| m.saturated, 100.0, 4),
            o(set.and_then(|s| s.changes.get(&r.frame).copied()), 2),
            o(f.meta.ccd_temp, 1),
            o(f.meta.set_temp, 1),
            csv_field(&r.summary()),
        ];
        out.push_str(&row.join(","));
        out.push('\n');
    }
    std::fs::write(path, out)?;
    Ok(())
}

pub fn bytes(n: u64) -> String {
    match n {
        n if n >= 1 << 30 => format!("{:.1} GB", n as f64 / (1u64 << 30) as f64),
        n if n >= 1 << 20 => format!("{:.0} MB", n as f64 / (1u64 << 20) as f64),
        n => format!("{:.0} KB", n as f64 / 1024.0),
    }
}

/// A plan, grouped by kind and destination folder.
pub fn print_plan(p: &crux_core::plan::Plan) {
    use crux_core::plan::MoveKind;
    let rel = |path: &std::path::Path| path.strip_prefix(&p.root).map(|r| r.display().to_string()).unwrap_or_else(|_| path.display().to_string());
    println!("\nMove plan for {} (made {})", p.root.display(), p.created);
    for kind in [MoveKind::Reject, MoveKind::Retire, MoveKind::Organise, MoveKind::Library, MoveKind::Combine] {
        let ms: Vec<&crux_core::plan::Move> = p.moves.iter().filter(|m| m.kind == kind).collect();
        if ms.is_empty() {
            continue;
        }
        let size: u64 = ms.iter().map(|m| m.size).sum();
        println!("\n  {} — {} files, {}", kind.name(), ms.len(), bytes(size));
        let mut by_dir: BTreeMap<String, Vec<&crux_core::plan::Move>> = BTreeMap::new();
        for m in &ms {
            by_dir.entry(rel(m.to.parent().unwrap_or(&m.to))).or_default().push(m);
        }
        for (dir, v) in &by_dir {
            println!("    → {dir}\\  ({})", v.len());
            for m in v.iter().take(8) {
                println!("        {}  ← {}", m.to.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), m.reason);
            }
            if v.len() > 8 {
                println!("        … and {} more", v.len() - 8);
            }
        }
    }
    if !p.combine.is_empty() {
        println!("\n  Library sets combined in the index: {}", p.combine.iter().map(|g| g.join(" + ")).collect::<Vec<_>>().join("; "));
    }
    for n in &p.notes {
        println!("  · {n}");
    }
    if !p.conflicts.is_empty() {
        println!("\n  Clashes ({}), so this plan can't be applied:", p.conflicts.len());
        for c in &p.conflicts {
            println!("    ✗ {c}");
        }
    }
    println!("\n  {} moves, {} in total", p.moves.len(), bytes(p.total_bytes()));
}
