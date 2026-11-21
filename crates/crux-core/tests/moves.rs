//! Plan, apply and undo on synthetic frames: rejects go to BAD\<reason>,
//! clashes and changed files stop everything, undo restores every byte.

use crux_core::cache::Cache;
use crux_core::header::Header;
use crux_core::moves::{self, Options};
use crux_core::plan::{self, MoveKind, PlanOptions};
use crux_core::synth::{render, Params};
use crux_core::{fits, pipeline, Config};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

fn write_light(dir: &Path, n: usize, p: &Params) {
    let (img, _) = render(p);
    let minute = 10 * n;
    let (hh, mm) = (21 + minute / 60, minute % 60);
    let mut h = Header::default();
    for (k, v) in [
        ("IMAGETYP", "LIGHT".to_string()),
        ("EXPTIME", "300.0".into()),
        ("FILTER", "H".into()),
        ("OBJECT", "LMC Panel 7".into()),
        ("EGAIN", "0.243".into()),
        ("DATE-LOC", format!("2026-11-20T{hh:02}:{mm:02}:00")),
        ("DATE-OBS", format!("2026-11-20T{:02}:{mm:02}:00", hh - 11)),
    ] {
        h.push(k, v, "");
    }
    let path = dir.join(format!("LIGHT/PANEL_7/2026-11-20_{hh:02}-{mm:02}-00_H_-10.00_300.00s_{n:04}.fits"));
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    fits::write_image(&path, &img, 16, &h).unwrap();
}

/// Relative path → file bytes, for every file outside .crux.
fn snapshot(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for e in walkdir::WalkDir::new(root).into_iter().flatten() {
        let p = e.path();
        if e.file_type().is_file() && !p.components().any(|c| c.as_os_str() == ".crux") {
            out.insert(p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"), std::fs::read(p).unwrap());
        }
    }
    out
}

#[test]
fn plan_apply_undo() {
    let tmp = Tmp(std::env::temp_dir().join(format!("crux-moves-{}", std::process::id())));
    let d = &tmp.0.join("target");
    let base = Params { width: 900, height: 600, n_stars: 350, ..Default::default() };
    for n in 0..6 {
        write_light(d, n, &Params { fwhm: 3.5 + 0.1 * (n % 3) as f64, noise_seed: 100 + n as u64, shift: (n as f64 * 3.0, 0.0), ..base.clone() });
    }
    write_light(d, 6, &Params { fwhm: 7.0, noise_seed: 6, ..base.clone() }); // bad focus
    write_light(d, 7, &Params { trail: Some((0.0, 100.0, 25.0, 60.0, 3.0)), noise_seed: 7, ..base.clone() }); // satellite
    let before = snapshot(d);

    let mut cfg = Config::default();
    cfg.library.use_folder(&std::env::temp_dir().join("crux-test-no-library")); // never the real one
    let cache_dir = tmp.0.join("cache");
    let mut cache = Cache::load(&cache_dir);
    let t = pipeline::run(d, &cfg, &mut cache, true, &|_, _, _| {});
    cache.save().unwrap();
    let p = plan::build(&t, &cfg, &PlanOptions::default(), Some(&cache_dir));
    assert!(p.conflicts.is_empty(), "{:?}", p.conflicts);
    let dests: Vec<String> = p.moves.iter().map(|m| m.to.strip_prefix(d).unwrap().to_string_lossy().replace('\\', "/")).collect();
    assert_eq!(p.moves.len(), 2, "{dests:?}");
    assert!(dests.iter().any(|x| x.starts_with("BAD/focus/") && x.ends_with("_0006.fits")), "{dests:?}");
    assert!(dests.iter().any(|x| x.starts_with("BAD/trail/") && x.ends_with("_0007.fits")), "{dests:?}");
    assert!(p.moves.iter().all(|m| m.kind == MoveKind::Reject));

    // A keep override leaves the soft frame where it is.
    let kept = plan::build(&t, &cfg, &PlanOptions { keep: vec!["*_0006.fits".into()], ..Default::default() }, None);
    assert_eq!(kept.moves.len(), 1);

    // A clash stops everything.
    let clash = &p.moves[0].to;
    std::fs::create_dir_all(clash.parent().unwrap()).unwrap();
    std::fs::write(clash, b"decoy").unwrap();
    let j0 = tmp.0.join("j0.jsonl");
    assert!(moves::apply(&p, None, &j0, &Options::default()).is_err());
    assert!(!j0.exists());
    std::fs::remove_file(clash).unwrap();
    std::fs::remove_dir_all(d.join("BAD")).unwrap();
    assert_eq!(snapshot(d), before);

    // A file changed since planning stops everything.
    let src = &p.moves[1].from;
    let f = std::fs::OpenOptions::new().write(true).open(src).unwrap();
    let old = f.metadata().unwrap().modified().unwrap();
    f.set_modified(old + std::time::Duration::from_secs(5)).unwrap();
    assert!(!moves::check(&p).is_empty());
    f.set_modified(old).unwrap();
    drop(f);
    assert!(moves::check(&p).is_empty(), "{:?}", moves::check(&p));

    // Apply through the copy-and-verify path.
    let j = tmp.0.join("journal.jsonl");
    let out = moves::apply(&p, None, &j, &Options { force_copy: true, ..Default::default() }).unwrap();
    assert_eq!((out.moved, out.copied), (2, 2));
    for m in &p.moves {
        assert!(!m.from.exists() && m.to.exists());
    }
    // The cache followed the files: nothing is re-measured.
    let mut cache = Cache::load(&cache_dir);
    let t2 = pipeline::run(d, &cfg, &mut cache, true, &|_, _, _| {});
    assert_eq!(t2.analysed, 0);
    // Rejects already in BAD aren't planned again.
    assert!(plan::build(&t2, &cfg, &PlanOptions::default(), None).moves.is_empty());

    // Undo restores every byte and removes the folders it made.
    let u = moves::undo(&j, &tmp.0.join("journal.undo.jsonl"), &Options::default()).unwrap();
    assert_eq!(u.moved, 2);
    assert_eq!(snapshot(d), before);
    assert!(!d.join("BAD").exists());
    // A second undo finds the files already back and refuses.
    assert!(moves::undo(&j, &tmp.0.join("again.jsonl"), &Options::default()).is_err());

    // Your decisions (from the GUI): keep the soft frame, reject a good one.
    let mut dec = crux_core::decisions::Decisions::load(&cache_dir).unwrap();
    let soft = p.moves.iter().find(|m| m.to.to_string_lossy().contains("focus")).unwrap().from.clone();
    let good = d_frame(d_path(d_dir(&soft)), "_0002.fits");
    dec.set(&soft, Some(crux_core::grade::Grade::Good));
    dec.set(&good, Some(crux_core::grade::Grade::Reject));
    dec.save().unwrap();
    let mut cache = Cache::load(&cache_dir);
    let mut t3 = pipeline::run(d, &cfg, &mut cache, true, &|_, _, _| {});
    crux_core::decisions::Decisions::load(&cache_dir).unwrap().apply_overrides(&t3.scan.frames, &mut t3.reports);
    let p3 = plan::build(&t3, &cfg, &PlanOptions::default(), Some(&cache_dir));
    let dests: Vec<String> = p3.moves.iter().map(|m| m.to.strip_prefix(d).unwrap().to_string_lossy().replace('\\', "/")).collect();
    assert_eq!(p3.moves.len(), 2, "{dests:?}");
    assert!(dests.iter().any(|x| x.starts_with("BAD/manual/") && x.ends_with("_0002.fits")), "{dests:?}");
    assert!(!dests.iter().any(|x| x.ends_with("_0006.fits")), "{dests:?}");
    // The pinned reject follows its file through apply and undo.
    let j2 = tmp.0.join("j2.jsonl");
    moves::apply(&p3, None, &j2, &Options::default()).unwrap();
    let moved = p3.moves.iter().find(|m| m.from == good).unwrap().to.clone();
    let dd = crux_core::decisions::Decisions::load(&cache_dir).unwrap();
    assert_eq!(dd.get(&moved), Some(crux_core::grade::Grade::Reject));
    assert_eq!(dd.get(&good), None);
    moves::undo(&j2, &tmp.0.join("j2.undo.jsonl"), &Options::default()).unwrap();
    assert_eq!(crux_core::decisions::Decisions::load(&cache_dir).unwrap().get(&good), Some(crux_core::grade::Grade::Reject));
    assert_eq!(snapshot(d), before);

    // Organise with a layout: tokens, a panel correction, `..` out of the folder.
    let mut cfg2 = cfg.clone();
    cfg2.layout.light = "{night}/{panel}/{pier}".into();
    cfg2.layout.reject = "../rejects/{target}/{reason}".into();
    let mut opts = PlanOptions { organise: true, ..Default::default() };
    let first = d_frame(d.join("LIGHT").join("PANEL_7"), "_0000.fits");
    opts.panels.insert(crux_core::cache::key(&first), 13);
    let po = plan::build(&t3, &cfg2, &opts, None);
    assert!(po.conflicts.is_empty(), "{:?}", po.conflicts);
    let dest = |suffix: &str| po.moves.iter().find(|m| m.from.to_string_lossy().ends_with(suffix)).map(|m| m.to.clone()).unwrap();
    assert_eq!(dest("_0000.fits"), d.join("2026-11-20").join("PANEL_13").join(first.file_name().unwrap()));
    assert!(dest("_0001.fits").starts_with(d.join("2026-11-20").join("PANEL_7")));
    let rej = dest("_0007.fits");
    assert!(rej.starts_with(tmp.0.join("rejects").join("LMC").join("trail")), "{}", rej.display());
}

fn d_dir(p: &Path) -> &Path {
    p.parent().unwrap()
}
fn d_path(p: &Path) -> PathBuf {
    p.to_path_buf()
}
/// The frame in `dir` whose name ends with `suffix`.
fn d_frame(dir: PathBuf, suffix: &str) -> PathBuf {
    std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).find(|p| p.to_string_lossy().ends_with(suffix)).unwrap()
}
