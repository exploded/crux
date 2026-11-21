//! Flat matching with dust fingerprints, and the flat library, on
//! synthetic frames: two flat sets share their dust (can be combined), a
//! later one has the dust moved (optical train changed).

use crux_core::cache::Cache;
use crux_core::grade::Grade;
use crux_core::header::Header;
use crux_core::image::Image;
use crux_core::library::{self, Library};
use crux_core::synth::Rng;
use crux_core::{fits, pipeline, Config};
use std::path::{Path, PathBuf};

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

const W: usize = 1600;
const H: usize = 1056;

fn vignetting(x: f64, y: f64) -> f64 {
    let (u, v) = (x / W as f64 - 0.5, y / H as f64 - 0.5);
    1.0 - 0.8 * (u * u + v * v)
}

/// Dust shadows: 3% dips, 50 px sigma, at fixed places.
fn dust(spots: &[(f64, f64)], x: f64, y: f64) -> f64 {
    let mut v = 1.0;
    for &(sx, sy) in spots {
        v *= 1.0 - 0.03 * (-((x - sx).powi(2) + (y - sy).powi(2)) / (2.0 * 50.0 * 50.0)).exp();
    }
    v
}

fn spots(seed: u64) -> Vec<(f64, f64)> {
    let mut r = Rng::new(seed);
    (0..10).map(|_| (200.0 + r.uniform() * (W as f64 - 400.0), 200.0 + r.uniform() * (H as f64 - 400.0))).collect()
}

fn frame(seed: u64, noise: f64, f: impl Fn(usize, usize) -> f64) -> Image {
    let mut rng = Rng::new(seed);
    let mut img = Image::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let v = (f(x, y) + noise * rng.gauss()).round().clamp(0.0, 65535.0);
            img.data[y * W + x] = (v / 65535.0) as f32;
        }
    }
    img
}

fn write(dir: &Path, rel: &str, img: &Image, kv: &[(&str, String)]) {
    let mut h = Header::default();
    for (k, v) in kv {
        h.push(*k, v.clone(), "");
    }
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    fits::write_image(&path, img, 16, &h).unwrap();
}

fn kv(kind: &str, exp: f64, date: &str) -> Vec<(&'static str, String)> {
    vec![
        ("IMAGETYP", kind.to_string()),
        ("EXPTIME", format!("{exp}")),
        ("GAIN", "100".into()),
        ("OFFSET", "50".into()),
        ("FILTER", "H".into()),
        ("OBJECT", "Test".into()),
        ("DATE-LOC", date.to_string()),
    ]
}

fn lights(d: &Path, dust_spots: &[(f64, f64)], day: u32) {
    for n in 0..8u64 {
        let mut rng = Rng::new(900 + n);
        let stars: Vec<(f64, f64)> = (0..60).map(|_| (rng.uniform() * W as f64, rng.uniform() * H as f64)).collect();
        let img = frame(950 + n, 8.0, |x, y| {
            let (fx, fy) = (x as f64, y as f64);
            let mut v = 500.0 + 600.0 * vignetting(fx, fy) * dust(dust_spots, fx, fy);
            for &(sx, sy) in &stars {
                let r2 = (fx - sx).powi(2) + (fy - sy).powi(2);
                if r2 < 50.0 {
                    v += 3000.0 * (-r2 / 4.0).exp();
                }
            }
            v
        });
        write(d, &format!("LIGHT/L_{n:04}.fits"), &img, &kv("LIGHT", 300.0, &format!("2026-11-{day}T22:{:02}:00", n * 5)));
    }
}

#[test]
fn dust_confirms_flats_and_the_library_remembers() {
    let tmp = Tmp(std::env::temp_dir().join(format!("crux-lib-{}", std::process::id())));
    let d = &tmp.0.join("night");
    // The moved dust: every spot shifted by half the usable area.
    let p1 = spots(1);
    let (aw, ah) = (W as f64 - 400.0, H as f64 - 400.0);
    let p2: Vec<(f64, f64)> = p1.iter().map(|&(x, y)| (200.0 + (x - 200.0 + aw / 2.0) % aw, 200.0 + (y - 200.0 + ah / 2.0) % ah)).collect();
    for n in 0..4u64 {
        write(d, &format!("bias/B_{n:04}.fits"), &frame(n, 4.0, |_, _| 500.0), &kv("BIAS", 0.0, &format!("2026-11-20T20:00:{n:02}")));
    }
    // A (19 Nov) and B (21 Nov): same dust, B's vignetting 1% different. C (23 Nov): dust moved.
    let sets: [(&str, u32, &[(f64, f64)], f64); 3] = [("A", 20, &p1, 0.0), ("B", 22, &p1, 0.01), ("C", 24, &p2, 0.0)];
    for (name, day, sp, tilt) in sets {
        for n in 0..5u64 {
            let img = frame(day as u64 * 100 + n, 30.0, |x, y| {
                let (fx, fy) = (x as f64, y as f64);
                500.0 + 25000.0 * vignetting(fx, fy) * dust(sp, fx, fy) * (1.0 + tilt * (fx / W as f64 - 0.5))
            });
            write(d, &format!("FLAT/{name}/F_{name}_{n:04}.fits"), &img, &kv("FLAT", 1.0, &format!("2026-11-{day}T06:00:{n:02}")));
        }
    }
    lights(d, &p1, 24); // night of 24 Nov: C is nearest, but A and B have the right dust.

    let mut cfg = Config::default();
    cfg.library.use_folder(&tmp.0.join("library"));
    let t = pipeline::run(d, &cfg, &mut Cache::disabled(), true, &|_, _, _| {});
    assert!(t.library.is_none());
    let label = |gid: usize| t.grouping.groups[gid].label.clone();
    let set_named = |name: &str| {
        t.grouping
            .groups
            .iter()
            .find(|g| g.kind == crux_core::classify::FrameType::Flat && t.scan.frames[g.members[0]].path().to_string_lossy().contains(&format!("FLAT{}{name}", std::path::MAIN_SEPARATOR)))
            .unwrap()
            .id
    };
    let (a, b, c) = (set_named("A"), set_named("B"), set_named("C"));

    // Pairs: A–B share dust and agree; C's dust differs.
    let pair = |x: usize, y: usize| t.cal.pairs.iter().find(|p| (p.a, p.b) == (x, y) || (p.a, p.b) == (y, x)).unwrap();
    assert!(pair(a, b).combinable, "{:?}", pair(a, b));
    assert!(pair(a, b).dust.unwrap() > 0.9, "{:?}", pair(a, b));
    assert!(pair(a, c).dust.unwrap() < 0.3 && !pair(a, c).combinable, "{:?}", pair(a, c));

    // The lights' night: nearest is C, whose dust doesn't match; A or B does.
    assert_eq!(t.cal.choices.len(), 1, "{:?}", t.cal.choices);
    let ch = &t.cal.choices[0];
    let chosen = ch.chosen.as_ref().unwrap();
    assert_eq!(chosen.group, Some(c), "chose {} — {}", chosen.label, ch.reason);
    assert_eq!(ch.grade, Grade::Warn, "{}", ch.reason);
    assert!(chosen.dust.unwrap() < 0.15, "{}", ch.reason);
    assert!(ch.reason.contains("optical train") && (ch.reason.contains(&label(a)) || ch.reason.contains(&label(b))), "{}", ch.reason);
    for o in &ch.others {
        assert!(o.dust.unwrap() > 0.3, "{o:?}");
    }

    // Library: add the three sets, reload, check trains and combinable pairs.
    let root = &cfg.library.root;
    let mut lib = Library::new(&cfg.library);
    for (set, map) in library::sets_from_triage(&t, root) {
        assert!(lib.upsert(set, &map).unwrap());
    }
    lib.save().unwrap();
    let lib = Library::load(&cfg.library).unwrap().unwrap();
    assert_eq!(lib.sets.len(), 3);
    let home = &lib.sets[0].home;
    assert!(home.starts_with(root.join("H")), "{}", home.display());
    let pairs = lib.pairs(&cfg);
    assert_eq!(pairs.iter().filter(|p| p.combinable).count(), 1, "{pairs:?}");
    let trains = lib.trains(&pairs, &cfg);
    assert_eq!(trains.len(), 2, "{trains:?}");

    // Combining A and B changes the library; undo puts it back, maps too.
    let before = lib.references();
    let t = pipeline::run(d, &cfg, &mut Cache::disabled(), true, &|_, _, _| {});
    let opts = crux_core::plan::PlanOptions { combine: true, rejects: false, retire: false, ..Default::default() };
    let p = crux_core::plan::build(&t, &cfg, &opts, None);
    assert_eq!(p.combine.len(), 1, "{:?}", p.notes);
    let j = tmp.0.join("combine.jsonl");
    crux_core::moves::apply(&p, None, &j, &Default::default()).unwrap();
    assert_eq!(Library::load(&cfg.library).unwrap().unwrap().sets.len(), 2);
    crux_core::moves::undo(&j, &tmp.0.join("combine.undo.jsonl"), &Default::default()).unwrap();
    let lib = Library::load(&cfg.library).unwrap().unwrap();
    assert_eq!(lib.sets.len(), 3);
    assert_eq!(lib.references(), before);

    // A folder of lights only: flats come from the library.
    let d2 = &tmp.0.join("lights-only");
    lights(d2, &p1, 25);
    let t2 = pipeline::run(d2, &cfg, &mut Cache::disabled(), true, &|_, _, _| {});
    assert!(t2.library.is_some());
    let ch = &t2.cal.choices[0];
    let chosen = ch.chosen.as_ref().unwrap();
    assert!(chosen.library_id.as_deref().is_some_and(|id| id.starts_with("H_2026-11-23")), "{} — {}", chosen.label, ch.reason);
    assert_eq!(ch.grade, Grade::Warn, "{}", ch.reason);
    assert!(ch.reason.contains("library H 2026-11-2"), "{}", ch.reason);
}

/// Refresh: flats, darks and bias found anywhere under a root, in place;
/// a folder of lights alone gets library flats and finds the bias; sets
/// whose files go away leave the library.
#[test]
fn refresh_finds_calibration_frames_in_every_folder() {
    use std::sync::atomic::AtomicBool;
    let tmp = Tmp(std::env::temp_dir().join(format!("crux-refresh-{}", std::process::id())));
    let root = tmp.0.join("data");
    let p1 = spots(1);
    let flats = root.join("2026-11-20 FLATS");
    for (name, day) in [("A", 20u32), ("B", 22)] {
        for n in 0..5u64 {
            let img = frame(day as u64 * 100 + n, 30.0, |x, y| 500.0 + 25000.0 * vignetting(x as f64, y as f64) * dust(&p1, x as f64, y as f64));
            write(&flats, &format!("FLAT/{name}/F_{n:04}.fits"), &img, &kv("FLAT", 1.0, &format!("2026-11-{day}T06:00:{n:02}")));
        }
    }
    let darks = root.join("calibration");
    for n in 0..4u64 {
        write(&darks, &format!("bias/B_{n:04}.fits"), &frame(n, 4.0, |_, _| 500.0), &kv("BIAS", 0.0, &format!("2026-11-20T20:00:{n:02}")));
    }
    let image = root.join("2026-11-23 Test");
    lights(&image, &p1, 23);
    write(&root, "loose.fits", &frame(7, 4.0, |_, _| 500.0), &kv("BIAS", 0.0, "2026-11-20T20:10:00"));

    let mut cfg = Config::default();
    cfg.library.use_folder(&tmp.0.join("library"));
    let roots = [root.clone()];
    let caches = tmp.0.join("caches");
    let rep = library::refresh(&cfg, &roots, Some(&caches), &|_| {}, &AtomicBool::new(false)).unwrap();
    assert_eq!(rep.folders, 3, "{rep:?}");
    assert_eq!(rep.added.len(), 2, "{rep:?}");
    assert_eq!(rep.loose, vec![(root.clone(), 1)]);
    let lib = Library::load(&cfg.library).unwrap().unwrap();
    assert_eq!(lib.sets.len(), 2);
    assert!(lib.sets.iter().all(|s| s.files.iter().all(|f| f.starts_with(&flats))), "files stay where they are");
    let offsets = library::offset_folders(&cfg);
    assert_eq!(offsets, vec![darks.clone()], "{:?}", lib.folders);

    // The lights' folder has no flats or bias of its own (flats at dawn on
    // 22 Nov belong to the night of the 21st).
    let t = pipeline::run(&image, &cfg, &mut Cache::disabled(), true, &|_, _, _| {});
    let ch = &t.cal.choices[0];
    let chosen = ch.chosen.as_ref().unwrap();
    assert!(chosen.library_id.as_deref().is_some_and(|id| id.starts_with("H_2026-11-21")), "{} — {}", chosen.label, ch.reason);
    let others = crux_core::fingerprint::other_folders(&image, &cfg, &[]);
    assert!(others.iter().any(|(d, f)| d == &darks && f.len() == 4), "{:?}", others.iter().map(|o| &o.0).collect::<Vec<_>>());

    // A second refresh measures nothing; a set whose files are gone is dropped.
    let rep = library::refresh(&cfg, &roots, Some(&caches), &|_| {}, &AtomicBool::new(false)).unwrap();
    assert_eq!((rep.measured, rep.added.len(), rep.removed.len()), (0, 0, 0), "{rep:?}");
    std::fs::remove_dir_all(flats.join("FLAT").join("B")).unwrap();
    let rep = library::refresh(&cfg, &roots, Some(&caches), &|_| {}, &AtomicBool::new(false)).unwrap();
    assert_eq!(rep.removed, vec!["H 2026-11-21".to_string()], "{rep:?}");
    assert_eq!(Library::load(&cfg.library).unwrap().unwrap().sets.len(), 1);
}
