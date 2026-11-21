//! End-to-end: synthetic bias, darks, flats and lights through the pipeline.
//! Each calibration defect must be caught with the right check.

use crux_core::cache::Cache;
use crux_core::classify::FrameType;
use crux_core::grade::{Check, Grade};
use crux_core::header::Header;
use crux_core::image::Image;
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

/// Integer ADU frame from a per-pixel function plus Gaussian noise.
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

fn common(kind: &str, exp: f64, date: &str, ccd: f64) -> Vec<(&'static str, String)> {
    vec![
        ("IMAGETYP", kind.to_string()),
        ("EXPTIME", format!("{exp}")),
        ("GAIN", "100".into()),
        ("OFFSET", "50".into()),
        ("CCD-TEMP", format!("{ccd}")),
        ("SET-TEMP", "-10".into()),
        ("DATE-LOC", date.to_string()),
    ]
}

#[test]
fn calibration_defects_are_caught() {
    let tmp = Tmp(std::env::temp_dir().join(format!("crux-cal-{}", std::process::id())));
    let d = &tmp.0;

    // Bias: 0..5 clean, 6 offset changed, 7 light leak (bright corner), 8 row banding.
    let mut band_rng = Rng::new(77);
    let bands: Vec<f64> = (0..H).map(|_| 2.0 * band_rng.gauss()).collect();
    for n in 0..9u64 {
        let img = match n {
            8 => frame(n, 4.0, |_, y| 500.0 + bands[y]),
            6 => frame(n, 4.0, |_, _| 530.0),
            7 => frame(n, 4.0, |x, y| 500.0 + 40.0 * (-(((x as f64) / 300.0).powi(2) + ((y as f64) / 300.0).powi(2))).exp()),
            _ => frame(n, 4.0, |_, _| 500.0),
        };
        write(d, &format!("bias/BIAS_{n:04}.fits"), &img, &common("BIAS", 0.0, &format!("2026-11-20T20:00:{n:02}"), -10.0));
    }
    // Darks: 0..6 clean, 7 sensor warm.
    for n in 0..8u64 {
        let img = frame(100 + n, 4.5, |_, _| 501.0);
        let ccd = if n == 7 { -6.0 } else { -10.0 };
        write(d, &format!("dark/DARK_{n:04}.fits"), &img, &common("DARK", 300.0, &format!("2026-11-20T20:1{n}:00"), ccd));
    }
    // Good flats (night of 19 Nov): 0..4 clean, 5 saturated, 6 changed (5% tilt).
    for n in 0..7u64 {
        let img = match n {
            5 => frame(200 + n, 30.0, |x, y| 500.0 + 70000.0 * vignetting(x as f64, y as f64)),
            6 => frame(200 + n, 30.0, |x, y| 500.0 + 25000.0 * vignetting(x as f64, y as f64) * (1.0 + 0.05 * (y as f64 / H as f64 - 0.5))),
            _ => frame(200 + n, 30.0, |x, y| 500.0 + 25000.0 * vignetting(x as f64, y as f64)),
        };
        let mut kv = common("FLAT", 1.0, &format!("2026-11-20T06:00:{n:02}"), -10.0);
        kv.push(("FILTER", "H".into()));
        write(d, &format!("FLAT/good/FLAT_{n:04}.fits"), &img, &kv);
    }
    // Bad flats (night of 21 Nov): a 12% left-right twilight gradient in every frame.
    for n in 0..5u64 {
        let img = frame(300 + n, 30.0, |x, y| 500.0 + 25000.0 * vignetting(x as f64, y as f64) * (1.06 - 0.12 * x as f64 / W as f64));
        let mut kv = common("FLAT", 1.0, &format!("2026-11-22T06:00:{n:02}"), -10.0);
        kv.push(("FILTER", "H".into()));
        write(d, &format!("FLAT/bad/FLAT_{n:04}.fits"), &img, &kv);
    }
    // Lights (night of 20 Nov): vignetted sky over the bias, a few stars at
    // different places in each frame (dithering). Twelve, so each half of the
    // consistency check has enough.
    for n in 0..12u64 {
        let mut rng = Rng::new(400 + n);
        let stars: Vec<(f64, f64)> = (0..60).map(|_| (rng.uniform() * W as f64, rng.uniform() * H as f64)).collect();
        let img = frame(500 + n, 8.0, |x, y| {
            let mut v = 500.0 + 600.0 * vignetting(x as f64, y as f64);
            for &(sx, sy) in &stars {
                let r2 = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                if r2 < 50.0 {
                    v += 3000.0 * (-r2 / 4.0).exp();
                }
            }
            v
        });
        let mut kv = common("LIGHT", 300.0, &format!("2026-11-20T22:{:02}:00", n * 5), -10.0);
        kv.push(("FILTER", "H".into()));
        kv.push(("OBJECT", "Test".into()));
        write(d, &format!("LIGHT/LIGHT_{n:04}.fits"), &img, &kv);
    }

    let mut cfg = Config::default();
    cfg.library.use_folder(&std::env::temp_dir().join("crux-test-no-library")); // never the real one
    let mut cache = Cache::disabled();
    let t = pipeline::run(d, &cfg, &mut cache, true, &|_, _, _| {});
    assert!(t.cal_errors.is_empty(), "{:?}", t.cal_errors);

    let report = |dir: &str, n: usize| {
        let name = format!("_{n:04}.fits");
        t.cal
            .reports
            .iter()
            .find(|r| {
                let p = t.scan.frames[r.frame].path().to_string_lossy().replace('\\', "/");
                p.contains(&format!("/{dir}/")) && p.ends_with(&name)
            })
            .unwrap_or_else(|| panic!("no report for {dir} {n}"))
    };
    let checks = |dir: &str, n: usize| report(dir, n).flags.iter().filter(|f| f.grade != Grade::Good).map(|f| (f.check, f.grade)).collect::<Vec<_>>();

    for n in 0..6 {
        assert!(checks("bias", n).is_empty(), "bias {n}: {:?}", report("bias", n).flags);
    }
    assert!(checks("bias", 6).contains(&(Check::Level, Grade::Reject)), "{:?}", report("bias", 6).flags);
    assert!(checks("bias", 7).iter().any(|c| c.0 == Check::Structure), "{:?}", report("bias", 7).flags);
    assert!(checks("bias", 8).iter().any(|c| c.0 == Check::Banding), "{:?}", report("bias", 8).flags);
    let bias_set = t.cal.sets.iter().find(|s| t.grouping.groups[s.group].kind == FrameType::Bias).unwrap();
    let rn = bias_set.read_noise_adu.expect("read noise from pairs");
    assert!((rn - 4.0).abs() < 0.3, "read noise {rn}");
    for n in 0..7 {
        assert!(checks("dark", n).is_empty(), "dark {n}: {:?}", report("dark", n).flags);
    }
    assert!(checks("dark", 7).contains(&(Check::Temperature, Grade::Reject)), "{:?}", report("dark", 7).flags);

    for n in 0..5 {
        assert!(checks("good", n).is_empty(), "good flat {n}: {:?}", report("good", n).flags);
    }
    assert!(checks("good", 5).contains(&(Check::Saturated, Grade::Reject)), "{:?}", report("good", 5).flags);
    assert!(checks("good", 6).contains(&(Check::FlatChange, Grade::Reject)), "{:?}", report("good", 6).flags);

    let set_of = |dir: &str| {
        let f = report(dir, 0).frame;
        let g = t.grouping.groups.iter().find(|g| g.kind == FrameType::Flat && g.members.contains(&f)).unwrap();
        t.cal.set(g.id).unwrap()
    };
    let good = set_of("good");
    let v = good.vs_lights.as_ref().expect("good flats checked against lights");
    assert!(v.error.spread < 2.0 && v.grade == Grade::Good, "{:?}", v);
    let bad = set_of("bad");
    let v = bad.vs_lights.as_ref().expect("bad flats checked against lights");
    assert_eq!(v.grade, Grade::Reject, "{:?} masked {}", v.error.edges, v.error.masked);
    assert!(v.error.edges.left_right() < -8.0, "{:?}", v.error.edges);
    for n in 0..5 {
        assert!(checks("bad", n).contains(&(Check::FlatMatch, Grade::Reject)), "{:?}", report("bad", n).flags);
    }
    let pair = t.cal.pairs.first().expect("flat sets compared");
    assert!(pair.spread > 8.0, "{}", pair.spread);
}
