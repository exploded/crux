//! End-to-end: a synthetic night of FITS frames through the full pipeline.
//! Each defect must be caught with the right check; clean frames must pass.

use crux_core::cache::Cache;
use crux_core::grade::{Check, Grade};
use crux_core::header::Header;
use crux_core::synth::{render, Params};
use crux_core::{fits, pipeline, Config};
use std::path::PathBuf;

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

fn write_frame(dir: &PathBuf, n: usize, p: &Params) {
    let (img, _) = render(p);
    let minute = 10 * n; // 21:00 local onwards (10:00 UTC, AEDT)
    let (hh, mm) = (21 + minute / 60, minute % 60);
    let mut h = Header::default();
    for (k, v) in [
        ("IMAGETYP", "LIGHT".to_string()),
        ("EXPTIME", "300.0".into()),
        ("FILTER", "H".into()),
        ("GAIN", "100".into()),
        ("OFFSET", "50".into()),
        ("EGAIN", "0.243".into()),
        ("OBJECT", "Test Field".into()),
        ("RA", "87.65".into()),
        ("DEC", "-68.32".into()),
        ("SITELAT", "-37.81".into()),
        ("SITELONG", "144.96".into()),
        ("DATE-LOC", format!("2026-11-20T{hh:02}:{mm:02}:00")),
        ("DATE-OBS", format!("2026-11-20T{:02}:{mm:02}:00", hh - 11)),
    ] {
        h.push(k, v, "");
    }
    let name = format!("LIGHT/2026-11-20_{hh:02}-{mm:02}-00_H_-10.00_300.00s_{n:04}.fits");
    let path = dir.join(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    fits::write_image(&path, &img, 16, &h).unwrap();
}

#[test]
fn synthetic_night_is_graded_correctly() {
    let tmp = Tmp(std::env::temp_dir().join(format!("crux-grade-{}", std::process::id())));
    let d = &tmp.0;
    let base = Params { width: 900, height: 600, n_stars: 350, ..Default::default() };
    // 0..6 clean, dithered, with seeing jitter.
    for n in 0..7 {
        let p = Params {
            fwhm: 3.5 + 0.1 * (n % 3) as f64,
            noise_seed: 100 + n as u64,
            shift: (n as f64 * 3.0, n as f64 * -2.0),
            ..base.clone()
        };
        write_frame(d, n, &p);
    }
    write_frame(d, 7, &Params { fwhm: 6.5, noise_seed: 7, ..base.clone() }); // bad focus
    write_frame(d, 8, &Params { ecc: 0.72, pa_deg: 25.0, noise_seed: 8, ..base.clone() }); // trailing
    write_frame(d, 9, &Params { sky_adu: 1500.0, noise_seed: 9, ..base.clone() }); // bright sky
    write_frame(d, 10, &Params { gradient: 0.6, noise_seed: 10, ..base.clone() }); // gradient
    write_frame(d, 11, &Params { n_stars: 0, sky_adu: 1.0, noise_adu: 8.0, noise_seed: 11, ..base.clone() }); // cap on
    write_frame(d, 12, &Params { transparency: 0.45, noise_seed: 12, shift: (5.0, 7.0), ..base.clone() }); // thin cloud
    write_frame(d, 13, &Params { cloud_patch: Some((0.25, 0.35)), noise_seed: 13, ..base.clone() }); // patchy
    write_frame(d, 14, &Params { trail: Some((0.0, 100.0, 25.0, 60.0, 3.0)), noise_seed: 14, ..base.clone() }); // satellite

    let mut cfg = Config::default();
    cfg.library.use_folder(&std::env::temp_dir().join("crux-test-no-library")); // never the real one
    let mut cache = Cache::disabled();
    let t = pipeline::run(d, &cfg, &mut cache, true, &|_, _, _| {});
    assert!(t.errors.is_empty(), "{:?}", t.errors);
    assert_eq!(t.reports.len(), 15);

    let by_seq = |n: usize| {
        let suffix = format!("_{n:04}.fits");
        t.reports.iter().find(|r| t.scan.frames[r.frame].meta.file_name().ends_with(&suffix)).unwrap()
    };
    let checks = |n: usize| by_seq(n).flags.iter().filter(|f| f.grade != Grade::Good).map(|f| f.check).collect::<Vec<_>>();

    let tv: Vec<String> = (0..15)
        .map(|n| {
            let r = by_seq(n);
            format!("{n}: {:?} n={:?} ref={:?}", r.transparency.as_ref().map(|t| (t.value * 100.0).round() / 100.0), r.transparency.as_ref().map(|t| t.n_matched), r.transparency.as_ref().map(|t| t.reference))
        })
        .collect();
    for n in 0..7 {
        assert_eq!(by_seq(n).grade, Grade::Good, "clean frame {n}: {}\n{}", by_seq(n).summary(), tv.join("\n"));
    }
    assert_eq!(by_seq(7).grade, Grade::Reject, "{}", by_seq(7).summary());
    assert!(checks(7).contains(&Check::Focus));
    assert!(by_seq(7).summary().contains("night median"));

    assert_eq!(by_seq(8).grade, Grade::Reject, "{}", by_seq(8).summary());
    assert!(checks(8).contains(&Check::Tracking));
    assert!(!checks(8).contains(&Check::Focus), "elongation shouldn't read as focus: {}", by_seq(8).summary());

    assert!(by_seq(9).grade >= Grade::Warn, "{}", by_seq(9).summary());
    assert!(checks(9).contains(&Check::Sky));
    assert!(by_seq(9).summary().contains("e-/px/s"));

    assert!(checks(10).contains(&Check::Gradient), "{}", by_seq(10).summary());
    assert!(by_seq(10).summary().contains("right"), "{}", by_seq(10).summary());

    assert_eq!(by_seq(11).grade, Grade::Reject);
    assert!(checks(11).contains(&Check::Type), "{}", by_seq(11).summary());

    // Thin cloud: plenty of stars, normal sky, but every star at 45%.
    let tr = by_seq(12).transparency.as_ref().expect("transparency measured");
    assert!((tr.value - 0.45).abs() < 0.06, "transparency {}", tr.value);
    assert_eq!(by_seq(12).grade, Grade::Reject, "{}", by_seq(12).summary());
    assert!(checks(12).contains(&Check::Transparency));
    assert!(!checks(12).contains(&Check::Stars), "star count shouldn't be what catches thin cloud");

    assert!(checks(13).contains(&Check::Patchy), "{}", by_seq(13).summary());
    assert!(by_seq(13).summary().contains("left"), "{}", by_seq(13).summary());

    // Satellite: 60 ADU against 20 ADU noise → reject, and nothing else wrong.
    assert_eq!(checks(14), vec![Check::Trail], "{}", by_seq(14).summary());
    assert_eq!(by_seq(14).grade, Grade::Reject);
    // No trails reported on the clean, blurred or elongated frames.
    for n in [0, 1, 2, 3, 4, 5, 6, 7, 8] {
        assert!(!checks(n).contains(&Check::Trail), "frame {n}: {}", by_seq(n).summary());
    }

    // Clean frames measure close to 1.0.
    for n in 0..7 {
        let v = by_seq(n).transparency.as_ref().map(|t| t.value).unwrap_or(0.0);
        assert!(v > 0.9, "clean frame {n} transparency {v}");
    }
}
