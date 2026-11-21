//! End-to-end: write tiny synthetic NINA-style files, scan, group, match.

use crux_core::cache::Cache;
use crux_core::header::Header;
use crux_core::{fits, group, scan, xisf, Config, FrameType, Image};
use std::path::{Path, PathBuf};

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

fn write(dir: &Path, rel: &str, cards: &[(&str, &str)]) {
    let p = dir.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    let mut h = Header::default();
    for (k, v) in cards {
        h.push(*k, *v, "");
    }
    let img = Image::new(8, 6);
    if rel.ends_with(".xisf") {
        xisf::write_image(&p, &img, &h, None).unwrap();
    } else {
        fits::write_image(&p, &img, 16, &h).unwrap();
    }
}

fn light(dir: &Path, panel: u32, loc: &str, filter: &str) {
    let name = format!("LIGHT/PANEL_{panel}/{}_{filter}_-10.00_300.00s_0000.fits", loc.replace('T', "_").replace(':', "-"));
    let obj = format!("LMC Panel {panel}");
    write(
        dir,
        &name,
        &[("IMAGETYP", "LIGHT"), ("EXPTIME", "300.0"), ("FILTER", filter), ("GAIN", "100"), ("OFFSET", "50"), ("XBINNING", "1"), ("SET-TEMP", "-10.0"), ("CCD-TEMP", "-10.1"), ("OBJECT", &obj), ("DATE-LOC", loc)],
    );
}

#[test]
fn scan_group_and_match() {
    let tmp = Tmp(std::env::temp_dir().join(format!("crux-scan-{}", std::process::id())));
    let d = &tmp.0;
    light(d, 1, "2026-11-05T21:21:13", "H");
    light(d, 1, "2026-11-06T01:30:00", "H"); // same night
    light(d, 1, "2026-11-06T22:00:00", "H"); // next night
    light(d, 2, "2026-11-05T23:00:00", "O");
    write(d, "BAD/2026-11-05_22-00-00_H_-10.00_300.00s_0001.fits", &[("IMAGETYP", "LIGHT"), ("EXPTIME", "300.0"), ("FILTER", "H"), ("GAIN", "100"), ("OFFSET", "50"), ("OBJECT", "LMC Panel 1"), ("DATE-LOC", "2026-11-05T22:00:00"), ("SET-TEMP", "-10")]);
    for (night_dir, t) in [("2026-10-18", "2026-10-19T06:25:44"), ("2026-10-19", "2026-10-20T06:32:03")] {
        write(d, &format!("FLAT/{night_dir}/FLAT_{}_H_-10.00_0.84s_0000.fits", t.replace('T', "_").replace(':', "-")), &[("IMAGETYP", "FLAT"), ("EXPTIME", "0.84"), ("FILTER", "H"), ("GAIN", "100"), ("OFFSET", "50"), ("DATE-LOC", t)]);
    }
    write(d, "FLAT.bad/FLAT_2026-11-04_06-05-58_H_-10.00_4.36s_0000.fits", &[("IMAGETYP", "FLAT"), ("EXPTIME", "4.36"), ("FILTER", "H"), ("GAIN", "100"), ("OFFSET", "50"), ("DATE-LOC", "2026-11-04T06:05:58")]);
    write(d, "dark/masterDark_BIN-1_8x6_EXPOSURE-300.00s.xisf", &[("IMAGETYP", "Master Dark"), ("EXPTIME", "300.0"), ("XBINNING", "1")]);
    write(d, "bias/masterBias_BIN-1_8x6.xisf", &[("IMAGETYP", "Master Bias"), ("XBINNING", "1")]);
    // A master flat integrated from the retired set (its span overlaps it).
    write(d, "master/masterFlat_BIN-1_8x6_FILTER-H_mono.xisf", &[("IMAGETYP", "Master Flat"), ("FILTER", "H"), ("DATE-OBS", "2026-11-03T19:06:00"), ("DATE-END", "2026-11-03T19:07:00")]);
    // Header says DARK, folder says LIGHT: a disagreement to report.
    write(d, "LIGHT/odd.fits", &[("IMAGETYP", "DARK"), ("EXPTIME", "60"), ("GAIN", "100"), ("OFFSET", "50")]);
    // WBPP output folders are skipped.
    write(d, "calibrated/x_c.xisf", &[("IMAGETYP", "LIGHT")]);

    let cfg = Config::default();
    let mut cache = Cache::load(&d.join(".crux"));
    let r = scan::scan(d, &cfg, &mut cache, &|_, _| {});
    assert!(r.errors.is_empty(), "{:?}", r.errors);
    assert_eq!(r.frames.len(), 12);
    cache.save().unwrap();

    let conflicts: Vec<_> = r.frames.iter().filter(|f| f.class.has_conflict()).collect();
    assert_eq!(conflicts.len(), 1);
    assert!(conflicts[0].meta.file_name() == "odd.fits");

    let g = group::build(&r.frames, &cfg);
    let lights: Vec<_> = g.by_kind(FrameType::Light).collect();
    assert_eq!(lights.len(), 3, "{:#?}", lights.iter().map(|l| &l.label).collect::<Vec<_>>());
    let p1 = lights.iter().find(|l| l.label.contains("Panel 1") && l.label.contains("2026-11-05")).unwrap();
    assert_eq!(p1.members.len(), 3);
    assert_eq!(p1.n_rejected, 1);

    let cal = g.light_calibration.iter().find(|c| c.group == p1.id).unwrap();
    assert_eq!(cal.darks.len(), 1, "{:?}", cal.missing);
    assert!(g.groups[cal.darks[0]].master);
    assert_eq!(cal.bias.len(), 1);
    assert_eq!(cal.flats.len(), 2);
    assert_eq!(cal.flats[0].nights_apart, -17);
    assert!(cal.notes.iter().any(|n| n.contains("gain")), "{:?}", cal.notes);

    let o = g.light_calibration.iter().find(|c| g.groups[c.group].filter.as_deref() == Some("O")).unwrap();
    assert!(o.missing.iter().any(|m| m.contains("no O flats")), "{:?}", o.missing);

    let retired: Vec<_> = g.by_kind(FrameType::Flat).filter(|f| f.retired).collect();
    assert_eq!(retired.len(), 2, "raw set in FLAT.bad plus the master built from it");
    assert_eq!(g.warnings.len(), 1, "{:?}", g.warnings);
    let master_flat = retired.iter().find(|f| f.master).unwrap();
    assert!(g.groups[master_flat.built_from.unwrap()].retired);
    assert_eq!(g.sessions.len(), 2);

    // Rescan hits the cache for every file.
    let mut cache = Cache::load(&d.join(".crux"));
    let r2 = scan::scan(d, &cfg, &mut cache, &|_, _| {});
    assert_eq!(r2.stats.cache_hits, 12);
}
