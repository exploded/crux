//! Work out each frame's type and role from the sources we trust, in order:
//! IMAGETYP header, N.I.N.A./WBPP file-name prefix, folder name, then pixels.

use crate::config::{wildcard_match, Config};
use crate::meta::{parse_nina_name, FrameMeta};
use chrono::{Duration, NaiveDate, NaiveDateTime};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FrameType {
    Light,
    Flat,
    Dark,
    Bias,
    DarkFlat,
    Unknown,
}

impl FrameType {
    pub fn name(self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Flat => "flat",
            Self::Dark => "dark",
            Self::Bias => "bias",
            Self::DarkFlat => "dark flat",
            Self::Unknown => "unknown",
        }
    }
}

/// Raw sub, integrated master, or a processing product (calibrated,
/// registered, reference frame...) that triage should leave alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    Raw,
    Master,
    Processed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeSource {
    Header,
    FileName,
    Folder,
    Pixels,
}

impl TypeSource {
    pub fn name(self) -> &'static str {
        match self {
            Self::Header => "IMAGETYP",
            Self::FileName => "file name",
            Self::Folder => "folder",
            Self::Pixels => "pixels",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Classification {
    pub frame_type: FrameType,
    pub role: Role,
    /// Which source decided the type.
    pub source: Option<TypeSource>,
    /// Every source that had an opinion.
    pub votes: Vec<(TypeSource, FrameType)>,
    /// Filed as a reject: under a reject folder such as BAD\ or FLAT.bad\,
    /// or named like one (`BAD_…`). Lights crux grades good are still
    /// processed (`process::job`).
    pub in_reject_dir: bool,
    /// Named as a reject (`BAD_…`) but not under a reject folder.
    #[serde(default)]
    pub reject_name_only: bool,
    /// Local time of the exposure start, best available.
    pub local_time: Option<NaiveDateTime>,
    /// Observing night: the local date at the evening start (noon to noon).
    pub night: Option<NaiveDate>,
}

impl Classification {
    pub fn conflicts(&self) -> Vec<(TypeSource, FrameType)> {
        self.votes.iter().copied().filter(|(_, t)| *t != self.frame_type).collect()
    }
    pub fn has_conflict(&self) -> bool {
        !self.conflicts().is_empty()
    }
}

/// Map an IMAGETYP / imageType / prefix string to a type and master flag.
pub fn parse_type_word(s: &str) -> Option<(FrameType, bool)> {
    let w: String = s.to_ascii_lowercase().chars().filter(|c| c.is_ascii_alphabetic()).collect();
    let master = w.starts_with("master");
    let w = w.trim_start_matches("master");
    let w = w.trim_end_matches("frame").trim_end_matches("field");
    let t = match w {
        "light" | "lights" | "object" | "science" => FrameType::Light,
        "flat" | "flats" | "skyflat" | "domeflat" => FrameType::Flat,
        "dark" | "darks" => FrameType::Dark,
        "bias" | "biases" | "offset" | "zero" => FrameType::Bias,
        "darkflat" | "flatdark" | "darkflats" | "flatdarks" => FrameType::DarkFlat,
        _ => return None,
    };
    Some((t, master))
}

fn folder_type(name: &str) -> Option<FrameType> {
    let lower = name.to_ascii_lowercase();
    // FLAT.bad, flats_old and similar still say "flat".
    let base = lower.split(['.', '_', ' ', '-']).next().unwrap_or("");
    parse_type_word(base).filter(|(_, m)| !m).map(|(t, _)| t)
}

/// Classify one frame. `root` is the scan root; only folders below it vote.
pub fn classify(meta: &FrameMeta, root: &Path, cfg: &Config) -> Classification {
    let mut votes = Vec::new();
    let mut master = false;

    if let Some((t, m)) = meta.imagetyp.as_deref().and_then(parse_type_word) {
        votes.push((crate::classify::TypeSource::Header, t));
        master |= m;
    } else if let Some((t, m)) = meta.xisf_image_type.as_deref().and_then(parse_type_word) {
        votes.push((TypeSource::Header, t));
        master |= m;
    }

    let stem = meta.path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let nina = parse_nina_name(stem);
    if let Some((t, m)) = nina.prefix.as_deref().and_then(|p| parse_type_word(p.split('_').next().unwrap_or(p))) {
        votes.push((TypeSource::FileName, t));
        master |= m;
    }

    let rel_dirs: Vec<String> = meta
        .path
        .parent()
        .and_then(|p| p.strip_prefix(root).ok())
        .map(|p| p.components().filter_map(|c| c.as_os_str().to_str().map(str::to_string)).collect())
        .unwrap_or_default();
    if let Some(t) = rel_dirs.iter().rev().find_map(|d| folder_type(d)) {
        votes.push((TypeSource::Folder, t));
    }
    let in_folder = rel_dirs.iter().any(|d| cfg.grouping.reject_dirs.iter().any(|p| wildcard_match(p, d)));
    let named = cfg.grouping.reject_names.iter().any(|p| wildcard_match(p, meta.file_name()));
    let (in_reject_dir, reject_name_only) = (in_folder || named, named && !in_folder);

    let (frame_type, source) = votes.first().map(|&(s, t)| (t, Some(s))).unwrap_or((FrameType::Unknown, None));

    let lower = stem.to_ascii_lowercase();
    // MaxIm's "Dark Master bin1x1 -20C 1200 seconds", "Flat Master - …".
    master |= meta.combined || lower.split(|c: char| !c.is_ascii_alphanumeric()).any(|w| w == "master");
    let processed = meta.planes > 1
        || (meta.calibrated && !master)
        || lower.starts_with("ln_reference")
        || lower.ends_with("_c")
        || lower.ends_with("_c_r")
        || lower.ends_with("_r")
        || lower.ends_with("_autocrop")
        || lower.contains("_c_cc")
        || lower.starts_with("drizzle")
        || (master && frame_type == FrameType::Light);
    let role = if processed {
        Role::Processed
    } else if master {
        Role::Master
    } else {
        Role::Raw
    };

    let local_time = meta
        .date_loc
        .or_else(|| meta.date_obs.map(|t| cfg.site.to_local(t)))
        .or(nina.time);
    let night = local_time.map(night_of);

    Classification { frame_type, role, source, votes, in_reject_dir, reject_name_only, local_time, night }
}

/// The observing night a local time belongs to: noon to noon, so 01:30 on
/// the 6th is the night of the 5th, and dawn flats at 06:25 are too.
pub fn night_of(local: NaiveDateTime) -> NaiveDate {
    (local - Duration::hours(12)).date()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Header;
    use crate::meta::{from_header, FileFormat};
    use std::path::PathBuf;

    fn meta(path: &str, imagetyp: Option<&str>) -> FrameMeta {
        let mut h = Header::default();
        if let Some(t) = imagetyp {
            h.push("IMAGETYP", t, "");
        }
        h.push("DATE-LOC", "2026-11-06T01:30:00", "");
        from_header(PathBuf::from(path), 0, 0, FileFormat::Fits, 10, 10, "16".into(), None, h)
    }

    #[test]
    fn maxim_and_ccdstack_files() {
        let cfg = Config::default();
        let root = Path::new("C:/data");
        let with = |path: &str, typ: &str, cards: &[(&str, &str)]| {
            let mut h = Header::default();
            h.push("IMAGETYP", typ, "");
            for (k, v) in cards {
                h.push(*k, *v, "");
            }
            classify(&from_header(PathBuf::from(path), 0, 0, FileFormat::Fits, 10, 10, "-32".into(), None, h), root, &cfg)
        };
        let raw = with("C:/data/NGC6188-001H.fit", "Light Frame", &[]);
        assert_eq!((raw.frame_type, raw.role), (FrameType::Light, Role::Raw));
        let stack = with("C:/data/stacked/Median NGC6188L.FIT", "Light Frame", &[("COMMENT", "created by CCDStack"), ("CALIBRAT", "1"), ("HISTORY", "COMBINE: MEDIAN of the following SOURCEs")]);
        assert_eq!(stack.role, Role::Processed);
        let colour = with("C:/data/stacked/color LRGB.FIT", "Light Frame", &[("NAXIS", "3"), ("NAXIS3", "3")]);
        assert_eq!(colour.role, Role::Processed);
        let cal = with("C:/data/Calibrated/NGC3372-001B.FIT", "Light Frame", &[("COMMENT", "created by CCDStack"), ("CALIBRAT", "1")]);
        assert_eq!(cal.role, Role::Processed);
        let dark = with("C:/data/Dark Master bin1x1 -20C 1200 seconds.FIT", "Dark Frame", &[("COMMENT", "created by CCDStack"), ("CALIBRAT", "0")]);
        assert_eq!((dark.frame_type, dark.role), (FrameType::Dark, Role::Master));
        let flat = with("C:/data/Flat Master - AT12IN_Ha_1x1_skyflat0.FIT", "Flat", &[]);
        assert_eq!((flat.frame_type, flat.role), (FrameType::Flat, Role::Master));
    }

    #[test]
    fn pixinsight_rgb_xisf_is_processed() {
        // PixInsight's finished images carry no IMAGETYP or NAXIS3; the
        // channel count is only in the XISF geometry (NGC 2244's
        // `master\combined_…_NBtoRGB_stars.xisf` was graded as a light).
        let dir = std::env::temp_dir().join(format!("crux-rgb-xisf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("combined_Final_SHO.xisf");
        let px = vec![0.1f32; 8];
        crate::xisf::write_planes(&path, 4, 2, &[&px, &px, &px], &Header::default(), &[]).unwrap();
        let m = crate::meta::read_meta(&path).unwrap();
        assert_eq!(m.planes, 3);
        assert_eq!(classify(&m, &dir, &Config::default()).role, Role::Processed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn night_rule() {
        let t = |s| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M").unwrap();
        assert_eq!(night_of(t("2026-11-06 01:30")).to_string(), "2026-11-05");
        assert_eq!(night_of(t("2026-11-05 21:00")).to_string(), "2026-11-05");
        assert_eq!(night_of(t("2026-10-19 06:25")).to_string(), "2026-10-18");
        assert_eq!(night_of(t("2026-10-19 12:00")).to_string(), "2026-10-19");
    }

    #[test]
    fn utc_to_melbourne_follows_daylight_saving() {
        let cfg = Config::default();
        let t = |s| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M").unwrap();
        // September: AEST (+10). November: AEDT (+11).
        assert_eq!(cfg.site.to_local(t("2026-09-11 20:15")), t("2026-09-12 06:15"));
        assert_eq!(cfg.site.to_local(t("2026-11-03 19:11")), t("2026-11-04 06:11"));
    }

    #[test]
    fn sources_and_conflicts() {
        let cfg = Config::default();
        let root = Path::new("C:/data");
        let c = classify(&meta("C:/data/LIGHT/PANEL_1/2026-11-06_01-30-00_H_-10.00_300.00s_0000.fits", Some("LIGHT")), root, &cfg);
        assert_eq!((c.frame_type, c.role, c.source), (FrameType::Light, Role::Raw, Some(TypeSource::Header)));
        assert!(!c.has_conflict());
        assert_eq!(c.night.unwrap().to_string(), "2026-11-05");

        let c = classify(&meta("C:/data/LIGHT/FLAT_2026-10-19_06-25-44_O_-10.00_0.84s_0000.fits", Some("FLAT")), root, &cfg);
        assert_eq!(c.frame_type, FrameType::Flat);
        assert_eq!(c.conflicts(), vec![(TypeSource::Folder, FrameType::Light)]);

        let c = classify(&meta("C:/data/FLAT.bad/x.fits", None), root, &cfg);
        assert_eq!((c.frame_type, c.source, c.in_reject_dir), (FrameType::Flat, Some(TypeSource::Folder), true));
        assert!(!c.reject_name_only);

        // N.I.N.A.'s BAD_ prefix marks a reject wherever the file is.
        let c = classify(&meta("C:/data/LIGHT/PANEL_6/BAD_2026-11-08_22-03-25_H_-10.00_300.00s_0003.fits", Some("LIGHT")), root, &cfg);
        assert_eq!((c.frame_type, c.in_reject_dir, c.reject_name_only), (FrameType::Light, true, true));
        let c = classify(&meta("C:/data/BAD/BAD_2026-11-09_22-13-54_H_-10.00_300.00s_0001.fits", Some("LIGHT")), root, &cfg);
        assert_eq!((c.in_reject_dir, c.reject_name_only), (true, false));
        let c = classify(&meta("C:/data/LIGHT/2026-11-08_22-03-25_H_BADPIX_0003.fits", Some("LIGHT")), root, &cfg);
        assert!(!c.in_reject_dir);

        let c = classify(&meta("C:/data/dark/masterDark_BIN-1_6248x4176_EXPOSURE-300.00s.xisf", Some("Master Dark")), root, &cfg);
        assert_eq!((c.frame_type, c.role), (FrameType::Dark, Role::Master));

        let c = classify(&meta("C:/data/master/masterLight_BIN-1_FILTER-H_mono_PANEL-1.xisf", Some("Master Light")), root, &cfg);
        assert_eq!(c.role, Role::Processed);

        // The root folder's own name never votes.
        let c = classify(&meta("C:/flats/x.fits", None), Path::new("C:/flats"), &cfg);
        assert_eq!(c.frame_type, FrameType::Unknown);
    }
}
