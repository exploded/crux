//! Per-frame metadata pulled from headers (no pixels).

use crate::header::{parse_sexagesimal, Header};
use crate::{fits, xisf};
use anyhow::{bail, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileFormat {
    Fits,
    Xisf,
    /// Canon raw (`cr2.rs`).
    Cr2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameMeta {
    pub path: PathBuf,
    pub file_size: u64,
    pub mtime_ns: i64,
    pub format: FileFormat,
    pub width: u32,
    pub height: u32,
    /// FITS BITPIX, or XISF sample format name.
    pub sample: String,
    pub imagetyp: Option<String>,
    pub exptime: Option<f64>,
    pub filter: Option<String>,
    pub gain: Option<f64>,
    pub offset: Option<f64>,
    pub egain: Option<f64>,
    pub xbin: Option<u32>,
    pub ccd_temp: Option<f64>,
    pub set_temp: Option<f64>,
    pub instrume: Option<String>,
    pub telescop: Option<String>,
    pub object: Option<String>,
    /// Degrees.
    pub ra: Option<f64>,
    /// Degrees.
    pub dec: Option<f64>,
    /// UTC.
    pub date_obs: Option<NaiveDateTime>,
    /// Local time, from DATE-LOC.
    pub date_loc: Option<NaiveDateTime>,
    /// UTC end of the exposure, or of the last sub for a master (DATE-END).
    pub date_end: Option<NaiveDateTime>,
    pub pierside: Option<String>,
    pub airmass: Option<f64>,
    pub focpos: Option<f64>,
    pub foctemp: Option<f64>,
    pub cloudcvr: Option<f64>,
    pub mpsas: Option<f64>,
    pub sitelat: Option<f64>,
    pub sitelong: Option<f64>,
    pub focallen: Option<f64>,
    pub xpixsz: Option<f64>,
    /// XISF imageType attribute ("MasterDark", ...).
    pub xisf_image_type: Option<String>,
    /// The header says this is a stack of several frames (CCDStack or MaxIm
    /// `COMBINE`, PixInsight `ImageIntegration`, a CCDStack file that isn't a
    /// calibrated single frame).
    #[serde(default)]
    pub combined: bool,
    /// The header says it's calibrated (CCDStack/MaxIm `CALIBRAT = 1`,
    /// PixInsight `ImageCalibration`).
    #[serde(default)]
    pub calibrated: bool,
    /// Colour planes (NAXIS3); 1 for mono.
    #[serde(default = "one")]
    pub planes: u32,
    /// All keywords except COMMENT/HISTORY, for display.
    pub header: Header,
}

fn one() -> u32 {
    1
}

/// What COMMENT and HISTORY cards say about how a file was made:
/// (combined, calibrated).
fn provenance(h: &Header) -> (bool, bool) {
    let notes: Vec<String> = h.cards.iter().filter(|c| matches!(c.key.as_str(), "COMMENT" | "HISTORY")).map(|c| c.value.to_ascii_lowercase()).collect();
    let has = |s: &str| notes.iter().any(|n| n.contains(s));
    let calibrat = h.i64("CALIBRAT");
    let ccdstack = has("created by ccdstack");
    let calibrated = calibrat == Some(1) || has("imagecalibration") || h.contains("PEDESTAL") && has("calibrat");
    let combined = has("combine:") || has("imageintegration") || (ccdstack && calibrat != Some(1));
    (combined, calibrated)
}

impl FrameMeta {
    pub fn file_name(&self) -> &str {
        self.path.file_name().and_then(|n| n.to_str()).unwrap_or("")
    }

    /// Temperature used for grouping: the set point if present, else the sensor reading.
    pub fn temp(&self) -> Option<f64> {
        self.set_temp.or(self.ccd_temp)
    }

    pub fn binning(&self) -> u32 {
        self.xbin.unwrap_or(1)
    }

    pub fn size_key(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }

    /// The Bayer pattern of a raw one-shot-colour frame (None for mono and
    /// for debayered or stacked colour images).
    pub fn bayer(&self) -> Option<crate::cfa::Bayer> {
        crate::cfa::from_header(&self.header)
    }
}

pub fn format_of(path: &Path) -> Option<FileFormat> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "fits" | "fit" | "fts" => Some(FileFormat::Fits),
        "xisf" => Some(FileFormat::Xisf),
        "cr2" => Some(FileFormat::Cr2),
        _ => None,
    }
}

/// File size and modification time (ns since the Unix epoch), the cache key.
pub fn file_stamp(path: &Path) -> Result<(u64, i64)> {
    let md = std::fs::metadata(path)?;
    let mtime = md
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0);
    Ok((md.len(), mtime))
}

/// Read a frame's header (never the pixels) and extract metadata.
pub fn read_meta(path: &Path) -> Result<FrameMeta> {
    let (file_size, mtime_ns) = file_stamp(path)?;
    let format = match format_of(path) {
        Some(f) => f,
        None => bail!("not a FITS, XISF or CR2 file: {}", path.display()),
    };
    let (header, width, height, sample, xisf_image_type, channels) = match format {
        FileFormat::Fits => {
            let l = fits::read_layout(path)?;
            let (w, h) = (l.width() as u32, l.height() as u32);
            (l.header, w, h, l.bitpix.to_string(), None, 1)
        }
        FileFormat::Xisf => {
            let l = xisf::read_layout(path)?;
            (l.header, l.width as u32, l.height as u32, l.sample_format.name().to_string(), l.image_type, l.channels)
        }
        FileFormat::Cr2 => {
            let l = crate::cr2::read_layout(path)?;
            (l.header, l.width, l.height, "16".to_string(), None, 1)
        }
    };
    let mut m = from_header(path.to_path_buf(), file_size, mtime_ns, format, width, height, sample, xisf_image_type, header);
    // XISF keeps the channel count in its geometry, not in NAXIS3: a
    // PixInsight RGB image has no NAXIS keywords at all.
    m.planes = m.planes.max(channels as u32);
    Ok(m)
}

#[allow(clippy::too_many_arguments)]
pub fn from_header(
    path: PathBuf,
    file_size: u64,
    mtime_ns: i64,
    format: FileFormat,
    width: u32,
    height: u32,
    sample: String,
    xisf_image_type: Option<String>,
    header: Header,
) -> FrameMeta {
    let h = &header;
    let ra = h.f64("RA").or_else(|| h.get("OBJCTRA").and_then(parse_sexagesimal).map(|hours| hours * 15.0));
    let dec = h.f64("DEC").or_else(|| h.get("OBJCTDEC").and_then(parse_sexagesimal));
    let (combined, calibrated) = provenance(h);
    FrameMeta {
        combined,
        calibrated,
        planes: h.i64("NAXIS3").filter(|&n| h.i64("NAXIS") == Some(3) && n > 1).map(|n| n as u32).unwrap_or(1),
        imagetyp: h.str("IMAGETYP").or_else(|| h.str("FRAME")),
        exptime: h.f64("EXPTIME").or_else(|| h.f64("EXPOSURE")),
        filter: filter_of(h),
        gain: h.f64("GAIN").or_else(|| h.f64("ISOSPEED")),
        offset: h.f64("OFFSET"),
        egain: h.f64("EGAIN"),
        xbin: h.i64("XBINNING").map(|v| v as u32),
        ccd_temp: h.f64("CCD-TEMP"),
        set_temp: h.f64("SET-TEMP"),
        instrume: h.str("INSTRUME").or_else(|| h.str("CAMERA").map(|c| camera_model(&c))),
        telescop: h.str("TELESCOP"),
        object: h.str("OBJECT"),
        ra,
        dec,
        date_obs: dates(h).0,
        date_loc: dates(h).1,
        date_end: h.get("DATE-END").and_then(parse_datetime),
        pierside: h.str("PIERSIDE"),
        airmass: h.f64("AIRMASS"),
        focpos: h.f64("FOCPOS").or_else(|| h.f64("FOCUSPOS")),
        foctemp: h.f64("FOCTEMP").or_else(|| h.f64("FOCUSTEM")),
        cloudcvr: h.f64("CLOUDCVR"),
        mpsas: h.f64("MPSAS"),
        sitelat: h.f64("SITELAT").or_else(|| h.get("SITELAT").and_then(parse_sexagesimal)).or_else(|| h.f64("LATITUDE")),
        sitelong: h.f64("SITELONG").or_else(|| h.get("SITELONG").and_then(parse_sexagesimal)).or_else(|| h.f64("LONGITUD")),
        focallen: h.f64("FOCALLEN"),
        xpixsz: h.f64("XPIXSZ"),
        header: header.without_commentary(),
        path,
        file_size,
        mtime_ns,
        format,
        width,
        height,
        sample,
        xisf_image_type,
    }
}

/// (UTC, local) start of exposure. Celestron's Origin writes DATE-OBS in
/// local time (its comment says so) with `TIMEZONE = '+1000'`; read as UTC
/// it put every frame 10 hours late (wrong night, Sun and Moon).
fn dates(h: &Header) -> (Option<NaiveDateTime>, Option<NaiveDateTime>) {
    let obs = h.get("DATE-OBS").and_then(parse_datetime);
    let loc = h.get("DATE-LOC").and_then(parse_datetime);
    let local_obs = h.cards.iter().any(|c| c.key == "DATE-OBS" && c.comment.to_lowercase().contains("local"));
    if !local_obs {
        return (obs, loc);
    }
    let utc = obs.zip(h.get("TIMEZONE").and_then(parse_utc_offset)).map(|(t, off)| t - chrono::Duration::minutes(off));
    (utc, loc.or(obs))
}

/// "+1000", "+10:00", "-0530", "10" → minutes east of UTC.
fn parse_utc_offset(s: &str) -> Option<i64> {
    let t = s.trim();
    let (sign, rest) = match t.strip_prefix('-') {
        Some(r) => (-1, r),
        None => (1, t.strip_prefix('+').unwrap_or(t)),
    };
    let digits: String = rest.chars().filter(|c| c.is_ascii_digit()).collect();
    let (hh, mm) = match digits.len() {
        1 | 2 => (digits.parse::<i64>().ok()?, 0),
        3 | 4 => (digits[..digits.len() - 2].parse::<i64>().ok()?, digits[digits.len() - 2..].parse::<i64>().ok()?),
        _ => return None,
    };
    (hh <= 14 && mm < 60).then_some(sign * (hh * 60 + mm))
}

/// The filter, with Celestron Origin's "Nebula" spelled out as a line
/// filter. "Nebula" is the Origin's filter-drawer setting: Celestron's own
/// Nebula filter passes H-alpha, H-beta and OIII, and the owner's note with
/// the test data says other line filters went in that drawer too (Askar D1,
/// Antlia Tri-Band RGB), all with H-alpha and OIII. So `cfa::dual_band`
/// splits it: H from red, O (with any H-beta) from green and blue.
fn filter_of(h: &Header) -> Option<String> {
    let f = h.str("FILTER")?;
    let origin = h.get("CREATOR").is_some_and(|c| c.trim_start().starts_with("Origin"));
    Some(if origin && f.trim().eq_ignore_ascii_case("nebula") { "Nebula HaOIII".to_string() } else { f })
}

/// A camera keyword without a serial number: "Origin678C-0123456789abcdef0"
/// → "Origin678C" (bundles carry the model, never a device's identity).
fn camera_model(c: &str) -> String {
    match c.rsplit_once('-') {
        Some((model, id)) if id.len() >= 8 && id.chars().all(|ch| ch.is_ascii_hexdigit()) => model.to_string(),
        _ => c.to_string(),
    }
}

pub fn parse_datetime(s: &str) -> Option<NaiveDateTime> {
    let t = s.trim().trim_end_matches('Z');
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%d %H:%M:%S"] {
        if let Ok(d) = NaiveDateTime::parse_from_str(t, fmt) {
            return Some(d);
        }
    }
    // Some writers use a "+00:00" suffix.
    chrono::DateTime::parse_from_rfc3339(s.trim()).ok().map(|d| d.naive_utc())
}

/// Fill in what the header left out from the file name (applied after the
/// scan, so cached metadata gets it too). So far: the filter of an ASIAIR
/// file (`Light_NGC 292_300.0s_Bin1_gain120_20251120-212057_0.0C_HaO3_0001`
/// has no FILTER keyword).
pub fn fill_from_name(m: &mut FrameMeta) {
    if m.filter.as_deref().is_none_or(|f| f.trim().is_empty()) {
        if let Some(f) = m.path.file_stem().and_then(|s| s.to_str()).and_then(parse_asiair_filter) {
            m.filter = Some(f);
        }
    }
}

/// The filter in an ASIAIR file name:
/// `<Type>_<object>_<exp>s_Bin<n>_[<filter>_]gain<g>_<yyyymmdd-hhmmss>_<temp>C_[<filter>_]<nnnn>`.
/// Only names with ASIAIR's date-time and `Bin` tokens; the filter is the
/// one token after the exposure that is none of the known kinds.
pub fn parse_asiair_filter(stem: &str) -> Option<String> {
    let parts: Vec<&str> = stem.split('_').collect();
    let is_stamp = |s: &str| s.len() == 15 && s.as_bytes()[8] == b'-' && s.bytes().enumerate().all(|(i, b)| i == 8 || b.is_ascii_digit());
    let num = |s: &str| s.parse::<f64>().is_ok();
    let is_exp = |s: &str| s.strip_suffix("ms").or_else(|| s.strip_suffix('s')).is_some_and(num);
    let lower = |s: &str| s.to_ascii_lowercase();
    if !parts.iter().any(|p| is_stamp(p)) || !parts.iter().any(|p| lower(p).starts_with("bin") && num(&p[3..])) {
        return None;
    }
    let e = parts.iter().position(|p| is_exp(p))?;
    let known = |s: &str| {
        let l = lower(s);
        is_stamp(s)
            || is_exp(s)
            || (l.starts_with("bin") && num(&s[3..]))
            || (l.starts_with("gain") && num(&s[4..]))
            || (l.starts_with("offset") && num(&s[6..]))
            || s.strip_suffix('C').is_some_and(num)
            || (s.len() == 4 && s.bytes().all(|b| b.is_ascii_digit()))
    };
    let cands: Vec<&str> = parts[e + 1..].iter().copied().filter(|p| !p.is_empty() && !known(p)).collect();
    (cands.len() == 1).then(|| cands[0].to_string())
}

/// What a N.I.N.A. file name tells us:
/// `[PREFIX_]2026-11-05_21-21-13_H_-10.00_300.00s_0000`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NinaName {
    pub prefix: Option<String>,
    /// Local time.
    pub time: Option<NaiveDateTime>,
    pub filter: Option<String>,
    pub exposure: Option<f64>,
}

pub fn parse_nina_name(stem: &str) -> NinaName {
    let parts: Vec<&str> = stem.split('_').collect();
    let is_date = |s: &str| s.len() == 10 && s.as_bytes()[4] == b'-' && s.as_bytes()[7] == b'-' && s[..4].bytes().all(|b| b.is_ascii_digit());
    let Some(di) = parts.iter().position(|p| is_date(p)) else {
        return NinaName { prefix: parts.first().filter(|_| parts.len() > 1).map(|s| s.to_string()), ..Default::default() };
    };
    let prefix = if di > 0 { Some(parts[..di].join("_")) } else { None };
    let time = parts
        .get(di + 1)
        .and_then(|t| NaiveDateTime::parse_from_str(&format!("{} {}", parts[di], t), "%Y-%m-%d %H-%M-%S").ok());
    let exposure = parts.iter().skip(di + 2).find_map(|p| p.strip_suffix('s').and_then(|v| v.parse::<f64>().ok()));
    let filter = parts.get(di + 2).filter(|f| f.parse::<f64>().is_err() && !f.ends_with('s')).map(|s| s.to_string());
    NinaName { prefix, time, filter, exposure }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nina_names() {
        let n = parse_nina_name("2026-11-05_21-21-13_H_-10.00_300.00s_0000");
        assert_eq!(n.prefix, None);
        assert_eq!(n.filter.as_deref(), Some("H"));
        assert_eq!(n.exposure, Some(300.0));
        assert_eq!(n.time.unwrap().to_string(), "2026-11-05 21:21:13");
        let f = parse_nina_name("FLAT_2026-10-19_06-25-44_O_-10.00_0.84s_0000");
        assert_eq!(f.prefix.as_deref(), Some("FLAT"));
        assert_eq!(f.exposure, Some(0.84));
        let m = parse_nina_name("masterDark_BIN-1_6248x4176_EXPOSURE-300.00s");
        assert_eq!(m.prefix.as_deref(), Some("masterDark"));
        assert_eq!(m.time, None);
    }

    #[test]
    fn origin_headers() {
        let mut h = Header::default();
        h.cards.push(crate::header::Card { key: "DATE-OBS".into(), value: "2026-07-18T18:41:34".into(), comment: "Start of exposure (local time)".into() });
        h.push("TIMEZONE", "+1000", "");
        h.push("CREATOR", "Origin 1.4.6182", "");
        h.push("FILTER", "Nebula", "");
        h.push("ISOSPEED", "1000", "");
        h.push("CAMERA", "Origin678C-0123456789abcdef0", "");
        h.push("LATITUDE", "-37.6", "");
        let m = from_header(PathBuf::from("Light0001.fits"), 0, 0, FileFormat::Fits, 10, 10, "16".into(), None, h);
        assert_eq!(m.date_obs.unwrap().to_string(), "2026-07-18 08:41:34");
        assert_eq!(m.date_loc.unwrap().to_string(), "2026-07-18 18:41:34");
        assert_eq!(m.filter.as_deref(), Some("Nebula HaOIII"));
        assert_eq!(crate::cfa::dual_band(m.filter.as_deref().unwrap()), Some(["H", "O"]));
        assert_eq!(m.gain, Some(1000.0));
        assert_eq!(m.instrume.as_deref(), Some("Origin678C"));
        assert_eq!(m.sitelat, Some(-37.6));
        assert_eq!(parse_utc_offset("-05:30"), Some(-330));
        assert_eq!(parse_utc_offset("10"), Some(600));
    }

    #[test]
    fn asiair_names() {
        assert_eq!(parse_asiair_filter("Light_NGC 292_300.0s_Bin1_gain120_20251120-212057_0.0C_HaO3_0001").as_deref(), Some("HaO3"));
        assert_eq!(parse_asiair_filter("Light_NGC 2264_300.0s_Bin1_HaO3_gain120_20251215-001501_0.0C_0001").as_deref(), Some("HaO3"));
        assert_eq!(parse_asiair_filter("Flat_70.0ms_Bin1_HaO3_gain120_20251215-062509_9.8C_0001").as_deref(), Some("HaO3"));
        assert_eq!(parse_asiair_filter("Dark_300.0s_Bin1_gain120_20251117-121748_-0.3C_0001"), None);
        assert_eq!(parse_asiair_filter("Light_mosaic_Alnilam_60.0s_LP_20251211-215756"), None, "Seestar: no Bin token");
        assert_eq!(parse_asiair_filter("2026-11-05_21-21-13_H_-10.00_300.00s_0000"), None, "N.I.N.A.");
    }

    #[test]
    fn datetimes() {
        assert!(parse_datetime("2026-11-05T10:21:13.8379288").is_some());
        assert!(parse_datetime("2026-11-05T10:21:13Z").is_some());
        assert!(parse_datetime("2026-11-05T10:21:13.123+00:00").is_some());
    }
}
