//! All tunable settings and thresholds in one place.
//!
//! Loaded from `crux.toml` (see `Config::load`). Every field has a default,
//! so a config file only needs the values you want to change.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub site: Site,
    pub camera: Camera,
    pub scan: Scan,
    pub grouping: Grouping,
    pub layout: Layout,
    pub library: Library,
    pub process: Process,
    pub analysis: Analysis,
    pub thresholds: Thresholds,
    pub cache: CacheSettings,
    /// Mosaic assembly (milestone 14).
    pub mosaic: crate::process::mosaic::Settings,
    /// The graphics card.
    pub gpu: GpuSettings,
    /// Other programs crux can use: ASTAP's star database, RC Astro's CLI.
    pub tools: Tools,
    /// Whether the app looks for a newer release (see `update`).
    pub updates: Updates,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            site: Site::default(),
            camera: Camera::default(),
            scan: Scan::default(),
            grouping: Grouping::default(),
            layout: Layout::default(),
            library: Library::default(),
            process: Process::default(),
            analysis: Analysis::default(),
            thresholds: Thresholds::default(),
            cache: CacheSettings::default(),
            mosaic: Default::default(),
            gpu: GpuSettings::default(),
            tools: Tools::default(),
            updates: Updates::default(),
        }
    }
}

/// Where other programs' files are, when crux can't find them itself.
/// `None` searches the usual install folders. The environment variables
/// `CRUX_CATALOG` and `CRUX_RC_ASTRO` win over these.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Tools {
    /// The folder holding ASTAP's star database (`d50_*.1476`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub catalog: Option<PathBuf>,
    /// RC Astro's CLI (`rc-astro.exe`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rc_astro: Option<PathBuf>,
}

impl Tools {
    /// Make these the process-wide choices (`Catalog::find`, `rcastro::find`).
    pub fn apply(&self) {
        crate::catalog::set_folder(self.catalog.clone());
        crate::rcastro::set_path(self.rc_astro.clone());
    }
}

/// The desktop app's update check (see `update`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Updates {
    /// Ask GitHub for the latest release when the app starts, at most
    /// once a day.
    pub check: bool,
}

impl Default for Updates {
    fn default() -> Self {
        Updates { check: true }
    }
}

/// Whether crux computes on the graphics card (see `gpu`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum GpuUse {
    /// When crux finds a suitable card; the CPU otherwise.
    #[default]
    Auto,
    /// Always the CPU.
    Off,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct GpuSettings {
    pub r#use: GpuUse,
}

/// Where each folder's cache lives and when it's cleaned (see `cachedir`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CacheSettings {
    /// Keep the cache inside each data folder (`<folder>\.crux`), as crux
    /// did before. Default: in the per-user data folder.
    pub in_folder: bool,
    /// Empty the rebuildable part of a folder's cache when it hasn't been
    /// opened for this many days (0 = never). Decisions and plans stay.
    pub unused_days: u32,
}

impl Default for CacheSettings {
    fn default() -> Self {
        CacheSettings { in_folder: false, unused_days: 90 }
    }
}

/// Pixel analysis settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Analysis {
    /// Background mesh cell size in pixels.
    pub mesh_cell: usize,
    /// Star detection threshold, in σ of the smoothed background noise.
    pub detect_sigma: f32,
    pub min_star_area: usize,
    /// Components larger than this (pixels) are nebula knots or bloated stars.
    pub max_star_area: usize,
    pub edge_margin: usize,
    /// Peak ADU at or above which a star counts as saturated.
    pub saturation_adu: f32,
    /// Measure at most this many stars per frame, brightest first.
    pub max_measure: usize,
    /// Stars below this SNR are counted but not used for shape statistics.
    pub min_shape_snr: f32,
    /// Stars kept per frame in the cache (for transparency and overlays).
    pub keep_stars: usize,
    /// Saturated stars kept as well, for matching and plate solving only.
    pub keep_saturated: usize,
    /// Frames analysed at once (0 = automatic). Each needs about 350 MB for a 26 MP frame.
    pub threads: usize,
    /// Look for satellite and aircraft trails.
    pub trails: bool,
    /// Shortest trail to report, full-resolution pixels.
    pub trail_min_length: f32,
    /// Hough significance a line needs before it's verified.
    pub trail_min_score: f32,
}

impl Default for Analysis {
    fn default() -> Self {
        Analysis {
            mesh_cell: 64,
            detect_sigma: 5.0,
            min_star_area: 5,
            max_star_area: 3000,
            edge_margin: 16,
            saturation_adu: 60000.0,
            max_measure: 1500,
            min_shape_snr: 30.0,
            keep_stars: 300,
            keep_saturated: 100,
            threads: 0,
            trails: true,
            trail_min_length: 600.0,
            trail_min_score: 8.0,
        }
    }
}

impl Analysis {
    pub fn thread_count(&self) -> usize {
        if self.threads > 0 {
            return self.threads;
        }
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 16)
    }
}

/// Grading thresholds. Ratios compare a frame with its cohort's median
/// (focus, stars, tracking) or best frames (sky). `min_z` guards against
/// flagging tiny differences in a very consistent cohort.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Thresholds {
    /// Fewest frames for a cohort comparison; smaller cohorts get absolute checks only.
    pub min_cohort: usize,
    /// Robust sigma is floored at this fraction of the median.
    pub sigma_floor: f64,
    pub min_z: f64,
    pub focus: Focus,
    pub tracking: Tracking,
    pub stars: StarCount,
    pub sky: Sky,
    pub gradient: Gradient,
    pub transparency: Transparency,
    pub trails: Trails,
    pub calibration: Calibration,
}

/// Bias, dark and flat checks. Per-frame values are compared with the
/// frame's own set. Flat limits come from FlatChecker.js.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Calibration {
    /// Median level away from the set's median, ADU.
    pub level_warn_adu: f64,
    pub level_reject_adu: f64,
    /// Noise / set median.
    pub noise_warn: f64,
    pub noise_reject: f64,
    /// Large-scale structure above the set's median, ADU (light leaks).
    pub structure_warn_adu: f64,
    pub structure_reject_adu: f64,
    /// Hot-pixel fraction / set median (darks).
    pub hot_warn: f64,
    /// Sensor temperature away from the set point, °C.
    pub temp_warn_c: f64,
    pub temp_reject_c: f64,
    /// Longest exposure that still counts as a bias, s.
    pub bias_max_exposure: f64,
    /// Fraction of pixels at 99.5% of full scale that makes a flat saturated.
    pub flat_saturated_fraction: f64,
    /// Flat level after the offset, fraction of full scale.
    pub flat_min_level: f64,
    pub flat_max_level: f64,
    /// Change from the set's median flat, % peak-to-peak (smoothed at 64 px).
    pub flat_change_warn: f64,
    pub flat_change_reject: f64,
    /// Error left in calibrated lights, % peak-to-peak (smoothed at 144 px).
    /// FlatChecker warns at 3%; the good October LMC flats leave 3.3–3.5%
    /// against the November lights (mostly a top-to-bottom sky gradient
    /// common to the lights), so crux warns from 4%.
    pub flat_error_warn: f64,
    pub flat_error_reject: f64,
    /// Masked fraction above which the error map probably includes the target.
    pub flat_mask_warn: f64,
    /// Lights within this many nights of a flat set are used to check it.
    pub flat_lights_days: i64,
    pub flat_min_lights: usize,
    /// Most lights stacked for the illumination (nearest in time first).
    pub flat_max_lights: usize,
    /// Row or column banding / set median.
    pub banding_warn: f64,
    /// Bias level away from `camera.bias_level_adu`, ADU.
    pub bias_expected_tolerance_adu: f64,
    /// Dust fingerprint correlation, flat against the lights' illumination:
    /// at or above `dust_match` the flats fit; below `dust_mismatch` the
    /// optical train has probably changed. On the LMC data the right flats
    /// score 0.19–0.51 per night (0.6 over all nights); shifted controls 0.08 or less.
    pub dust_match: f64,
    pub dust_mismatch: f64,
    /// Flat set against flat set: at or above this the dust is the same.
    pub dust_same_train: f64,
    /// Sets of one filter that differ by no more than this (smoothed, %
    /// peak-to-peak) and share their dust can be combined. James: "about
    /// 1.5%"; the October LMC sets differ by 1.6% (H) and 1.9% (O).
    pub combine_pct: f64,
    /// A flat set's median difference from the other library sets of its
    /// filter with the same dust, %. Good LMC-era sets sit at 2–4.4%; the
    /// cloudy 2026-10-29 and FLAT.bad sets at 12–21%.
    pub consensus_warn_pct: f64,
    pub consensus_reject_pct: f64,
}

impl Default for Calibration {
    fn default() -> Self {
        Calibration {
            level_warn_adu: 5.0,
            level_reject_adu: 20.0,
            noise_warn: 1.2,
            noise_reject: 1.5,
            structure_warn_adu: 3.0,
            structure_reject_adu: 10.0,
            hot_warn: 1.5,
            temp_warn_c: 1.0,
            temp_reject_c: 3.0,
            bias_max_exposure: 0.1,
            flat_saturated_fraction: 0.001,
            flat_min_level: 0.15,
            flat_max_level: 0.8,
            flat_change_warn: 1.0,
            flat_change_reject: 3.0,
            flat_error_warn: 4.0,
            flat_error_reject: 6.0,
            flat_mask_warn: 0.25,
            flat_lights_days: 60,
            flat_min_lights: 5,
            flat_max_lights: 400,
            banding_warn: 1.5,
            bias_expected_tolerance_adu: 50.0,
            dust_match: 0.3,
            dust_mismatch: 0.15,
            dust_same_train: 0.6,
            combine_pct: 2.0,
            consensus_warn_pct: 5.0,
            consensus_reject_pct: 8.0,
        }
    }
}

/// James rejected 35 of the 50 LMC frames with trails. The trails he kept
/// were mostly fainter than the pixel noise, which stacking rejection removes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Trails {
    /// Reject when a trail's brightness is at least this many times the
    /// frame's pixel noise; fainter trails warn.
    pub reject_sigma: f64,
}

impl Default for Trails {
    fn default() -> Self {
        Trails { reject_sigma: 1.0 }
    }
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            min_cohort: 4,
            sigma_floor: 0.05,
            min_z: 3.0,
            focus: Focus::default(),
            tracking: Tracking::default(),
            stars: StarCount::default(),
            sky: Sky::default(),
            gradient: Gradient::default(),
            transparency: Transparency::default(),
            trails: Trails::default(),
            calibration: Calibration::default(),
        }
    }
}

/// Transparency = median flux ratio of the same stars against the field's
/// best frame. Star signal (and so SNR) scales with it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Transparency {
    pub warn: f64,
    pub reject: f64,
    /// Warn when the worst 4×4 cell is this far below the frame's median cell.
    pub patchy_warn: f64,
    /// Fewest matched stars for a measurement.
    pub min_matches: usize,
    /// Fewest frames of the same field to compare.
    pub min_frames: usize,
    /// Atmospheric extinction, mag per airmass, to say how much of a loss
    /// airmass explains. A typical H-alpha value: the LMC data couldn't fit
    /// it (airmass varies by only ~0.08 within a panel).
    pub extinction_k: f64,
}

impl Default for Transparency {
    fn default() -> Self {
        Transparency { warn: 0.75, reject: 0.5, patchy_warn: 0.35, min_matches: 15, min_frames: 3, extinction_k: 0.1 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Focus {
    /// FWHM / night median at which to warn.
    pub warn_ratio: f64,
    pub reject_ratio: f64,
    /// Warn regardless of the cohort above this FWHM (px). 8 px = 5.4" at 0.67"/px.
    pub abs_warn_px: f64,
}

impl Default for Focus {
    fn default() -> Self {
        Focus { warn_ratio: 1.25, reject_ratio: 1.5, abs_warn_px: 8.0 }
    }
}

/// Tracking is judged mainly against the night: on the LMC data PixInsight
/// measures kept frames at eccentricity 0.46–0.79 (median 0.52), so a fixed
/// 0.5 or 0.6 limit would reject frames James keeps.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Tracking {
    /// Absolute median eccentricity to warn / reject at, when elongation shares one direction.
    pub warn_ecc: f64,
    pub reject_ecc: f64,
    /// Warn / reject when eccentricity exceeds the night median by this much (and min_z).
    pub rel_warn: f64,
    pub rel_reject: f64,
    /// How aligned position angles must be (0 = random, 1 = identical).
    pub min_coherence: f64,
    /// Corner minus centre eccentricity that points to optics (coma, tilt).
    pub optics_excess: f64,
}

impl Default for Tracking {
    fn default() -> Self {
        Tracking { warn_ecc: 0.75, reject_ecc: 0.85, rel_warn: 0.15, rel_reject: 0.25, min_coherence: 0.6, optics_excess: 0.15 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StarCount {
    /// Star count / night median below which to warn / reject. A gross check
    /// only: star count is a poor cloud detector (transparency handles that).
    pub warn_frac: f64,
    pub reject_frac: f64,
}

impl Default for StarCount {
    fn default() -> Self {
        StarCount { warn_frac: 0.3, reject_frac: 0.05 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Sky {
    /// Sky (e-/px/s) / the panel's best at which to warn / reject. A frame at
    /// 4× the sky is worth about a quarter of a good one.
    pub warn_ratio: f64,
    pub reject_ratio: f64,
    /// "Best" sky is this percentile of the panel's frames (robust to oddities).
    pub best_percentile: f64,
}

impl Default for Sky {
    fn default() -> Self {
        Sky { warn_ratio: 2.5, reject_ratio: 4.0, best_percentile: 0.1 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Gradient {
    /// Warn when the sky gradient across the frame differs from the panel's
    /// typical gradient by this many percent of the sky level. On the LMC
    /// data dawn-twilight gradients of 30–40% were common in kept frames
    /// (DBE removes them), so only extreme gradients warn.
    pub warn_pct: f64,
    /// Mention (without changing the grade) above this.
    pub note_pct: f64,
}

impl Default for Gradient {
    fn default() -> Self {
        Gradient { warn_pct: 50.0, note_pct: 20.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Site {
    /// Degrees, north positive. Used when SITELAT is missing.
    pub latitude: f64,
    /// Degrees, east positive. Used when SITELONG is missing.
    pub longitude: f64,
    /// IANA time zone, used to turn DATE-OBS (UTC) into local time when a
    /// frame has no DATE-LOC. WBPP masters never have DATE-LOC, and a fixed
    /// offset is an hour out for half the year.
    pub timezone: String,
}

impl Default for Site {
    fn default() -> Self {
        Site { latitude: -37.81, longitude: 144.96, timezone: "Australia/Melbourne".into() }
    }
}

impl Site {
    pub fn tz(&self) -> chrono_tz::Tz {
        self.timezone.parse().unwrap_or(chrono_tz::Australia::Melbourne)
    }

    /// Whether `timezone` is an IANA zone name crux knows (`tz` falls back
    /// to Melbourne otherwise).
    pub fn timezone_ok(&self) -> bool {
        self.timezone.parse::<chrono_tz::Tz>().is_ok()
    }

    /// Local wall-clock time for a UTC time.
    pub fn to_local(&self, utc: chrono::NaiveDateTime) -> chrono::NaiveDateTime {
        use chrono::TimeZone;
        self.tz().from_utc_datetime(&utc).naive_local()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Camera {
    /// e-/ADU, used when EGAIN is missing.
    pub egain: f64,
    /// Expected bias offset in ADU at the usual gain/offset (for bias checks).
    pub bias_level_adu: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Camera { egain: 0.243, bias_level_adu: 500.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Scan {
    pub extensions: Vec<String>,
    /// Folder names (case-insensitive, `*` wildcards) skipped entirely.
    /// The defaults are PixInsight WBPP outputs, not raw subs.
    pub exclude_dirs: Vec<String>,
}

impl Default for Scan {
    fn default() -> Self {
        Scan {
            extensions: ["fits", "fit", "fts", "xisf", "cr2"].map(String::from).to_vec(),
            exclude_dirs: [".crux", "calibrated", "registered", "logs", "*.OLD", "mosaic*", "fastIntegration", "cosmetized", "debayered", "drizzle*"]
                .map(String::from)
                .to_vec(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Grouping {
    /// Darks match lights within this many °C.
    pub dark_temp_tolerance_c: f64,
    /// Relative exposure tolerance for dark matching (0.005 = 0.5%).
    pub exposure_tolerance: f64,
    /// Folder names that hold rejects (case-insensitive, `*` wildcards).
    pub reject_dirs: Vec<String>,
    /// File names that mark a reject (case-insensitive, `*` wildcards);
    /// N.I.N.A. and hand renames use a `BAD_` prefix.
    pub reject_names: Vec<String>,
}

impl Default for Grouping {
    fn default() -> Self {
        Grouping {
            dark_temp_tolerance_c: 1.0,
            exposure_tolerance: 0.005,
            reject_dirs: ["BAD", "*.bad", "reject*"].map(String::from).to_vec(),
            reject_names: vec!["BAD_*".into()],
        }
    }
}

/// Target folder layout, relative to the target folder. Placeholders:
/// `{object}`, `{panel}`, `{filter}`, `{night}`, `{exposure}`, `{reason}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Layout {
    pub light: String,
    pub flat: String,
    pub flat_retired: String,
    pub dark: String,
    pub bias: String,
    pub master: String,
    pub reject: String,
}

impl Default for Layout {
    fn default() -> Self {
        Layout {
            light: "LIGHT/{panel}".into(),
            flat: "FLAT/{night}".into(),
            flat_retired: "FLAT.bad".into(),
            dark: "dark".into(),
            bias: "bias".into(),
            master: "master".into(),
            reject: "BAD/{reason}".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Library {
    /// Where `crux plan --to-library` files flat sets
    /// (`<root>\<filter>\<night>`). Sets are used wherever they are
    /// otherwise; nothing is written here unless you apply such a plan.
    pub root: PathBuf,
    /// Where the library's index is kept: `index.db` (the same database as
    /// the image index by default), never in the data. Default
    /// `%LOCALAPPDATA%\crux`.
    pub dir: PathBuf,
    /// Folders "Refresh calibration library" (`crux library refresh`)
    /// searches: every folder inside each is scanned for flats, darks, bias
    /// and dark flats, which stay where they are. Default: the folder that
    /// holds `root` (`C:\PixInsight`).
    pub search: Vec<PathBuf>,
}

impl Default for Library {
    fn default() -> Self {
        let data = crate::paths::data_dir().unwrap_or_else(|| PathBuf::from("crux"));
        let root = if cfg!(windows) { PathBuf::from(r"C:\PixInsight\FlatLibrary") } else { data.join("FlatLibrary") };
        Library { root, dir: data, search: Vec::new() }
    }
}

impl Library {
    /// `--library DIR`: the index and any filed sets both go in `DIR`
    /// (tests point this at scratch, so nothing reaches the real library).
    pub fn use_folder(&mut self, dir: &Path) {
        self.root = dir.to_path_buf();
        self.dir = dir.to_path_buf();
    }

    /// The folders to search: `CRUX_LIBRARY_SEARCH` (a path list, as in
    /// PATH), else `search`, else the folder holding `root`.
    pub fn search_roots(&self) -> Vec<PathBuf> {
        if let Some(v) = std::env::var_os("CRUX_LIBRARY_SEARCH").filter(|v| !v.is_empty()) {
            return std::env::split_paths(&v).collect();
        }
        if !self.search.is_empty() {
            return self.search.clone();
        }
        self.root.parent().filter(|p| !p.as_os_str().is_empty()).map(|p| vec![p.to_path_buf()]).unwrap_or_default()
    }
}

/// Native processing (milestone 11 onward).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Process {
    /// Where calibrated and registered frames and masters go. Never inside
    /// the data. Default `C:\crux-work` on Windows.
    pub workspace: PathBuf,
    /// Process lights graded warn as well as good.
    pub include_warn: bool,
    /// Also write calibrated (unregistered) frames.
    pub keep_calibrated: bool,
    /// Hot pixels: replace pixels this many sigmas above the master dark's
    /// median (0 = off). WBPP: 10.
    pub hot_sigma: f32,
    /// Output pedestal when more than this fraction of pixels would fall
    /// below zero (0 = never). WBPP "auto": 0.0001.
    pub pedestal_limit: f64,
    /// Lanczos-3 clamping threshold (0 = off, the default).
    pub lanczos_clamp: f32,
    /// Integrate each field and filter into a master light (milestone 12).
    pub integrate: bool,
    /// "local" (match changing sky gradients, like LocalNormalization) or
    /// "global".
    pub normalisation: String,
    /// "snr" ((signal ÷ noise)², like PSF Signal Weight), "exposure" or
    /// "equal".
    pub weights: String,
    /// "auto" (WBPP's choice by stack size), "esd", "winsorized", "sigma",
    /// "percentile" or "none".
    pub rejection: String,
    /// Also write a FITS copy (32-bit float, `.fits`, same name) beside
    /// each integrated master, mosaic and finished image, for programs
    /// that don't read XISF. XISF stays the main output.
    pub fits: bool,
}

impl Default for Process {
    fn default() -> Self {
        let workspace = if cfg!(windows) {
            PathBuf::from(r"C:\crux-work")
        } else {
            crate::paths::data_dir().unwrap_or_else(|| PathBuf::from("crux")).join("crux-work")
        };
        Process {
            workspace,
            include_warn: true,
            keep_calibrated: false,
            hot_sigma: 10.0,
            pedestal_limit: 0.0001,
            lanczos_clamp: 0.0,
            integrate: true,
            normalisation: "local".into(),
            weights: "snr".into(),
            rejection: "auto".into(),
            fits: false,
        }
    }
}

impl Config {
    /// Load `explicit` if given, else `<dir>/crux.toml`, else
    /// `crux.toml` in the settings folder (`%APPDATA%\crux` on Windows; see
    /// `paths`), else defaults.
    pub fn load(explicit: Option<&Path>, dir: Option<&Path>) -> Result<(Config, Option<PathBuf>)> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(p) = explicit {
            candidates.push(p.to_path_buf());
        } else {
            if let Some(d) = dir {
                candidates.push(d.join("crux.toml"));
            }
            if let Some(a) = crate::paths::config_dir() {
                candidates.push(a.join("crux.toml"));
            }
        }
        for p in candidates {
            if p.is_file() {
                let text = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
                let cfg: Config = toml::from_str(&text).with_context(|| format!("parse {}", p.display()))?;
                crate::gpu::set_enabled(cfg.gpu.r#use == GpuUse::Auto);
                cfg.tools.apply();
                return Ok((cfg, Some(p)));
            }
            if explicit.is_some() {
                anyhow::bail!("config file {} not found", p.display());
            }
        }
        crate::gpu::set_enabled(true);
        Tools::default().apply();
        Ok((Config::default(), None))
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    /// The per-user settings file (`%APPDATA%\crux\crux.toml` on Windows),
    /// used when a folder has no `crux.toml` of its own.
    pub fn user_path() -> Option<PathBuf> {
        crate::paths::config_dir().map(|d| d.join("crux.toml"))
    }

    /// Only the values that differ from crux's defaults, as TOML, so a
    /// saved file keeps following defaults that change later.
    pub fn changes_toml(&self) -> String {
        use toml::Value;
        fn diff(a: &Value, b: Option<&Value>) -> Option<Value> {
            match (a, b) {
                (Value::Table(x), Some(Value::Table(y))) => {
                    let mut t = toml::value::Table::new();
                    for (k, v) in x {
                        if let Some(d) = diff(v, y.get(k)) {
                            t.insert(k.clone(), d);
                        }
                    }
                    (!t.is_empty()).then_some(Value::Table(t))
                }
                (a, Some(b)) if a == b => None,
                (a, _) => Some(a.clone()),
            }
        }
        let (Ok(mine), Ok(def)) = (Value::try_from(self), Value::try_from(Config::default())) else { return String::new() };
        diff(&mine, Some(&def)).map(|v| toml::to_string_pretty(&v).unwrap_or_default()).unwrap_or_default()
    }

    /// Save the values that differ from the defaults to the per-user file
    /// (whole file at once: a reader never sees half of it). Comments in a
    /// hand-edited file aren't kept.
    pub fn save_user(&self) -> Result<PathBuf> {
        let p = Config::user_path().context("no settings folder on this system")?;
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))?;
        }
        let text = format!(
            "# crux settings: only values that differ from crux's defaults.\n# Saved by crux's Settings page; a folder's own crux.toml is used\n# instead of this file.\n\n{}",
            self.changes_toml()
        );
        crate::paths::write_atomic(&p, text.replace('\n', crate::paths::EOL).as_bytes()).with_context(|| format!("write {}", p.display()))?;
        Ok(p)
    }
}

/// Case-insensitive match with `*` wildcards.
pub fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.to_lowercase().chars().collect();
    let t: Vec<char> = text.to_lowercase().chars().collect();
    let (mut pi, mut ti, mut star, mut mark) = (0, 0, None, 0);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == t[ti] || p[pi] == '?') {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcards() {
        assert!(wildcard_match("mosaic*", "mosaic_claude_2026-11-12"));
        assert!(wildcard_match("*.OLD", "master.old"));
        assert!(wildcard_match("BAD", "bad"));
        assert!(!wildcard_match("*.bad", "FLAT"));
        assert!(wildcard_match("*.bad", "FLAT.bad"));
    }

    #[test]
    fn defaults_roundtrip() {
        let c = Config::default();
        let back: Config = toml::from_str(&c.to_toml()).unwrap();
        assert_eq!(back.site.latitude, c.site.latitude);
        let partial: Config = toml::from_str("[site]\nlatitude = 1.0\n").unwrap();
        assert_eq!(partial.site.latitude, 1.0);
        assert_eq!(partial.site.longitude, Site::default().longitude);
    }

    #[test]
    fn changes_only() {
        assert_eq!(Config::default().changes_toml().trim(), "");
        let mut c = Config::default();
        c.process.fits = true;
        c.site.timezone = "Europe/London".into();
        c.library.search = vec![PathBuf::from(r"D:\Astro")];
        let t = c.changes_toml();
        assert!(t.contains("fits = true") && t.contains("Europe/London") && t.contains("Astro"), "{t}");
        assert!(!t.contains("latitude") && !t.contains("workspace") && !t.contains("mesh_cell"), "{t}");
        let back: Config = toml::from_str(&t).unwrap();
        assert!(back.process.fits);
        assert_eq!(back.site.timezone, "Europe/London");
        assert_eq!(back.site.latitude, c.site.latitude);
        assert_eq!(back.library.search, c.library.search);
    }

    #[test]
    fn tools_saved_only_when_chosen() {
        let mut c = Config::default();
        assert!(!c.changes_toml().contains("tools"));
        c.tools.catalog = Some(PathBuf::from("/data/astap"));
        let t = c.changes_toml();
        assert!(t.contains("[tools]") && t.contains("catalog") && !t.contains("rc_astro"), "{t}");
        let back: Config = toml::from_str(&t).unwrap();
        assert_eq!(back.tools, c.tools);
    }

    #[test]
    fn catalog_falls_back_past_a_wrong_folder() {
        // A chosen folder without a database must not hide the usual
        // install (or, without one, give a database that isn't there).
        let empty = std::env::temp_dir().join(format!("crux-cat-test-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        let usual = crate::catalog::Catalog::find_with(None).map(|(c, _)| c.dir);
        let chosen = crate::catalog::Catalog::find_with(Some(empty.clone())).map(|(c, _)| c.dir);
        assert_eq!(chosen, usual);
        let missing = crate::catalog::Catalog::find_with(Some(empty.join("not-there"))).map(|(c, _)| c.dir);
        assert_eq!(missing, usual);
        let _ = std::fs::remove_dir(&empty);
        if let (Some(dir), None) = (usual, std::env::var_os("CRUX_CATALOG")) {
            let (_, src) = crate::catalog::Catalog::find_with(Some(dir)).unwrap();
            assert_eq!(src, crate::catalog::Source::Setting);
        }
        assert!(crate::rcastro::find_with(Some(PathBuf::from("/no/such/rc-astro"))).map(|(p, _)| p) == crate::rcastro::find_with(None).map(|(p, _)| p));
    }
}
