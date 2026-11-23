mod dev;
mod html;
mod report;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use crux_core::cache::Cache;
use crux_core::pipeline::{self, Triage};
use crux_core::Config;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "crux", version, about = "Astrophotography sub-frame triage")]
struct Cli {
    /// Config file (default: <dir>\crux.toml, then %APPDATA%\crux\crux.toml).
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// Where to keep the scan cache (default: one folder per data folder in
    /// %LOCALAPPDATA%\crux\folders; see `crux cache`).
    #[arg(long, global = true, env = "CRUX_CACHE_DIR")]
    cache_dir: Option<PathBuf>,
    /// Don't read or write the cache.
    #[arg(long, global = true)]
    no_cache: bool,
    /// Flat library folder: its database and any sets filed with --to-library (default: library.dir, %LOCALAPPDATA%\crux, and library.root).
    #[arg(long, global = true, env = "CRUX_LIBRARY")]
    library: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Read every FITS/XISF file under <dir>, classify and measure the frames.
    Scan {
        dir: PathBuf,
        /// Read headers only; skip pixel analysis.
        #[arg(long)]
        headers_only: bool,
        /// List every frame.
        #[arg(long)]
        list: bool,
        /// Write the scan result as JSON to this file.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Grade every frame, group sessions and calibration sets, match calibration.
    Report {
        dir: PathBuf,
        /// Read headers only; skip pixel analysis and grading.
        #[arg(long)]
        headers_only: bool,
        /// Write per-frame metrics and grades as CSV.
        #[arg(long)]
        csv: Option<PathBuf>,
        /// Write a standalone HTML report with charts.
        #[arg(long)]
        html: Option<PathBuf>,
        /// Write everything as JSON.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// The flat library: index flat sets and compare them.
    Library {
        #[command(subcommand)]
        cmd: LibCmd,
    },
    /// Build a move plan: rejects to BAD\<reason>, bad flat sets to FLAT.bad,
    /// and optionally organise, move flats into the library, combine sets.
    /// Nothing moves; review it, then `crux apply <plan>`.
    Plan {
        dir: PathBuf,
        /// Where to write the plan (default: <cache dir>\plans\plan-<time>.json).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Don't move rejected frames.
        #[arg(long)]
        no_rejects: bool,
        /// Don't retire flat sets crux rejects.
        #[arg(long)]
        no_retire: bool,
        /// Also move raw frames into the layout template (config `layout`).
        #[arg(long)]
        organise: bool,
        /// Also move good flat sets into the flat library.
        #[arg(long)]
        to_library: bool,
        /// Also combine library flat sets that agree (same filter, within combine_pct, same dust).
        #[arg(long)]
        combine: bool,
        /// Keep files matching this name pattern even if crux rejects them (repeatable; * and ? wildcards).
        #[arg(long, value_name = "PATTERN")]
        keep: Vec<String>,
        /// Reject files matching this name pattern whatever their grade.
        #[arg(long, value_name = "PATTERN")]
        reject: Vec<String>,
        /// Retire a flat set: FILTER:NIGHT, e.g. H:2026-10-29.
        #[arg(long, value_name = "FILTER:NIGHT")]
        retire_set: Vec<String>,
        /// Never retire this flat set, even if crux rejects it.
        #[arg(long, value_name = "FILTER:NIGHT")]
        keep_set: Vec<String>,
    },
    /// Apply a plan after checking every move (no clashes, nothing changed
    /// since planning). Writes a journal for undo.
    Apply {
        plan: PathBuf,
        /// Don't ask for confirmation.
        #[arg(long)]
        yes: bool,
        /// Journal path (default: next to the plan).
        #[arg(long)]
        journal: Option<PathBuf>,
        /// Copy and verify even on one drive (for testing the cross-drive path).
        #[arg(long, hide = true)]
        force_copy: bool,
    },
    /// Reverse an applied plan from its journal.
    Undo {
        journal: PathBuf,
        /// Don't ask for confirmation.
        #[arg(long)]
        yes: bool,
        #[arg(long, hide = true)]
        force_copy: bool,
    },
    /// List files matching a smart-view query, e.g. `crux find <dir> "grade:reject t<0.5"`.
    /// Keys: panel, filter, night, grade, type, pier, target, folder, reason;
    /// comparisons on t, fwhm, ecc, stars, sky, alt, exp; other words match the path.
    Find {
        dir: PathBuf,
        query: String,
        /// Headers only (no grades or measurements).
        #[arg(long)]
        headers_only: bool,
    },
    /// Rank the darks, bias and dark flats for <dir>'s lights and flats
    /// across the library (the folders beside it), by hot-pixel fingerprint,
    /// level, headers and date.
    Darks {
        dir: PathBuf,
        /// More folders to search (e.g. raw darks kept elsewhere).
        #[arg(long)]
        also: Vec<PathBuf>,
    },
    /// Calibrate, fix hot pixels and register <dir>'s good lights into the
    /// workspace (config process.workspace, default C:\crux-work). Reads the
    /// data in place; writes nothing there.
    Process {
        dir: PathBuf,
        /// Workspace root (default: process.workspace in the config).
        #[arg(long)]
        workspace: Option<PathBuf>,
        /// Also write calibrated (unregistered) frames.
        #[arg(long)]
        keep_calibrated: bool,
        /// Leave out lights graded warn.
        #[arg(long)]
        no_warn: bool,
        /// Use the light whose file name contains this as the reference.
        #[arg(long)]
        reference: Option<String>,
        /// Show the plan only.
        #[arg(long)]
        dry_run: bool,
        /// More folders to look in for darks and bias (the folders beside
        /// <dir> are always searched when <dir> has none).
        #[arg(long)]
        also: Vec<PathBuf>,
        /// Calibrate and register only; don't integrate master lights.
        #[arg(long)]
        no_integrate: bool,
    },
    /// Run RC Astro's tools on an image of yours with your own licence
    /// (their stand-alone CLI, rc-astro): BlurXTerminator, then
    /// NoiseXTerminator, then StarXTerminator, as chosen. Without an image,
    /// shows what's installed and licensed. Outputs are marked RCASTRO and
    /// are never used for training crux's own tools.
    RcAstro {
        /// FITS or XISF image (a linear master, for BlurXTerminator).
        image: Option<PathBuf>,
        /// Output file (FITS or XISF, by extension).
        #[arg(long)]
        out: Option<PathBuf>,
        /// BlurXTerminator: "correct" (correct only) or "sharpen".
        #[arg(long)]
        bxt: Option<String>,
        /// BlurXTerminator sharpen stars, 0–0.7 (default 0.5).
        #[arg(long)]
        sharpen_stars: Option<f32>,
        /// BlurXTerminator adjust star halos, −0.5–0.5 (default 0).
        #[arg(long, allow_hyphen_values = true)]
        adjust_halos: Option<f32>,
        /// BlurXTerminator sharpen nonstellar, 0–1 (default 0.5).
        #[arg(long)]
        sharpen_nonstellar: Option<f32>,
        /// BlurXTerminator PSF diameter, pixels (default: automatic).
        #[arg(long)]
        psf: Option<f32>,
        /// NoiseXTerminator.
        #[arg(long)]
        nxt: bool,
        /// NoiseXTerminator denoise, 0–1 (default 0.9).
        #[arg(long)]
        denoise: Option<f32>,
        /// StarXTerminator (starless image).
        #[arg(long)]
        sxt: bool,
        /// With --sxt, also write the stars-only image (<out>_stars).
        #[arg(long)]
        stars: bool,
        /// Compute device: auto, cpu, gpu, gpuN.
        #[arg(long)]
        device: Option<String>,
    },
    /// Make a training bundle from <dir>'s registered frames: pairs of
    /// stacks made from different frames of one field, cut into patches,
    /// plus star cutouts and camera details, for training crux's denoising.
    /// <dir> is a data folder crux has processed, its workspace folder, or
    /// a folder of registered frames from WBPP or Siril. Holds no file
    /// names, site or observer; bundle.html shows what's in it.
    Collect {
        dir: PathBuf,
        /// Bundle folder (default: <workspace>\train\<folder name>).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Workspace root (default: process.workspace in the config).
        #[arg(long)]
        workspace: Option<PathBuf>,
        /// Size limit, MB.
        #[arg(long, default_value_t = 500.0)]
        max_mb: f64,
        /// Halves only (no quarter-stack pairs).
        #[arg(long)]
        no_quarters: bool,
        /// Keep the integrated stacks in <out>\.stacks.
        #[arg(long)]
        keep_stacks: bool,
        /// Replace an existing bundle in <out>.
        #[arg(long)]
        overwrite: bool,
        /// Every light frame directly in <dir> is already aligned (by MaxIm,
        /// DSS or another program).
        #[arg(long)]
        registered: bool,
        /// Don't ask first.
        #[arg(long)]
        yes: bool,
    },
    /// Plate-solve <dir>'s lights from their measured stars (ASTAP's D50
    /// star database, found in its install folder or CRUX_CATALOG).
    /// Solutions are kept in the library index in the cache folder.
    Solve {
        dir: PathBuf,
        /// Only files whose path contains this.
        #[arg(long)]
        only: Option<String>,
        /// Solve again even when the index has a solution.
        #[arg(long)]
        force: bool,
    },
    /// Finished images from each target's integrated masters: gradient
    /// removal, colour mix, stretch and finishing (milestone 13).
    Image {
        dir: PathBuf,
        /// Processing workspace (default: process.workspace in the config).
        #[arg(long)]
        workspace: Option<PathBuf>,
        /// Only targets whose name contains this.
        #[arg(long)]
        object: Option<String>,
        /// A saved recipe (`crux recipe list`) or a recipe file. Default: the
        /// target's last recipe, else crux's default for its channels.
        #[arg(long)]
        recipe: Option<String>,
        /// Start from this preset (e.g. "SHO", "Copper & Blue", "Natural").
        #[arg(long)]
        preset: Option<String>,
        /// Leave gradients alone.
        #[arg(long)]
        no_gradient: bool,
        /// Narrowband stars: "natural" (a starless mix with naturally
        /// coloured stars added back), "keep" (as the palette mixes them),
        /// "network" (natural, starless made by crux's star remover) or
        /// "classic" (natural, starless made by the classic fill).
        /// Default: the recipe's (natural, star remover).
        #[arg(long, value_parser = ["natural", "keep", "network", "classic"])]
        stars: Option<String>,
        /// Natural stars: one stretch for the stars and the starless image
        /// (stars recoloured in place) instead of stretching them separately
        /// and screening the stars on.
        #[arg(long)]
        one_stretch: bool,
        /// Natural stars stretched separately: how hard the star layer is
        /// stretched (1 = like the nebula on dark sky; default 0.6).
        #[arg(long)]
        star_size: Option<f64>,
        /// Natural stars stretched separately: star brightness (default 1).
        #[arg(long)]
        star_brightness: Option<f64>,
        /// Also save the recipe under this name.
        #[arg(long)]
        save_recipe: Option<String>,
        /// Remove noise from the linear masters first; optional amount 0–1
        /// (default 0.8).
        #[arg(long, num_args = 0..=1, default_missing_value = "0.8")]
        denoise: Option<f32>,
        /// Correct star shapes ("correct": coma, elongation) or also tighten
        /// them ("tighten"), on the linear masters.
        #[arg(long, value_parser = ["correct", "tighten", "off"])]
        sharpen: Option<String>,
        /// How much of the sharpening to apply, 0–1 (default 1).
        #[arg(long)]
        sharpen_amount: Option<f32>,
        /// Who does it: "crux" (crux's own networks) or "rc-astro" (your
        /// licensed BlurXTerminator and NoiseXTerminator).
        #[arg(long, value_parser = ["crux", "rc-astro"])]
        engine: Option<String>,
        /// Ignore the recipe's denoise and sharpen settings.
        #[arg(long)]
        no_enhance: bool,
        /// DIR holds masters stacked by another program (PixInsight WBPP's
        /// masterLight files, in DIR or DIR\master): one target named after
        /// the folder, read in place.
        #[arg(long)]
        masters: bool,
        /// Write only a JPEG with its longest side at most this many pixels
        /// (and the recipe), not the full-size XISF, PNG and JPEG.
        #[arg(long)]
        size: Option<usize>,
    },
    /// Assemble mosaics from a target's integrated panel masters ("LMC
    /// Panel 1", "LMC Panel 2", …): plate-solved, distortion-corrected,
    /// brightness-matched and feathered, one master per filter
    /// (milestone 14). `crux process` runs it after integrating.
    Mosaic {
        dir: PathBuf,
        /// Processing workspace (default: process.workspace in the config).
        #[arg(long)]
        workspace: Option<PathBuf>,
        /// Only targets whose name contains this.
        #[arg(long)]
        target: Option<String>,
        /// Canvas pixels per panel pixel (1 = full resolution). Default:
        /// the smallest that keeps the canvas under mosaic.max_megapixels.
        #[arg(long)]
        bin: Option<usize>,
        /// Blend width at panel edges, arcminutes.
        #[arg(long)]
        feather: Option<f64>,
        /// Linear plate solutions only (no lens distortion fit).
        #[arg(long)]
        no_distortion: bool,
        /// Leave panel brightness as integrated.
        #[arg(long)]
        no_match: bool,
        /// Planes only: no per-panel curvature in the brightness match.
        #[arg(long)]
        no_curvature: bool,
        /// Assemble again even when nothing changed.
        #[arg(long)]
        force: bool,
    },
    /// Watch a folder (N.I.N.A.'s output) and grade new subs as they land;
    /// starts with the latest night's summary (milestone 14).
    Watch {
        dir: PathBuf,
        /// Seconds between looks at the folder.
        #[arg(long, default_value_t = 10)]
        interval: u64,
        /// Stop after this many seconds (default: until Ctrl+C).
        #[arg(long = "for")]
        duration: Option<u64>,
        /// Print the latest night's summary and stop (the morning after).
        #[arg(long)]
        summary: bool,
        /// With --summary: this night (YYYY-MM-DD) instead of the latest.
        #[arg(long)]
        night: Option<chrono::NaiveDate>,
    },
    /// Process every data folder under a root: grade, calibrate, register,
    /// integrate, mosaic and render each target, then write a gallery
    /// (`autopilot.html` in the workspace). Estimates first; nothing in
    /// the data folders changes (milestone 14).
    Autopilot {
        root: PathBuf,
        /// Processing workspace (default: process.workspace in the config).
        #[arg(long)]
        workspace: Option<PathBuf>,
        /// Show the folders and the estimate only.
        #[arg(long)]
        dry_run: bool,
        /// Only folders whose name contains this.
        #[arg(long)]
        only: Option<String>,
        /// Stop after integrating (no mosaics or images).
        #[arg(long)]
        no_images: bool,
    },
    /// Saved recipes for the Image stage.
    Recipe {
        #[command(subcommand)]
        what: Option<RecipeCmd>,
    },
    /// Print the effective configuration as TOML.
    Config { dir: Option<PathBuf> },
    /// List the caches and their sizes, or clean them up.
    Cache {
        #[command(subcommand)]
        what: Option<CacheCmd>,
    },
    /// Start again from a fresh scan: empty every folder's measurements,
    /// maps and thumbnails, remove the caches of folders that no longer
    /// exist, move the library index aside (kept as a backup), and clear
    /// processing outputs in the workspace. Your data is never touched, and
    /// your decisions stay unless --decisions. Then run `crux library
    /// refresh` and open your folders.
    Fresh {
        /// Workspace outputs to clear: "intermediates" (registered and
        /// calibrated frames, network and preview caches; masters and
        /// images stay), "all" (everything crux made there, images and
        /// recipes too) or "keep".
        #[arg(long, default_value = "intermediates")]
        work: String,
        /// More workspaces to clear, besides process.workspace.
        #[arg(long = "workspace")]
        workspaces: Vec<PathBuf>,
        /// Also forget your decisions: kept and rejected frames, flat
        /// choices, panel corrections, layout rules and saved views.
        #[arg(long)]
        decisions: bool,
        /// Keep the library index and flat library.
        #[arg(long)]
        keep_index: bool,
        /// Only show what would be cleared.
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        yes: bool,
    },
    /// Validation tools for processing.
    #[command(hide = true)]
    Dev {
        #[command(subcommand)]
        cmd: dev::DevCmd,
    },
    /// Write a stretched PNG preview of a frame with detected trails (red) and stars (green).
    Preview {
        file: PathBuf,
        out: PathBuf,
        /// Longest side of the PNG in pixels.
        #[arg(long, default_value_t = 1600)]
        size: usize,
    },
    /// Show one file's metadata, classification, measurements and basic pixel statistics.
    Info {
        file: PathBuf,
        /// Also print every header keyword.
        #[arg(long)]
        header: bool,
    },
}

#[derive(Subcommand)]
enum RecipeCmd {
    /// Saved recipes.
    List,
    /// Print a recipe (saved name or file) as TOML.
    Show { name: String },
}

#[derive(Subcommand)]
enum CacheCmd {
    /// Every folder's cache: size, what can be rebuilt, last opened.
    List,
    /// Remove what crux can rebuild (measurements, maps, thumbnails) for
    /// these data folders; the next scan measures again. Your decisions
    /// and move journals stay.
    Empty {
        dirs: Vec<PathBuf>,
        /// Every folder's cache.
        #[arg(long)]
        all: bool,
    },
    /// Remove caches completely, decisions and journals included.
    Forget {
        dirs: Vec<PathBuf>,
        /// Caches whose data folder no longer exists.
        #[arg(long)]
        missing: bool,
        #[arg(long)]
        yes: bool,
    },
    /// Remove the old cache (`.crux`) from inside these data folders, once
    /// it has been copied across.
    RemoveOld {
        dirs: Vec<PathBuf>,
        /// Every old cache crux knows of.
        #[arg(long)]
        all: bool,
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Subcommand)]
enum LibCmd {
    /// Measure the flat sets under <dir> and add them to the library index.
    /// Files stay where they are; moving them into the library is a plan action.
    Add { dir: PathBuf },
    /// Scan every folder inside the search folders (config library.search,
    /// default the folder holding the library: C:\PixInsight) for flats,
    /// darks, bias and dark flats, in place. Flat sets go into the library;
    /// folders with darks or bias are remembered, so processing any folder
    /// finds them. Nothing is moved or written in the data folders. With
    /// --cache-dir, each folder's cache is a subfolder of it.
    Refresh {
        /// Search these instead of library.search.
        roots: Vec<PathBuf>,
    },
    /// List the library: sets, optical trains (by dust), sets that can be combined.
    List,
    /// The targets in the library folders (library.search, or the folders
    /// given), from headers: folders, hours by filter, nights, and notes
    /// where names and sky positions disagree. Reads nothing else.
    Targets {
        roots: Vec<PathBuf>,
        /// Also write the result as JSON.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Plan moving a target found in several folders into one:
    /// <library folder>\<target>\<folder>\…  Writes the plan; apply it with
    /// `crux apply <plan>` (undo with `crux undo <journal>`).
    Combine {
        /// The target's name, as `crux library targets` shows it.
        target: String,
        /// Library folders (default library.search).
        #[arg(long = "in")]
        roots: Vec<PathBuf>,
        /// Where to write the plan (default <data folder>\plans).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Start the library index and flat library again: the database is
    /// moved aside as index.db.<date-time>.bak, then rebuilt as folders are
    /// opened and the calibration library is refreshed.
    Reset {
        #[arg(long)]
        yes: bool,
    },
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    if !cli.no_cache && cli.cache_dir.is_none() && !matches!(cli.cmd, Cmd::Cache { .. } | Cmd::Fresh { .. }) {
        if let Ok((cfg, _)) = Config::load(cli.config.as_deref(), None) {
            let (n, b) = crux_core::cachedir::auto_clean(cfg.cache.unused_days);
            if n > 0 {
                eprintln!("Emptied {n} cache(s) unused for {} days ({})", cfg.cache.unused_days, report::bytes(b));
            }
        }
    }
    match &cli.cmd {
        Cmd::Scan { dir, headers_only, list, json } => {
            let (_, t) = triage(&cli, dir, !headers_only)?;
            report::print_scan(&t.scan, *list);
            report::print_analysis_stats(&t);
            if let Some(p) = json {
                std::fs::write(p, serde_json::to_vec_pretty(&t.scan)?).with_context(|| format!("write {}", p.display()))?;
                println!("\nWrote {}", p.display());
            }
        }
        Cmd::Report { dir, headers_only, csv, html, json } => {
            let (cfg, t) = triage(&cli, dir, !headers_only)?;
            report::print_report(&t, &cfg);
            if let Some(p) = html {
                html::write(&t, p)?;
                println!("\nWrote {}", p.display());
            }
            if let Some(p) = csv {
                report::write_csv(&t, &cfg, p)?;
                println!("\nWrote {}", p.display());
                if !t.cal.reports.is_empty() {
                    let cp = p.with_extension("calibration.csv");
                    report::write_cal_csv(&t, &cp)?;
                    println!("Wrote {}", cp.display());
                }
            }
            if let Some(p) = json {
                let v = serde_json::json!({
                    "scan": t.scan,
                    "grouping": t.grouping,
                    "reports": t.reports,
                    "metrics": t.metrics.iter().map(|(k, v)| (k.to_string(), v)).collect::<std::collections::BTreeMap<_, _>>(),
                    "calibration": t.cal,
                    "calibration_metrics": t.cal_metrics.iter().map(|(k, v)| (k.to_string(), v)).collect::<std::collections::BTreeMap<_, _>>(),
                });
                std::fs::write(p, serde_json::to_vec_pretty(&v)?).with_context(|| format!("write {}", p.display()))?;
                println!("\nWrote {}", p.display());
            }
        }
        Cmd::Library { cmd } => match cmd {
            LibCmd::Add { dir } => {
                let (cfg, t) = triage(&cli, dir, true)?;
                let root = cfg.library.root.clone();
                if let Some(e) = &t.library_error {
                    anyhow::bail!("can't read the library: {e}");
                }
                let mut lib = t.library.clone().unwrap_or_else(|| crux_core::library::Library::new(&cfg.library));
                let sets = crux_core::library::sets_from_triage(&t, &root);
                if sets.is_empty() {
                    println!("No flat sets found under {}", dir.display());
                    return Ok(());
                }
                let (mut added, mut updated) = (0, 0);
                for (set, map) in sets {
                    let label = set.label();
                    let grade = set.grade;
                    if lib.upsert(set, &map)? {
                        added += 1;
                        println!("  added    {label:<32} {}", grade.name());
                    } else {
                        updated += 1;
                        println!("  updated  {label:<32} {}", grade.name());
                    }
                }
                lib.save()?;
                println!("\n{added} added, {updated} updated in {}", crux_core::library::Library::db_path(&cfg.library.dir).display());
                report::print_library(&lib, &cfg);
            }
            LibCmd::Refresh { roots } => {
                let (mut cfg, _) = Config::load(cli.config.as_deref(), None)?;
                if let Some(l) = &cli.library {
                    cfg.library.use_folder(l);
                }
                let roots = if roots.is_empty() { cfg.library.search_roots() } else { roots.clone() };
                anyhow::ensure!(!roots.is_empty(), "nothing to search: give folders, or set library.search in crux.toml");
                println!("Searching {} for flats, darks and bias", roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", "));
                let current = std::sync::Mutex::new(String::new());
                let rep = crux_core::library::refresh(
                    &cfg,
                    &roots,
                    if cli.no_cache { None } else { cli.cache_dir.as_deref() },
                    &|e| match e {
                        crux_core::library::RefreshEvent::Folder { index, total, dir } => {
                            report::clear_progress();
                            *current.lock().unwrap() = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                            eprint!("\r  [{index}/{total}] {}", current.lock().unwrap());
                        }
                        crux_core::library::RefreshEvent::Progress { done, total } => {
                            eprint!("\r  {}: calibration frames {done}/{total}   ", current.lock().unwrap());
                        }
                    },
                    &std::sync::atomic::AtomicBool::new(false),
                )?;
                eprintln!("\r{:72}", "");
                for f in &rep.cal_folders {
                    let mut what = Vec::new();
                    for (n, w) in [(f.flats, "flats"), (f.darks, "darks"), (f.bias, "bias"), (f.dark_flats, "dark flats")] {
                        if n > 0 {
                            what.push(format!("{n} {w}"));
                        }
                    }
                    println!("  {:<36} {}{}", f.folder.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), what.join(", "), if f.masters > 0 { format!(" ({} masters)", f.masters) } else { String::new() });
                }
                for a in &rep.added {
                    println!("  added    {a}");
                }
                for r in &rep.removed {
                    println!("  removed  {r}");
                }
                for (r, n) in &rep.loose {
                    println!("  note: {n} image file{} directly in {} not scanned: put them in a folder inside it", if *n == 1 { "" } else { "s" }, r.display());
                }
                for e in rep.errors.iter().take(10) {
                    println!("  error: {e}");
                }
                println!("\n{} ({} frames measured)\nWrote {}", rep.summary(), rep.measured, rep.index.display());
                if let Some(lib) = crux_core::library::Library::load(&cfg.library)? {
                    report::print_library(&lib, &cfg);
                }
            }
            LibCmd::List => {
                let (mut cfg, _) = Config::load(cli.config.as_deref(), None)?;
                if let Some(l) = &cli.library {
                    cfg.library.use_folder(l);
                }
                match crux_core::library::Library::load(&cfg.library)? {
                    Some(lib) => report::print_library(&lib, &cfg),
                    None => println!("No flat library in {} yet (fill it with: crux library refresh)", crux_core::library::Library::db_path(&cfg.library.dir).display()),
                }
            }
            LibCmd::Targets { roots, json } => {
                let (cfg, lt) = library_targets(&cli, roots)?;
                report::print_targets(&lt);
                let _ = cfg;
                if let Some(p) = json {
                    std::fs::write(p, serde_json::to_vec_pretty(&lt)?).with_context(|| format!("write {}", p.display()))?;
                    println!("\nWrote {}", p.display());
                }
            }
            LibCmd::Combine { target, roots, out } => {
                let (cfg, lt) = library_targets(&cli, roots)?;
                let key = crux_core::targets::normalise(target).to_lowercase();
                let Some(ti) = lt.targets.iter().position(|t| t.name.to_lowercase() == key || t.name.eq_ignore_ascii_case(target)) else {
                    anyhow::bail!("no target called {target:?} in {} (see: crux library targets)", lt.roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", "));
                };
                let plan = lt.combine_plan(ti, &cfg, if cli.no_cache { None } else { cli.cache_dir.as_deref() })?;
                let path = match out {
                    Some(p) => p.clone(),
                    None => {
                        let dir = crux_core::paths::data_dir().context("no per-user data folder")?.join("plans");
                        std::fs::create_dir_all(&dir)?;
                        dir.join(format!("combine-{}-{}.json", crux_core::targets::folder_name(&lt.targets[ti].name), chrono::Local::now().format("%Y%m%d-%H%M%S")))
                    }
                };
                std::fs::write(&path, serde_json::to_vec_pretty(&plan)?).with_context(|| format!("write {}", path.display()))?;
                let c = lt.combine(ti).expect("combinable");
                println!("Combine {} into {}", lt.targets[ti].name, c.dest.display());
                for &u in &c.moving {
                    let n = plan.moves.iter().filter(|m| m.from.starts_with(&lt.units[u].path)).count();
                    println!("  {:<40} → {}  ({n} files)", lt.units[u].name, c.dest.join(&lt.units[u].name).display());
                }
                if let Some(u) = c.existing {
                    println!("  {:<40}   already there", lt.units[u].name);
                }
                for n in &plan.notes {
                    println!("  note: {n}");
                }
                for c in &plan.conflicts {
                    println!("  CLASH: {c}");
                }
                println!("\n{} moves, {}. Wrote {}", plan.moves.len(), report::bytes(plan.total_bytes()), path.display());
                println!("Apply it with: crux apply \"{}\"", path.display());
            }
            LibCmd::Reset { yes } => {
                let ix = crux_core::cachedir::default_index(cli.cache_dir.as_deref()).context("no per-user data folder")?;
                let db = crux_core::index::Index::db_path(&ix);
                if !db.is_file() && !ix.is_file() {
                    println!("No library index at {} — nothing to reset.", db.display());
                    return Ok(());
                }
                if !yes {
                    use std::io::IsTerminal;
                    anyhow::ensure!(std::io::stdin().is_terminal(), "add --yes to reset without a terminal");
                    eprint!("Move {} aside and start the library index and flat library again? Type yes: ", db.display());
                    let mut s = String::new();
                    std::io::stdin().read_line(&mut s)?;
                    anyhow::ensure!(s.trim() == "yes", "not reset");
                }
                match crux_core::index::Index::reset(&ix)? {
                    Some(b) => println!("Moved the old index to {}\nOpen your folders and run `crux library refresh` to rebuild it.", b.display()),
                    None => println!("Nothing to reset."),
                }
            }
        },
        Cmd::Plan { dir, out, no_rejects, no_retire, organise, to_library, combine, keep, reject, retire_set, keep_set } => {
            let (cfg, t) = triage(&cli, dir, true)?;
            let dec = if cli.no_cache { None } else { crux_core::decisions::Decisions::load(&cache_dir(&cli, dir)).ok() };
            let mut grades = std::collections::HashMap::new();
            if let Some(d) = &dec {
                for (k, o) in &d.overrides {
                    grades.insert(k.clone(), o.grade);
                }
            }
            let opts = crux_core::plan::PlanOptions {
                panels: dec.as_ref().map(|d| d.panels.iter().map(|(k, v)| (k.clone(), *v)).collect()).unwrap_or_default(),
                grades,
                rejects: !no_rejects,
                retire: !no_retire,
                organise: *organise,
                to_library: *to_library,
                combine: *combine,
                keep: keep.clone(),
                reject: reject.clone(),
                retire_sets: retire_set.clone(),
                keep_sets: keep_set.clone(),
            };
            let cache_dir = (!cli.no_cache).then(|| cache_dir(&cli, dir));
            let plan = crux_core::plan::build(&t, &cfg, &opts, cache_dir.as_deref());
            let path = match out {
                Some(p) => p.clone(),
                None => {
                    let base = cache_dir.clone().unwrap_or_else(std::env::temp_dir);
                    base.join("plans").join(format!("plan-{}.json", chrono::Local::now().format("%Y%m%d-%H%M%S")))
                }
            };
            report::print_plan(&plan);
            if plan.moves.is_empty() {
                println!("\nNothing to move.");
                return Ok(());
            }
            if let Some(d) = path.parent() {
                std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))?;
            }
            std::fs::write(&path, serde_json::to_vec_pretty(&plan)?).with_context(|| format!("write {}", path.display()))?;
            println!("\nPlan written to {}", path.display());
            if plan.conflicts.is_empty() {
                println!("Nothing has moved. To apply it: crux apply \"{}\"", path.display());
            } else {
                println!("It can't be applied until the clashes above are resolved.");
            }
        }
        Cmd::Apply { plan, yes, journal, force_copy } => {
            let p: crux_core::plan::Plan =
                serde_json::from_slice(&std::fs::read(plan).with_context(|| format!("read {}", plan.display()))?).with_context(|| format!("parse {}", plan.display()))?;
            if p.version != crux_core::plan::PLAN_VERSION {
                anyhow::bail!("{} is plan version {}, this crux applies version {}", plan.display(), p.version, crux_core::plan::PLAN_VERSION);
            }
            report::print_plan(&p);
            let problems = crux_core::moves::check(&p);
            if !problems.is_empty() {
                println!("\nCan't apply this plan; nothing has moved:");
                for x in &problems {
                    println!("  ✗ {x}");
                }
                anyhow::bail!("{} problem(s)", problems.len());
            }
            if p.moves.is_empty() {
                println!("\nNothing to move.");
                return Ok(());
            }
            let journal = journal.clone().unwrap_or_else(|| plan.with_extension("journal.jsonl"));
            confirm(*yes, &format!("Move {} files ({})?", p.moves.len(), report::bytes(p.total_bytes())))?;
            let o = crux_core::moves::apply(&p, Some(plan), &journal, &crux_core::moves::Options { force_copy: *force_copy, progress: None })?;
            println!(
                "\nMoved {} files ({}), {} by verified copy.",
                o.moved,
                report::bytes(o.bytes),
                o.copied
            );
            for n in &o.notes {
                println!("  {n}");
            }
            println!("Journal: {}\nTo reverse it: crux undo \"{}\"", o.journal.display(), o.journal.display());
        }
        Cmd::Undo { journal, yes, force_copy } => {
            let records = crux_core::moves::read_journal(journal)?;
            let moves: Vec<_> = records.iter().filter(|r| matches!(r, crux_core::moves::Record::Move { .. })).collect();
            println!("{} moves to reverse from {}", moves.len(), journal.display());
            let problems = crux_core::moves::check_undo(&records);
            if !problems.is_empty() {
                println!("\nCan't undo; nothing has moved:");
                for x in &problems {
                    println!("  ✗ {x}");
                }
                anyhow::bail!("{} problem(s)", problems.len());
            }
            if moves.is_empty() {
                println!("Nothing to undo.");
                return Ok(());
            }
            confirm(*yes, &format!("Move {} files back?", moves.len()))?;
            let undo_path = crux_core::moves::undo_path(journal);
            let o = crux_core::moves::undo(journal, &undo_path, &crux_core::moves::Options { force_copy: *force_copy, progress: None })?;
            println!("\nMoved {} files back ({}).", o.moved, report::bytes(o.bytes));
            for n in &o.notes {
                println!("  {n}");
            }
            println!("Undo journal: {}", o.journal.display());
        }
        Cmd::Info { file, header } => info(&cli, file, *header)?,
        Cmd::Find { dir, query, headers_only } => {
            let (_, t) = triage(&cli, dir, !headers_only)?;
            let q = crux_core::query::parse(query);
            for u in &q.unknown {
                eprintln!("ignored {u:?}: not a filter crux knows");
            }
            let recs = crux_core::index::records(&t, (!headers_only).then_some(&t.reports[..]));
            let hits: Vec<&crux_core::index::IndexFile> = recs.iter().filter(|r| q.matches(r)).collect();
            for r in &hits {
                let g = r.grade.map(|g| g.name()).unwrap_or("-");
                let tr = r.t.map(|v| format!("{v:.2}")).unwrap_or_else(|| "-".into());
                let fw = r.fwhm.map(|v| format!("{v:.2}")).unwrap_or_else(|| "-".into());
                println!("{:<7} T {:>5}  FWHM {:>5}  {}", g, tr, fw, r.path.strip_prefix(dir).unwrap_or(&r.path).display());
            }
            let bytes: u64 = hits.iter().map(|r| r.size).sum();
            println!("\n{} of {} files match ({})", hits.len(), recs.len(), report::bytes(bytes));
        }
        Cmd::Darks { dir, also } => darks(&cli, dir, also)?,
        Cmd::Process { dir, workspace, keep_calibrated, no_warn, reference, dry_run, also, no_integrate } => process(&cli, dir, workspace.as_deref(), *keep_calibrated, *no_warn, reference.clone(), *dry_run, also, *no_integrate)?,
        Cmd::RcAstro { image, out, bxt, sharpen_stars, adjust_halos, sharpen_nonstellar, psf, nxt, denoise, sxt, stars, device } => rc_astro(image.as_deref(), out.as_deref(), bxt.as_deref(), *sharpen_stars, *adjust_halos, *sharpen_nonstellar, *psf, *nxt, *denoise, *sxt, *stars, device.as_deref())?,
        Cmd::Collect { dir, out, workspace, max_mb, no_quarters, keep_stacks, overwrite, registered, yes } => collect(&cli, dir, out.as_deref(), workspace.as_deref(), *max_mb, *no_quarters, *keep_stacks, *overwrite, *registered, *yes)?,
        Cmd::Solve { dir, only, force } => solve(&cli, dir, only.as_deref(), *force)?,
        Cmd::Dev { cmd } => dev::run(cmd, &|d| triage(&cli, d, true).map(|x| x.1))?,
        Cmd::Preview { file, out, size } => preview(&cli, file, out, *size)?,
        Cmd::Cache { what } => cache_cmd(&cli, what.as_ref().unwrap_or(&CacheCmd::List))?,
        Cmd::Fresh { work, workspaces, decisions, keep_index, dry_run, yes } => {
            use crux_core::fresh::{self, Work};
            let (cfg, _) = Config::load(cli.config.as_deref(), None)?;
            let work = match work.to_lowercase().as_str() {
                "intermediates" => Work::Intermediates,
                "all" => Work::All,
                "keep" | "none" => Work::Keep,
                w => anyhow::bail!("--work {w}: use intermediates, all or keep"),
            };
            let mut ws = vec![cfg.process.workspace.clone()];
            ws.extend(workspaces.iter().cloned());
            let ix = crux_core::cachedir::default_index(cli.cache_dir.as_deref());
            let s = fresh::survey(ix.as_deref(), cli.cache_dir.as_deref(), &ws, &cfg.library.search_roots());
            let o = fresh::Options { measurements: true, missing: true, index: !keep_index, work, decisions: *decisions };
            let b = report::bytes;
            let existing = s.caches.iter().filter(|c| c.folder_exists).count() + usize::from(s.explicit.is_some());
            println!("Starting fresh clears:");
            println!("  measurements, maps and thumbnails of {existing} folder(s): {}", b(s.measurement_bytes()));
            if s.missing().count() > 0 {
                println!("  caches of {} folder(s) that no longer exist: {}", s.missing().count(), b(s.missing_bytes()));
            }
            match (&s.index, o.index) {
                (Some(ix), true) => println!("  the library index and flat library, moved aside as a backup: {}", crux_core::index::Index::db_path(ix).display()),
                (Some(_), false) => println!("  (the library index stays)"),
                (None, _) => println!("  (no library index yet)"),
            }
            for w in &s.workspaces {
                let n = s.work.iter().filter(|f| f.dir.starts_with(w)).count();
                if n == 0 {
                    println!("  (nothing crux made in {})", w.display());
                    continue;
                }
                match work {
                    Work::Keep => println!("  (processing outputs in {} stay)", w.display()),
                    Work::Intermediates => println!("  registered and calibrated frames, network and preview caches in {} ({n} folder(s)): {}", w.display(), b(s.work_bytes(work))),
                    Work::All => println!("  everything crux made in {} ({n} folder(s), images and recipes included): {}", w.display(), b(s.work_bytes(work))),
                }
            }
            for r in &s.refused {
                println!("  left alone: {r}");
            }
            println!("  your decisions: {}", if *decisions { format!("removed ({})", b(s.decision_bytes())) } else { "kept".into() });
            println!("Your data isn't touched. About {} would be freed.", b(s.bytes(&o)));
            if *dry_run {
                return Ok(());
            }
            confirm(*yes, "Start fresh?")?;
            let d = fresh::run(&s, &o)?;
            println!("{}", d.summary());
            for e in &d.errors {
                println!("  {e}");
            }
            if let Some(bak) = &d.index_backup {
                println!("The old index is at {}", bak.display());
            }
            println!("Next: `crux library refresh`, then open your folders.");
        }
        Cmd::Watch { dir, interval, duration, summary, night } => watch(&cli, dir, *interval, *duration, *summary, *night)?,
        Cmd::Autopilot { root, workspace, dry_run, only, no_images } => autopilot(&cli, root, workspace.as_deref(), *dry_run, only.as_deref(), *no_images)?,
        Cmd::Mosaic { dir, workspace, target, bin, feather, no_distortion, no_match, no_curvature, force } => {
            let (cfg, _) = Config::load(cli.config.as_deref(), Some(dir))?;
            let mut s = cfg.mosaic.clone();
            if let Some(b) = bin {
                s.bin = *b;
            }
            if let Some(f) = feather {
                s.feather_arcmin = *f;
            }
            if *no_distortion {
                s.distortion = false;
            }
            if *no_match {
                s.match_brightness = false;
            }
            if *no_curvature {
                s.curvature = false;
            }
            let ws = workspace.clone().unwrap_or_else(|| cfg.process.workspace.clone());
            let out = crux_core::process::job::out_dir(dir, &ws);
            mosaic(&out, target.as_deref(), &s, *force, cfg.process.fits)?
        }
        Cmd::Image { dir, workspace, object, recipe, preset, no_gradient, stars, one_stretch, star_size, star_brightness, save_recipe, denoise, sharpen, sharpen_amount, engine, no_enhance, masters, size } => image(&cli, dir, workspace.as_deref(), object.as_deref(), recipe.as_deref(), preset.as_deref(), *no_gradient, &StarArgs { mode: stars.clone(), one_stretch: *one_stretch, size: *star_size, brightness: *star_brightness }, save_recipe.as_deref(), &EnhanceArgs { denoise: *denoise, sharpen: sharpen.clone(), sharpen_amount: *sharpen_amount, engine: engine.clone(), off: *no_enhance }, *masters, *size)?,
        Cmd::Recipe { what } => match what.as_ref().unwrap_or(&RecipeCmd::List) {
            RecipeCmd::List => {
                let l = crux_core::render::recipe::list();
                if l.is_empty() {
                    println!("No saved recipes in {} (save one with crux image --save-recipe NAME, or from the GUI)", crux_core::render::recipe::dir().map(|d| d.display().to_string()).unwrap_or_default());
                }
                for (n, p) in l {
                    println!("{n:<28} {}", p.display());
                }
            }
            RecipeCmd::Show { name } => print!("{}", crux_core::render::recipe::find(name)?.to_toml()),
        },
        Cmd::Config { dir } => {
            let (cfg, from) = Config::load(cli.config.as_deref(), dir.as_deref())?;
            match from {
                Some(p) => println!("# loaded from {}", p.display()),
                None => println!("# defaults (no crux.toml found)"),
            }
            println!("# computing on: {}", crux_core::gpu::describe());
            print!("{}", cfg.to_toml());
        }
    }
    Ok(())
}

fn info(cli: &Cli, file: &Path, show_header: bool) -> Result<()> {
    let dir = file.parent().unwrap_or(Path::new("."));
    let (cfg, _) = Config::load(cli.config.as_deref(), Some(dir))?;
    let t = std::time::Instant::now();
    let mut m = crux_core::meta::read_meta(file)?;
    crux_core::meta::fill_from_name(&mut m);
    let t_head = t.elapsed();
    let c = crux_core::classify::classify(&m, dir, &cfg);
    println!("{}", file.display());
    println!("  {}x{} {:?} sample {}, {:.1} MB", m.width, m.height, m.format, m.sample, m.file_size as f64 / 1e6);
    println!(
        "  type {} ({:?}), from {}; local {} night {}",
        c.frame_type.name(),
        c.role,
        c.source.map(|s| s.name()).unwrap_or("nothing"),
        c.local_time.map(|t| t.to_string()).unwrap_or("?".into()),
        c.night.map(|n| n.to_string()).unwrap_or("?".into())
    );
    println!(
        "  filter {:?} exptime {:?} gain {:?} offset {:?} temp {:?} object {:?}",
        m.filter, m.exptime, m.gain, m.offset, m.temp(), m.object
    );
    if show_header {
        for card in &m.header.cards {
            println!("    {:<8} = {:<30} {}", card.key, card.value, card.comment);
        }
    }
    let t = std::time::Instant::now();
    let img = crux_core::read_image(file)?;
    let t_pix = t.elapsed();
    let mut s: Vec<f32> = img.data.iter().step_by(7).copied().collect();
    s.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| s[((s.len() - 1) as f64 * p) as usize];
    let k = img.adu_scale;
    println!(
        "  pixels (ADU): min {:.1}  median {:.1}  99.9% {:.1}  max {:.1}",
        q(0.0) * k,
        q(0.5) * k,
        q(0.999) * k,
        q(1.0) * k
    );
    let img = match m.bayer() {
        Some(b) => {
            println!("  one-shot colour ({b:?}): measured on its luminance");
            crux_core::cfa::measure_image(&img)
        }
        None => img,
    };
    let a = crux_core::analysis::analyze_light(&img, &cfg);
    {
        use crux_core::stars::*;
        let bg = crux_core::background::Background::estimate(&img, cfg.analysis.mesh_cell);
        let d = detect(&img, &bg, &crux_core::analysis::detect_params(&cfg, img.adu_scale));
        let n = |b: u8| d.stars.iter().filter(|s| s.flags & b != 0).count();
        println!(
            "  measured {} of {} components: {} clean, {} hot skipped, {} blended, {} saturated, {} failed",
            d.stars.len(),
            d.n_components,
            d.stars.iter().filter(|s| s.flags == 0).count(),
            d.n_hot,
            n(BLENDED),
            n(SATURATED),
            n(FAILED)
        );
    }
    let f = |v: Option<f32>| v.map(|x| format!("{x:.2}")).unwrap_or("-".into());
    println!(
        "  stars {} clean ({} for shape, {} saturated, {} components); FWHM {} HFR {} ecc {} PA {} coherence {} radial {}",
        a.n_stars,
        a.n_shape,
        a.n_saturated,
        a.n_components,
        f(a.fwhm),
        f(a.hfr),
        f(a.ecc),
        f(a.pa),
        f(a.coherence),
        f(a.radial)
    );
    println!("  regions (FWHM / ecc, top-left first):");
    for row in a.regions.chunks(3) {
        println!("    {}", row.iter().map(|r| format!("{:>5} / {:<5} ({:>4})", f(r.fwhm), f(r.ecc), r.n)).collect::<Vec<_>>().join("   "));
    }
    println!("  sky {:.1} ADU, noise {:.1} ADU, plane Δx {:.1} Δy {:.1} ADU", a.sky_adu, a.noise_adu, a.grad_x_adu, a.grad_y_adu);
    if let Some(t) = m.date_obs {
        let ctx = crux_core::astro::SkyContext::at(
            t,
            m.sitelat.unwrap_or(cfg.site.latitude),
            m.sitelong.unwrap_or(cfg.site.longitude),
            m.ra.zip(m.dec),
        );
        println!(
            "  Sun {:.1}° ({}), Moon {:.1}° alt, {:.0}% lit, {} from target; target alt {}",
            ctx.sun_alt,
            ctx.twilight.name(),
            ctx.moon_alt,
            ctx.moon_illum * 100.0,
            ctx.moon_sep.map(|s| format!("{s:.0}°")).unwrap_or("?".into()),
            ctx.target_alt.map(|s| format!("{s:.1}°")).unwrap_or("?".into())
        );
    }
    println!(
        "  header read {:.1} ms, pixel read {:.0} ms, analysis {:.0} ms",
        t_head.as_secs_f64() * 1e3,
        t_pix.as_secs_f64() * 1e3,
        a.seconds * 1e3
    );
    Ok(())
}

fn preview(cli: &Cli, file: &Path, out: &Path, size: usize) -> Result<()> {
    let dir = file.parent().unwrap_or(Path::new("."));
    let (cfg, _) = Config::load(cli.config.as_deref(), Some(dir))?;
    let img = crux_core::read_for_measuring(&crux_core::meta::read_meta(file)?)?;
    let a = crux_core::analysis::analyze_light(&img, &cfg);
    let (w, h, f, grey) = crux_core::stretch::preview(&img, size);
    let mut rgb: Vec<u8> = grey.iter().flat_map(|&v| [v, v, v]).collect();
    let mut put = |x: i64, y: i64, c: [u8; 3]| {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            let i = (y as usize * w + x as usize) * 3;
            rgb[i..i + 3].copy_from_slice(&c);
        }
    };
    let ff = f as f32;
    for s in &a.stars {
        let (cx, cy) = ((s.x / ff) as i64, (s.y / ff) as i64);
        for d in [-3i64, 3] {
            for t in -1..=1 {
                put(cx + d, cy + t, [60, 220, 90]);
                put(cx + t, cy + d, [60, 220, 90]);
            }
        }
    }
    for t in &a.trails {
        let n = ((t.x1 - t.x0).hypot(t.y1 - t.y0) / ff).ceil().max(1.0) as i64;
        for k in 0..=n {
            let u = k as f32 / n as f32;
            let (x, y) = ((t.x0 + u * (t.x1 - t.x0)) / ff, (t.y0 + u * (t.y1 - t.y0)) / ff);
            // Draw beside the trail, not on it, so it stays visible.
            let (nx, ny) = (-(t.y1 - t.y0), t.x1 - t.x0);
            let l = nx.hypot(ny).max(1e-6);
            for off in [-6.0f32, 6.0] {
                put((x + off * nx / l) as i64, (y + off * ny / l) as i64, [255, 60, 60]);
            }
        }
    }
    let file_out = std::fs::File::create(out).with_context(|| format!("create {}", out.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file_out), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&rgb)?;
    println!("Wrote {} ({w}x{h}); {} trails, {} stars marked", out.display(), a.trails.len(), a.stars.len());
    for t in &a.trails {
        println!(
            "  {} at {:.1}°, {:.0} px lit, {:.0} ADU, coverage {:.2}, score {:.0}: ({:.0},{:.0})–({:.0},{:.0})",
            t.kind.name(),
            t.angle,
            t.length,
            t.brightness_adu,
            t.coverage,
            t.score,
            t.x0,
            t.y0,
            t.x1,
            t.y1
        );
    }
    Ok(())
}

/// The library folders' targets: `roots`, else `library.search`.
fn library_targets(cli: &Cli, roots: &[PathBuf]) -> Result<(Config, crux_core::targets::LibraryTargets)> {
    let (mut cfg, _) = Config::load(cli.config.as_deref(), None)?;
    if let Some(l) = &cli.library {
        cfg.library.use_folder(l);
    }
    let roots = if roots.is_empty() { cfg.library.search_roots() } else { roots.to_vec() };
    anyhow::ensure!(!roots.is_empty(), "no library folders: give some, or set library.search in crux.toml");
    let ix = crux_core::cachedir::default_index(cli.cache_dir.as_deref()).and_then(|p| crux_core::index::Index::load(&p).ok());
    let lt = crux_core::targets::scan(&cfg, &roots, ix.as_ref());
    Ok((cfg, lt))
}

/// Darks, bias and dark flats ranked across `also`, the folders beside
/// `dir` and the calibration library's folders (headers only for those;
/// spikes cached in `dir`'s cache).
fn library_fingerprints(cli: &Cli, cfg: &Config, t: &Triage, dir: &Path, also: &[PathBuf]) -> Result<(crux_core::fingerprint::Fingerprints, usize)> {
    let others = crux_core::fingerprint::other_folders(dir, cfg, also);
    let cache = if cli.no_cache { Cache::disabled() } else { Cache::load(&cache_dir(cli, dir)) };
    let fp = crux_core::fingerprint::build(t, dir, &others, &cache, &|d, n| report::progress(crux_core::pipeline::Stage::Calibration, d, n));
    report::clear_progress();
    Ok((fp, others.len()))
}

fn darks(cli: &Cli, dir: &Path, also: &[PathBuf]) -> Result<()> {
    let (cfg, t) = triage(cli, dir, true)?;
    let (fp, n_others) = library_fingerprints(cli, &cfg, &t, dir, also)?;

    let c = &fp.candidates;
    println!("{} darks, bias and dark flats in {} folders ({:.1} s)
", c.len(), n_others + 1, fp.seconds);
    for f in &fp.families {
        let nights = match (f.first, f.last) {
            (Some(a), Some(b)) if a != b => format!("{} to {}", a, b),
            (Some(a), _) => a.to_string(),
            _ => String::new(),
        };
        println!("{} ({} frames, {nights}){}", f.label, f.n, if f.kind == crux_core::FrameType::Light { format!(": {} hot pixels in the {} sampled lights, noise {:.1} ADU", f.lights_hot, f.sample.len(), f.light_noise_adu.unwrap_or(0.0)) } else { String::new() });
        for (title, list, cur) in [(if f.kind == crux_core::FrameType::Light { "Darks" } else { "Dark flats" }, &f.darks, f.current_dark), ("Bias", &f.bias, f.current_bias)] {
            if list.is_empty() {
                println!("  {title}: none in the library");
                continue;
            }
            println!("  {title}:");
            for (k, r) in list.iter().enumerate() {
                let cd = &c[r.cand];
                let tag = if !r.usable { "  -" } else if k == 0 { "  *" } else { "   " };
                let now = if cur == Some(r.cand) { "  [matched now]" } else { "" };
                println!("{tag} {}  {}{now}", cd.label, cd.place());
                println!("       {}", r.summary);
                for x in &r.reasons {
                    println!("       - {x}");
                }
            }
        }
        if let Some(b) = f.better_dark() {
            println!("  Better than what's matched now: {}", c[b.cand].label);
        }
        println!();
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn rc_astro(image: Option<&Path>, out: Option<&Path>, bxt: Option<&str>, ss: Option<f32>, ash: Option<f32>, sn: Option<f32>, psf: Option<f32>, nxt: bool, dn: Option<f32>, sxt: bool, stars: bool, device: Option<&str>) -> Result<()> {
    use crux_core::rcastro::{self, Step};
    use std::io::Write;
    let exe = rcastro::find().ok_or_else(|| anyhow::anyhow!("RC Astro's CLI (rc-astro) isn't installed: get it from your RC Astro account, or set CRUX_RC_ASTRO to rc-astro.exe"))?;
    let cli = rcastro::probe(&exe)?;
    let Some(image) = image else {
        println!("rc-astro {} at {}", cli.version, cli.exe.display());
        for p in &cli.products {
            println!("  {:<4} {:<18} {}", p.key, p.name, if p.licensed { p.status.as_str() } else { "not licensed" });
        }
        if cli.schema != rcastro::SCHEMA {
            println!("note: its interface is version {}, crux was written for {}", cli.schema, rcastro::SCHEMA);
        }
        return Ok(());
    };
    let out = out.ok_or_else(|| anyhow::anyhow!("--out is needed"))?;
    anyhow::ensure!(out != image, "--out must differ from the input");
    let mut steps = Vec::new();
    if let Some(mode) = bxt {
        let mut b = rcastro::Bxt { correct_only: mode.eq_ignore_ascii_case("correct"), ..Default::default() };
        anyhow::ensure!(b.correct_only || mode.eq_ignore_ascii_case("sharpen"), "--bxt is \"correct\" or \"sharpen\"");
        if let Some(v) = ss { b.sharpen_stars = v; }
        if let Some(v) = ash { b.adjust_halos = v; }
        if let Some(v) = sn { b.sharpen_nonstellar = v; }
        b.psf_diameter = psf;
        steps.push(Step::Bxt(b));
    }
    if nxt || dn.is_some() {
        let mut n = rcastro::Nxt::default();
        if let Some(v) = dn { n.denoise = v; }
        steps.push(Step::Nxt(n));
    }
    if sxt || stars {
        steps.push(Step::Sxt(rcastro::Sxt { stars, unscreen: false }));
    }
    anyhow::ensure!(!steps.is_empty(), "choose at least one of --bxt, --nxt, --sxt");
    let t0 = std::time::Instant::now();
    let names: Vec<&str> = steps.iter().map(|s| s.name()).collect();
    println!("{} → {} with {}", image.display(), out.display(), names.join(", then "));
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let files = rcastro::run_chain(&cli, &steps, image, out, device, &|k, e| match e {
        rcastro::Event::Progress(p) => {
            print!("\r  {:<17} {p:5.1}%", steps[k].name());
            let _ = std::io::stdout().flush();
        }
        rcastro::Event::Device(d, n) => println!("  {} on {}{}", steps[k].name(), d.to_uppercase(), if n.is_empty() { String::new() } else { format!(" ({n})") }),
        rcastro::Event::Warning(w) => println!("\n  warning: {w}"),
        rcastro::Event::Status(s) if s == "complete" => println!(),
        rcastro::Event::Status(s) if s.starts_with("downloading") => println!("  {s}"),
        _ => {}
    }, &cancel)?;
    println!("done in {:.1} s:", t0.elapsed().as_secs_f64());
    for f in files {
        println!("  {}", f.display());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collect(cli: &Cli, dir: &Path, out: Option<&Path>, workspace: Option<&Path>, max_mb: f64, no_quarters: bool, keep_stacks: bool, overwrite: bool, registered: bool, yes: bool) -> Result<()> {
    use crux_core::collect;
    anyhow::ensure!(dir.is_dir(), "{} is not a folder", dir.display());
    let (cfg, _) = Config::load(cli.config.as_deref(), Some(dir))?;
    let ws = workspace.map(Path::to_path_buf).unwrap_or_else(|| cfg.process.workspace.clone());
    let name = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "folder".into());
    let out = out.map(Path::to_path_buf).unwrap_or_else(|| ws.join("train").join(crux_core::process::job::safe_name(&name)));
    if out.join("manifest.json").is_file() && !overwrite {
        anyhow::bail!("{} already holds a bundle; pass --overwrite to replace it", out.display());
    }
    // Never write a bundle into a folder holding something else (the
    // default `<workspace>\train\<name>` can be where data was put).
    if !out.join("manifest.json").is_file() && std::fs::read_dir(&out).is_ok_and(|mut d| d.next().is_some()) {
        anyhow::bail!("{} isn't empty and holds no bundle; choose another folder with --out", out.display());
    }
    let o = collect::Options { max_mb, quarters: !no_quarters, keep_stacks, ..Default::default() };
    let (stacks, notes) = collect::find_stacks(dir, &ws, registered)?;
    let usable: Vec<&collect::Stack> = stacks.iter().filter(|s| !s.pair_kinds(o.quarters).is_empty()).collect();
    for n in &notes {
        println!("note: {n}");
    }
    anyhow::ensure!(!usable.is_empty(), "no stacks with {} or more registered frames in {}", 2 * collect::MIN_PART, dir.display());
    println!("{} stacks to collect from:", usable.len());
    for s in &usable {
        let kinds: Vec<&str> = s.pair_kinds(o.quarters).iter().map(|k| k.0).collect();
        println!("  {:<28} {:>4} frames  {}", s.label(), s.frames.len(), kinds.join(", "));
    }
    let per = collect::patches_per_pair(&stacks, &o);
    println!("{per} patch pairs of {0} × {0} px per pair of stacks, up to {max_mb:.0} MB, into {1}", o.patch, out.display());
    println!("The bundle holds image patches, star cutouts and camera details. It holds no file names or paths, site, observer, or day and time.");
    confirm(yes, "Make the bundle?")?;
    if overwrite && out.is_dir() {
        // Only the bundle's own files; anything else in the folder stays.
        for e in std::fs::read_dir(&out)?.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if n == "manifest.json" || n == "bundle.html" || (n.starts_with('s') && n.ends_with(".fits")) {
                std::fs::remove_file(e.path())?;
            }
        }
    }
    let t0 = std::time::Instant::now();
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let usable: Vec<collect::Stack> = usable.into_iter().cloned().collect();
    let m = collect::collect(&usable, &out, &o, notes, &|l| println!("{l}"), &cancel)?;
    let patches: usize = m.sets.iter().flat_map(|s| &s.pairs).map(|p| p.patches.len()).sum();
    let bytes: u64 = std::fs::read_dir(&out)?.flatten().filter(|e| e.path().is_file()).filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum();
    println!("\n{} stacks, {patches} patch pairs, {:.1} MB in {:.0} s: {}", m.sets.len(), bytes as f64 / 1e6, t0.elapsed().as_secs_f64(), out.display());
    println!("Check what's in it: {}", out.join("bundle.html").display());
    Ok(())
}

fn process(cli: &Cli, dir: &Path, workspace: Option<&Path>, keep_calibrated: bool, no_warn: bool, reference: Option<String>, dry_run: bool, also: &[PathBuf], no_integrate: bool) -> Result<()> {
    use crux_core::process::job;
    let (cfg, t) = triage(cli, dir, true)?;
    let decisions = if cli.no_cache { None } else { crux_core::decisions::Decisions::load(&cache_dir(cli, dir)).ok() };
    let mut s = job::Settings::from_config(&cfg);
    if let Some(w) = workspace {
        s.workspace = w.to_path_buf();
    }
    s.keep_calibrated |= keep_calibrated;
    s.include_warn &= !no_warn;
    s.reference = reference;
    s.integrate &= !no_integrate;
    let cat = crux_core::catalog::Catalog::find();
    // Darks or bias missing here: rank the library's.
    let lacking = t.grouping.light_calibration.iter().any(|c| c.darks.is_empty() || c.bias.is_empty());
    let fp = if lacking || !also.is_empty() { Some(library_fingerprints(cli, &cfg, &t, dir, also)?.0) } else { None };
    let plan = job::plan(&t, dir, decisions.as_ref(), &s, cat.as_ref(), fp.as_ref())?;
    println!("Processing {} → {}", dir.display(), plan.out.display());
    println!("  {} lights to calibrate and register, {} left out", plan.n_lights(), plan.n_skipped());
    println!("
Calibration");
    for c in &plan.cals {
        let src = |x: &Option<job::Source>| x.as_ref().map(|s| s.label()).unwrap_or_else(|| "none".into());
        println!("  {}: dark {}, flat {}{}", c.label, src(&c.dark), src(&c.flat), if c.notes.is_empty() { String::new() } else { format!(" ({})", c.notes.join("; ")) });
    }
    for f in &plan.fields {
        println!("
{}: {} lights", f.object, f.lights.len());
        if !f.lights.is_empty() {
            println!("  reference {}", f.reference_path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default());
            let mut rms: Vec<f64> = f.lights.iter().filter(|l| l.frame != f.reference).map(|l| l.reg.rms).collect();
            rms.sort_by(|a, b| a.total_cmp(b));
            if !rms.is_empty() {
                println!("  registration: median {:.3} px rms, worst {:.3} px, {}–{} stars matched", rms[rms.len() / 2], rms[rms.len() - 1], f.lights.iter().map(|l| l.reg.matched).min().unwrap_or(0), f.lights.iter().map(|l| l.reg.matched).max().unwrap_or(0));
            }
        }
        let mut why: std::collections::BTreeMap<&str, usize> = Default::default();
        for (_, w) in &f.skipped {
            *why.entry(w.as_str()).or_default() += 1;
        }
        for (w, n) in why {
            println!("  left out: {n} {w}");
        }
    }
    if !plan.stacks.is_empty() {
        let c = &plan.settings.combine;
        println!(
            "\nIntegration: {} master light{} ({} normalisation, {} weights)",
            plan.stacks.len(),
            if plan.stacks.len() == 1 { "" } else { "s" },
            match c.normalisation {
                crux_core::process::combine::Normalisation::Local => "local",
                crux_core::process::combine::Normalisation::Global => "global",
            },
            match c.weighting {
                crux_core::process::combine::Weighting::Snr => "SNR",
                crux_core::process::combine::Weighting::Exposure => "exposure",
                crux_core::process::combine::Weighting::Equal => "equal",
            }
        );
        for st in &plan.stacks {
            let f = &plan.fields[st.field];
            let secs: f64 = st.lights.iter().map(|&i| f.lights[i].exptime).sum();
            println!("  {} · {}: {} lights, {:.2} h, {}", st.object, st.filter, st.lights.len(), secs / 3600.0, c.rejection_for(st.lights.len()).name());
        }
    }
    for n in &plan.notes {
        println!("note: {n}");
    }
    let pd = job::pending(&plan);
    println!("\nUp to date: {} of {} lights; {} of {} master lights to integrate", plan.n_lights() - pd.lights, plan.n_lights(), pd.stacks, plan.stacks.len());
    for (why, files) in job::pending_reasons(&plan) {
        println!("  out of date: {} ({why})", files.len());
        for f in files.iter().take(5) {
            println!("    {}", f.display());
        }
    }
    if dry_run {
        return Ok(());
    }
    println!();
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let rep = job::run(
        &plan,
        &|e| match e {
            job::Event::Master { label } => {
                report::clear_progress();
                println!("  {label}");
            }
            job::Event::Light { done, total, path, result } => {
                if let Some(err) = &result.error {
                    report::clear_progress();
                    println!("  failed: {}: {err}", path.display());
                }
                report::progress(crux_core::pipeline::Stage::Pixels, done, total);
            }
            job::Event::Integrate { index, total, label, step } => {
                let (what, d, n) = match step {
                    crux_core::process::combine::Step::Measure { done, total } => ("measuring", done, total),
                    crux_core::process::combine::Step::Combine { done, total } => ("combining", done, total),
                };
                report::progress_text(&format!("Integrating {index}/{total} {label}: {what} {d}/{n}"));
            }
            job::Event::Integrated { label, report: r, error, .. } => {
                report::clear_progress();
                match (r, error) {
                    (_, Some(e)) => println!("  {label}: failed: {e}"),
                    (Some(r), None) => print_integration(&label, &r),
                    (None, None) => println!("  {label}: up to date"),
                }
            }
        },
        &cancel,
    )?;
    report::clear_progress();
    println!(
        "Done in {:.1} s: {} processed, {} already up to date, {} failed{}",
        rep.seconds,
        rep.processed,
        rep.up_to_date,
        rep.failed,
        if rep.masters_built.is_empty() { String::new() } else { format!("; built {} master{}", rep.masters_built.len(), if rep.masters_built.len() == 1 { "" } else { "s" }) }
    );
    if rep.integrated + rep.integrations_up_to_date + rep.integration_failed > 0 {
        println!("Integrated {} master light{}, {} already up to date, {} failed", rep.integrated, if rep.integrated == 1 { "" } else { "s" }, rep.integrations_up_to_date, rep.integration_failed);
    }
    if plan.settings.integrate && cfg.mosaic.after_integrate && !crux_core::process::mosaic::groups(&plan.out).is_empty() {
        if crux_core::catalog::Catalog::find().is_some() {
            println!("
Mosaic");
            mosaic(&plan.out, None, &cfg.mosaic, false, cfg.process.fits)?;
        } else {
            println!("note: panels found but no star database to place them (install ASTAP with D50, or set CRUX_CATALOG); skipped the mosaic");
        }
    }
    println!("Output: {}", plan.out.display());
    Ok(())
}

fn capital(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

fn print_integration(label: &str, r: &crux_core::process::combine::Report) {
    let n = r.frames.len();
    let worst = r.frames.iter().min_by(|a, b| a.weight.total_cmp(&b.weight));
    println!(
        "  {label}: {n} frames, {:.2} h, {}; {:.2}% rejected; noise {:.2} ADU (median frame {:.2}): SNR x{:.2}, worth {:.1} frames{}  ({:.1} s)",
        r.total_exposure / 3600.0,
        r.rejection,
        r.rejected_total * 100.0,
        r.noise_master,
        r.noise_typical,
        r.snr_gain,
        r.equivalent_frames,
        r.fwhm.map(|f| format!("; FWHM {f:.2} px")).unwrap_or_default(),
        r.seconds
    );
    if let Some(w) = worst.filter(|w| w.weight < 0.5) {
        println!("      lowest weight {:.2}: {} (signal {:.2}, noise {:.1} ADU)", w.weight, w.name, w.signal, w.noise);
    }
    for w in &r.what_ifs {
        println!("      {}: SNR {:+.1}%{}", capital(&w.label), w.snr_change * 100.0, w.fwhm.map(|f| format!(", FWHM {f:.2} px")).unwrap_or_default());
    }
    println!("      {}", r.out.display());
}

fn hms(deg: f64) -> String {
    let h = deg.rem_euclid(360.0) / 15.0;
    let (hh, m) = (h.floor(), (h.fract() * 60.0));
    format!("{:02}h{:02}m{:04.1}s", hh as u32, m.floor() as u32, m.fract() * 60.0)
}

fn dms(deg: f64) -> String {
    let t = (deg.abs() * 3600.0).round() as u64;
    format!("{}{:02}°{:02}′{:02}″", if deg < 0.0 { "−" } else { "+" }, t / 3600, t / 60 % 60, t % 60)
}

fn solve(cli: &Cli, dir: &Path, only: Option<&str>, force: bool) -> Result<()> {
    let cat = crux_core::catalog::Catalog::find().ok_or_else(|| anyhow::anyhow!("no star database found: install ASTAP with its D50 database, or set CRUX_CATALOG to the folder holding d50_*.1476"))?;
    let (_, t) = triage(cli, dir, true)?;
    let mut index = if cli.no_cache { None } else { Some(crux_core::index::Index::load(&crux_core::cachedir::index_for(&cache_dir(cli, dir)))?) };
    let mut from_index = 0;
    println!("Star database: {} in {}", cat.name.to_uppercase(), cat.dir.display());
    let t0 = std::time::Instant::now();
    let (mut ok, mut fail) = (0, 0);
    let mut ids: Vec<usize> = t.metrics.keys().copied().filter(|&i| only.is_none_or(|o| t.scan.frames[i].meta.path.to_string_lossy().contains(o))).collect();
    ids.sort_by(|a, b| t.scan.frames[*a].meta.path.cmp(&t.scan.frames[*b].meta.path));
    // Two passes: frames that failed before any frame solved are retried
    // with the scale measured since (header focal lengths can be 4% out).
    struct Row {
        i: usize,
        hint: Option<crux_core::solve::Hint>,
        result: Option<crux_core::solve::Solution>,
        secs: f64,
        with_learned: bool,
    }
    let mut learned = crux_core::solve::Learned::default();
    let mut rows: Vec<Row> = Vec::new();
    for i in ids {
        let f = &t.scan.frames[i];
        let m = &t.metrics[&i];
        let Some(hint) = crux_core::solve::Hint::from_meta(&f.meta) else {
            rows.push(Row { i, hint: None, result: None, secs: 0.0, with_learned: false });
            continue;
        };
        let t1 = std::time::Instant::now();
        let known = if force { None } else { index.as_ref().and_then(|ix| ix.solution(&f.meta.path, f.meta.file_size, f.meta.mtime_ns)).map(|s| s.cloned()) };
        let with_learned = learned.knows(&hint);
        let result = match known {
            Some(s) => {
                from_index += 1;
                if let Some(x) = &s {
                    learned.learn(&hint, x);
                }
                s
            }
            None => {
                let s = learned.solve(&cat, &m.stars, m.width as f64, m.height as f64, &hint)?;
                if let Some(ix) = index.as_mut() {
                    ix.set_solution(&f.meta.path, f.meta.file_size, f.meta.mtime_ns, s.clone());
                }
                s
            }
        };
        rows.push(Row { i, hint: Some(hint), result, secs: t1.elapsed().as_secs_f64(), with_learned });
    }
    for r in rows.iter_mut().filter(|r| r.result.is_none() && !r.with_learned) {
        let Some(hint) = r.hint.as_ref().filter(|h| learned.knows(h)) else { continue };
        let f = &t.scan.frames[r.i];
        let m = &t.metrics[&r.i];
        let t1 = std::time::Instant::now();
        r.result = learned.solve(&cat, &m.stars, m.width as f64, m.height as f64, hint)?;
        r.secs += t1.elapsed().as_secs_f64();
        if let Some(ix) = index.as_mut() {
            ix.set_solution(&f.meta.path, f.meta.file_size, f.meta.mtime_ns, r.result.clone());
        }
    }
    for r in &rows {
        let f = &t.scan.frames[r.i];
        let name = f.meta.path.strip_prefix(dir).map(|p| p.display().to_string()).unwrap_or_else(|_| f.meta.file_name().to_string());
        if r.hint.is_none() {
            println!("  {name}: no RA/Dec or pixel scale in the header");
            fail += 1;
            continue;
        }
        match &r.result {
            Some(s) => {
                ok += 1;
                println!(
                    "  {name}: {} {}  rot {:+7.2}°{}  {:.4}″/px  {} stars, {:.2}″ rms; {}{:.1}′ from the mount position  ({:.2} s)",
                    hms(s.ra),
                    dms(s.dec),
                    s.rotation,
                    if s.mirrored { " mirrored" } else { "" },
                    s.scale,
                    s.matched,
                    s.rms,
                    s.target_offset_arcmin.map(|x| format!("{x:.1}′ from the target, ")).unwrap_or_default(),
                    s.offset_arcmin.unwrap_or(0.0),
                    r.secs
                );
            }
            None => {
                fail += 1;
                println!("  {name}: not solved ({} stars)", t.metrics[&r.i].stars.len());
            }
        }
    }
    if let Some(r) = rows.iter().find_map(|r| r.hint.as_ref().filter(|h| learned.knows(h))) {
        let measured = learned.hint(r).scale;
        if (measured / r.scale - 1.0).abs() > 0.01 {
            let f = rows.iter().find_map(|x| x.hint.as_ref().map(|_| &t.scan.frames[x.i].meta)).and_then(|m| m.focallen);
            println!(
                "Measured scale {measured:.4}″/px vs {:.4}″/px from the header{}",
                r.scale,
                f.map(|fl| format!(": a focal length of {:.0} mm, not {fl:.0}", fl * r.scale / measured)).unwrap_or_default()
            );
        }
    }
    println!("{ok} solved, {fail} not, in {:.1} s{}", t0.elapsed().as_secs_f64(), if from_index > 0 { format!(" ({from_index} from the index)") } else { String::new() });
    if let Some(ix) = &index {
        ix.save()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn print_night(t: &Triage, which: Option<chrono::NaiveDate>) {
    use crux_core::watch;
    let Some(night) = which.or_else(|| watch::latest_night(t)) else {
        println!("No lights yet.");
        return;
    };
    let s = watch::night_summary(t, night);
    let span = match (s.first, s.last) {
        (Some(a), Some(b)) => format!(", {}–{}", a.format("%H:%M"), b.format("%H:%M")),
        _ => String::new(),
    };
    println!("Night of {night}{span}: {}", s.headline());
    for (o, f, n, h) in &s.targets {
        println!("  {o} {f}: {n} ({h:.1} h)");
    }
    if !s.reasons.is_empty() {
        println!("  flagged for: {}", s.reasons.iter().map(|(r, n)| format!("{r} {n}")).collect::<Vec<_>>().join(", "));
    }
    for f in &s.flats {
        println!("  {f}");
    }
}

fn frame_line(t: &Triage, i: usize) -> String {
    let f = &t.scan.frames[i];
    let r = t.reports.iter().find(|r| r.frame == i);
    let m = t.metrics.get(&i);
    let mut s = format!(
        "{} {} {} {:.0} s",
        f.class.local_time.map(|x| x.format("%H:%M").to_string()).unwrap_or_else(|| "--:--".into()),
        f.meta.object.as_deref().unwrap_or("?"),
        f.meta.filter.as_deref().unwrap_or(""),
        f.meta.exptime.unwrap_or(0.0)
    );
    if let Some(m) = m {
        if let Some(fw) = m.fwhm {
            s.push_str(&format!(", FWHM {fw:.2} px"));
        }
        s.push_str(&format!(", {} stars", m.n_stars));
    }
    if let Some(r) = r {
        if let Some(tr) = &r.transparency {
            s.push_str(&format!(", T {:.2}", tr.value));
        }
        s.push_str(&format!(": {}", r.grade.name()));
        let why: Vec<&str> = r.flags.iter().filter(|fl| fl.grade != crux_core::grade::Grade::Good).map(|fl| fl.reason.as_str()).collect();
        if !why.is_empty() {
            s.push_str(&format!(" ({})", why.join("; ")));
        }
    } else if f.class.frame_type != crux_core::classify::FrameType::Light {
        s.push_str(&format!(": {:?}", f.class.frame_type));
    }
    s
}

/// Watch a folder: grade new subs as they finish writing.
fn watch(cli: &Cli, dir: &Path, interval: u64, duration: Option<u64>, summary: bool, night: Option<chrono::NaiveDate>) -> Result<()> {
    let (cfg, t) = triage(cli, dir, true)?;
    print_night(&t, night);
    if summary {
        return Ok(());
    }
    let mut w = crux_core::watch::Watcher::new(dir, &cfg);
    println!("Watching {} every {interval} s{}", dir.display(), if duration.is_some() { "" } else { " (Ctrl+C to stop)" });
    let t0 = std::time::Instant::now();
    loop {
        if duration.is_some_and(|d| t0.elapsed().as_secs() >= d) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(interval.max(1)));
        let new = w.poll();
        if new.is_empty() {
            continue;
        }
        let (_, t) = triage(cli, dir, true)?;
        for p in &new {
            let k = crux_core::paths::key(p);
            match t.scan.frames.iter().position(|f| crux_core::paths::key(&f.meta.path) == k) {
                Some(i) => println!("  new: {}", frame_line(&t, i)),
                None => println!("  new: {} (not an image crux reads)", p.display()),
            }
        }
        print_night(&t, None);
    }
    Ok(())
}

/// Process everything under a root.
fn autopilot(cli: &Cli, root: &Path, workspace: Option<&Path>, dry_run: bool, only: Option<&str>, no_images: bool) -> Result<()> {
    use crux_core::autopilot::{self, Event};
    let (cfg, _) = Config::load(cli.config.as_deref(), Some(root))?;
    let ws = workspace.map(Path::to_path_buf).unwrap_or_else(|| cfg.process.workspace.clone());
    let mut folders = autopilot::discover(root);
    if let Some(o) = only {
        let terms: Vec<String> = o.split(',').map(|t| t.trim().to_lowercase()).filter(|t| !t.is_empty()).collect();
        folders.retain(|f| f.file_name().is_some_and(|n| terms.iter().any(|t| n.to_string_lossy().to_lowercase().contains(t))));
    }
    eprintln!("Reading headers in {} folder{}...", folders.len(), if folders.len() == 1 { "" } else { "s" });
    let est = autopilot::estimate(root, &folders, &cfg, cli.cache_dir.as_deref(), &ws);
    for f in &est.folders {
        println!(
            "  {}: {} lights ({} measured), {}{}{}",
            f.dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            f.lights,
            f.measured,
            autopilot::human_bytes(f.light_bytes),
            if f.targets.is_empty() { String::new() } else { format!(", {}", f.targets.join(", ")) },
            if f.processed { ", processed before" } else { "" }
        );
    }
    println!("{}", est.summary());
    if dry_run || est.folders.is_empty() {
        return Ok(());
    }
    let opts = autopilot::Options { workspace: ws.clone(), cache_root: cli.cache_dir.clone(), config: cli.config.clone(), no_images };
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let stage = std::sync::Mutex::new(String::new());
    let rep = autopilot::run_all(
        &est,
        &opts,
        &|e| match e {
            Event::Folder { index, total, dir } => {
                report::clear_progress();
                println!("\n[{index}/{total}] {}", dir.display());
            }
            Event::Stage(s) => {
                report::clear_progress();
                println!("  {s}");
                *stage.lock().unwrap() = s;
            }
            Event::Progress { done, total } => report::progress_text(&format!("  {} {done}/{total}", stage.lock().unwrap())),
            Event::Finished(f) => {
                report::clear_progress();
                match &f.error {
                    Some(e) => println!("  failed: {e}"),
                    None => {
                        let n_img = f.targets.iter().filter(|t| t.image.is_some()).count();
                        println!("  {} lights ({} good, {} warn, {} reject); {} processed, {} up to date; {} image{} in {}", f.lights, f.good, f.warn, f.reject, f.processed, f.up_to_date, n_img, if n_img == 1 { "" } else { "s" }, autopilot::human_time(f.seconds));
                        for t in f.targets.iter().filter(|t| t.error.is_some()) {
                            println!("  {}: image failed: {}", t.object, t.error.as_deref().unwrap_or(""));
                        }
                    }
                }
                for i in &f.inbox {
                    println!("  needs checking: {i}");
                }
            }
        },
        &cancel,
    )?;
    println!("\nDone in {}. Gallery: {}", autopilot::human_time(rep.seconds), autopilot::gallery_path(&ws).display());
    Ok(())
}

/// Assemble every mosaic target in a processing output folder.
fn mosaic(out: &Path, target: Option<&str>, s: &crux_core::process::mosaic::Settings, force: bool, fits: bool) -> Result<()> {
    use crux_core::process::mosaic::{self, Step};
    let groups: Vec<mosaic::Group> = mosaic::groups(out).into_iter().filter(|g| target.is_none_or(|t| g.target.to_lowercase().contains(&t.to_lowercase()))).collect();
    if groups.is_empty() {
        println!("no mosaic targets in {} (needs integrated masters of two or more panels of one target)", out.display());
        return Ok(());
    }
    let cat = crux_core::catalog::Catalog::find().ok_or_else(|| anyhow::anyhow!("no star database found: install ASTAP with its D50 database, or set CRUX_CATALOG to the folder holding d50_*.1476"))?;
    let cancel = std::sync::atomic::AtomicBool::new(false);
    for g in &groups {
        println!("{}: {} panels, {}", g.target, g.panels(), g.filters().join(" "));
        let progress = |st: Step| match st {
            Step::Solve { done, total } if done == total => eprintln!("  solved {total} masters"),
            Step::Warp { filter, done, total } if done == total => eprintln!("  {filter}: warped {total} panels"),
            Step::Skipped { filter } => println!("  {filter}: unchanged (--force to redo)"),
            _ => {}
        };
        let reps = mosaic::assemble(g, out, s, &cat, force, fits, &progress, &cancel)?;
        if fits {
            for r in &reps {
                println!("  {}: FITS copy {}", r.filter, crux_core::fits::copy_path(&r.out).display());
            }
        }
        for r in &reps {
            let c = &r.canvas;
            println!("  {}: {} × {} px at {:.2}\"/px (bin {}), {} panels, {} frames, {:.1} h, {:.0} s → {}", r.filter, c.width, c.height, c.scale, c.bin, r.panels.len(), r.frames, r.exposure / 3600.0, r.seconds, r.out.display());
            let failed: Vec<&mosaic::PanelReport> = r.panels.iter().filter(|p| p.error.is_some()).collect();
            for p in &failed {
                println!("    {}: left out: {}", p.object, p.error.as_deref().unwrap_or(""));
            }
            let mut lin: Vec<f64> = r.panels.iter().filter(|p| p.error.is_none()).map(|p| p.rms_linear).collect();
            let mut fit: Vec<f64> = r.panels.iter().filter(|p| p.error.is_none()).map(|p| p.rms).collect();
            let mut dmax: Vec<f64> = r.panels.iter().filter(|p| p.error.is_none()).map(|p| p.distortion_max).collect();
            let med = |v: &mut Vec<f64>| crux_core::stats::median_f64(v).unwrap_or(f64::NAN);
            println!("    catalogue stars: {:.2} px rms with the plate solution alone, {:.2} px with distortion (up to {:.1} px corrected at the corners), medians over panels", med(&mut lin), med(&mut fit), med(&mut dmax));
            let mut worst: Vec<f64> = r.seams.iter().filter(|s| s.stars >= 5).map(|s| s.star_rms).collect();
            worst.sort_by(f64::total_cmp);
            println!("    overlaps: {} pairs; stars line up to {:.2} px rms (median, worst {:.2} px; panel pixels)", r.seams.len(), r.star_rms(), worst.last().copied().unwrap_or(f64::NAN));
            let mut before: Vec<f64> = r.seams.iter().map(|s| s.before.abs()).collect();
            let mut after: Vec<f64> = r.seams.iter().map(|s| s.after.abs()).collect();
            println!("    seams: median {:.3}σ before matching, {:.3}σ after (worst {:.3}σ)", med(&mut before), med(&mut after), r.worst_seam());
            let mut sc: Vec<(String, f64)> = r.panels.iter().filter(|p| p.error.is_none()).map(|p| (p.object.clone(), p.brightness.scale)).collect();
            sc.sort_by(|a, b| a.1.total_cmp(&b.1));
            if let (Some(lo), Some(hi)) = (sc.first(), sc.last()) {
                println!("    brightness scales {:.3} ({}) to {:.3} ({})", lo.1, lo.0, hi.1, hi.0);
            }
        }
    }
    Ok(())
}

/// `crux image`'s denoise and sharpen options.
struct EnhanceArgs {
    denoise: Option<f32>,
    sharpen: Option<String>,
    sharpen_amount: Option<f32>,
    engine: Option<String>,
    off: bool,
}

impl EnhanceArgs {
    fn apply(&self, e: &mut crux_core::render::enhance::EnhanceSettings) {
        use crux_core::render::enhance::{Engine, Sharpen};
        if self.off {
            e.denoise = false;
            e.sharpen = Sharpen::Off;
        }
        if let Some(a) = self.denoise {
            e.denoise = a > 0.0;
            e.denoise_amount = a.clamp(0.0, 1.0);
        }
        match self.sharpen.as_deref() {
            Some("correct") => e.sharpen = Sharpen::Correct,
            Some("tighten") => e.sharpen = Sharpen::Tighten,
            Some("off") => e.sharpen = Sharpen::Off,
            _ => {}
        }
        if let Some(a) = self.sharpen_amount {
            e.sharpen_amount = a.clamp(0.0, 1.0);
        }
        match self.engine.as_deref() {
            Some("crux") => e.engine = Engine::Crux,
            Some("rc-astro") => e.engine = Engine::RcAstro,
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// `crux image`'s star options.
struct StarArgs {
    mode: Option<String>,
    one_stretch: bool,
    size: Option<f64>,
    brightness: Option<f64>,
}

fn image(cli: &Cli, dir: &Path, workspace: Option<&Path>, object: Option<&str>, recipe: Option<&str>, preset: Option<&str>, no_gradient: bool, stars: &StarArgs, save_recipe: Option<&str>, enhance: &EnhanceArgs, masters: bool, size: Option<usize>) -> Result<()> {
    use crux_core::render::{self, palette, recipe::Recipe};
    let (cfg, _) = Config::load(cli.config.as_deref(), Some(dir))?;
    let ws = workspace.map(Path::to_path_buf).unwrap_or_else(|| cfg.process.workspace.clone());
    // Another program's masters: the target is named after the data folder
    // (the parent of a `master` folder), and so is its workspace folder.
    let named = if masters && dir.file_name().is_some_and(|n| n.eq_ignore_ascii_case("master")) { dir.parent().unwrap_or(dir) } else { dir };
    let out = crux_core::process::job::out_dir(named, &ws);
    let found = if masters {
        let object = named.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "target".into());
        render::targets_from_masters(dir, &object)
    } else {
        render::targets(&out)
    };
    let mut targets: Vec<render::Target> = found.into_iter().filter(|t| object.is_none_or(|o| t.object.to_lowercase().contains(&o.to_lowercase()))).collect();
    // An exact name ("LMC", the mosaic) wins over the names containing it.
    if let Some(o) = object {
        if targets.iter().any(|t| t.object.eq_ignore_ascii_case(o)) {
            targets.retain(|t| t.object.eq_ignore_ascii_case(o));
        }
    }
    anyhow::ensure!(!targets.is_empty(), "{}", if masters { format!("no masterLight files in {}", dir.display()) } else { format!("no integrated masters in {} (run crux process first)", out.display()) });
    let given = recipe.map(crux_core::render::recipe::find).transpose()?;
    let cancel = std::sync::atomic::AtomicBool::new(false);
    for t in &targets {
        let names = t.channels();
        let t0 = std::time::Instant::now();
        let dir_out = t.image_dir(&out);
        let mut r = match &given {
            Some(g) => g.clone(),
            None => Recipe::load(&dir_out.join("recipe.toml")).unwrap_or_else(|_| Recipe::for_channels(&names)),
        };
        if let Some(pn) = preset {
            let kind = palette::kind_of(&names);
            let ps = palette::presets(kind, &names);
            let p = ps.iter().find(|p| p.name.eq_ignore_ascii_case(pn)).ok_or_else(|| anyhow::anyhow!("no preset {pn:?} for {}: {}", t.object, ps.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")))?;
            r.apply_preset(p, &names);
            r.name = p.name.to_string();
        }
        r.fit_to(&names);
        if no_gradient {
            r.gradient.enabled = false;
        }
        if stars.one_stretch {
            r.stars.separate = false;
        }
        if let Some(v) = stars.size {
            r.stars.size = v.clamp(0.05, 3.0);
        }
        if let Some(v) = stars.brightness {
            r.stars.brightness = v.clamp(0.0, 3.0);
        }
        match stars.mode.as_deref() {
            Some("keep") => r.stars.mode = crux_core::render::stars::StarMode::Keep,
            Some("natural") => r.stars.mode = crux_core::render::stars::StarMode::Natural,
            Some("network") => {
                r.stars.mode = crux_core::render::stars::StarMode::Natural;
                r.stars.engine = crux_core::render::stars::StarEngine::Network;
            }
            Some("classic") => {
                r.stars.mode = crux_core::render::stars::StarMode::Natural;
                r.stars.engine = crux_core::render::stars::StarEngine::Classic;
            }
            _ => {}
        }
        if let Some(n) = save_recipe {
            r.name = n.to_string();
        }
        if r.name.is_empty() {
            r.name = if r.mix.preset.is_empty() { "Custom".into() } else { r.mix.preset.clone() };
        }
        println!("{}: {} ({:?}), recipe {:?}", t.object, t.masters.iter().map(|m| format!("{} {}×{:.0} min", m.channel, m.frames, m.exposure / 60.0)).collect::<Vec<_>>().join(", "), palette::kind_of(&names), r.name);
        enhance.apply(&mut r.enhance);
        let enhanced;
        let t = if r.enhance.active() {
            let te = std::time::Instant::now();
            // Only when the step or the 5% band changes (tiles finish fast).
            let last = std::sync::Mutex::new(String::new());
            enhanced = render::enhance::apply(t, &out, &r.enhance, &|p| {
                use std::io::Write;
                let key = format!("{}|{}|{}", p.master, p.step, (p.done * 20.0) as u32);
                let mut l = last.lock().unwrap_or_else(|e| e.into_inner());
                if *l == key {
                    return;
                }
                *l = key;
                eprint!("\r  enhance {} ({}/{}): {} {:3.0}%   ", p.channel, p.master + 1, p.masters, p.step, p.done * 100.0);
                let _ = std::io::stderr().flush();
            }, &cancel)?;
            eprintln!("\r{:72}", "");
            println!("  enhanced: {} in {:.1} s (on {})", r.enhance.describe(), te.elapsed().as_secs_f64(), crux_core::gpu::describe());
            &enhanced
        } else {
            t
        };
        let with_sl;
        let t = if r.stars.engine == crux_core::render::stars::StarEngine::Network && r.stars.mode == crux_core::render::stars::StarMode::Natural && render::stars_possible(&t.channels()) {
            let ts = std::time::Instant::now();
            with_sl = render::starnet::apply(t, &out, &|i, n, f| {
                use std::io::Write;
                eprint!("\r  removing stars {} of {}: {:3.0}%   ", i + 1, n, f * 100.0);
                let _ = std::io::stderr().flush();
            }, &cancel)?;
            eprintln!("\r{:72}", "");
            println!("  stars removed by the network in {:.1} s (on {})", ts.elapsed().as_secs_f64(), crux_core::gpu::describe());
            &with_sl
        } else {
            t
        };
        let p = render::prepare(t, &r.gradient, &r.stars, true, &cancel)?;
        for c in &p.channels {
            match &c.report {
                Some(g) => println!(
                    "  {} gradient: degree {}, {} of {} boxes kept, {:.1}% of the background across the frame, {:.2}% left",
                    c.name,
                    g.degree,
                    g.kept,
                    g.boxes,
                    g.amplitude_frac * 100.0,
                    g.residual_frac * 100.0
                ),
                None => println!("  {} gradient: off", c.name),
            }
        }
        let coef = render::coefficients(&p, &r);
        let names_p = p.names();
        let gains: Vec<String> = names_p.iter().zip(&coef.gain).filter(|(_, g)| **g > 0.0).map(|(n, g)| format!("{n} ×{g:.2}")).collect();
        println!("  mix {}: {:?} balance, gains {}", if r.mix.preset.is_empty() { "custom" } else { &r.mix.preset }, r.mix.balance, gains.join(", "));
        for (i, want, used) in &coef.limited {
            println!("  {} needed ×{want:.1} to match the reference but was held at ×{used:.1}: its background would be over 3× as rough", names_p[*i]);
        }
        if render::stars_on(&p, &r) {
            let at = |n: &str| names_p.iter().position(|c| c == n);
            let g = |n: &str| at(n).map(|i| p.star_gain[i] / at("H").map(|h| p.star_gain[h]).unwrap_or(1.0));
            let how = if render::separate_on(&p, &r) {
                format!("stretched separately (size {:.0}%, brightness {:.0}%) and screened onto the starless image", r.stars.size * 100.0, r.stars.brightness * 100.0)
            } else {
                "recoloured from their own light".to_string()
            };
            let white = if p.kind == palette::Kind::Mono {
                "grey".to_string()
            } else if p.kind == palette::Kind::Broadband {
                let gg = |n: &str| at(n).zip(at("G")).map(|(i, j)| p.star_gain[i] / p.star_gain[j]).unwrap_or(1.0);
                format!("median star white: R ×{:.2}, B ×{:.2}", gg("R"), gg("B"))
            } else {
                format!(
                    "median star white: O ×{:.2}{}; red {:.0}% of green",
                    g("O").unwrap_or(1.0),
                    g("S").filter(|_| at("H").is_some()).map(|x| format!(", S ×{x:.2}")).unwrap_or_default(),
                    r.stars.ha_to_green * 100.0
                )
            };
            println!("  stars: {} found ({:.1}% of the frame), {how} ({white})", p.n_stars, p.star_cover * 100.0);
        }
        let rgb = render::render_full(&p, &r)?;
        if let Some(max) = size {
            let jpg = render::write_small(&rgb, &dir_out, &t.object, &r, max)?;
            println!("  wrote {} ({:.1} s)", jpg.display(), t0.elapsed().as_secs_f64());
            continue;
        }
        let mut extra = crux_core::header::Header::default();
        extra.push("NCHANNEL", &names.len().to_string(), "masters mixed");
        let w = render::write(&rgb, &dir_out, &t.object, &r, &extra, cfg.process.fits)?;
        println!("  wrote {} with {}.png and .jpg ({:.1} s)", w.xisf.display(), if w.fits.is_some() { ".fits, " } else { "" }, t0.elapsed().as_secs_f64());
        if save_recipe.is_some() {
            println!("  saved recipe {}", crux_core::render::recipe::save_named(&r)?.display());
        }
    }
    Ok(())
}

fn cache_dir(cli: &Cli, dir: &Path) -> PathBuf {
    let in_folder = cli.cache_dir.is_none() && Config::load(cli.config.as_deref(), Some(dir)).is_ok_and(|(c, _)| c.cache.in_folder);
    crux_core::cachedir::resolve(cli.cache_dir.as_deref(), dir, in_folder)
}

fn cache_cmd(cli: &Cli, what: &CacheCmd) -> Result<()> {
    use crux_core::cachedir;
    let bytes = report::bytes;
    // The cache folder for a data folder argument.
    let of = |d: &PathBuf| -> Result<PathBuf> {
        match &cli.cache_dir {
            Some(c) => Ok(c.clone()),
            None => cachedir::for_folder(d).filter(|c| c.is_dir()).ok_or_else(|| anyhow::anyhow!("no cache for {}", d.display())),
        }
    };
    match what {
        CacheCmd::List => {
            let all = cachedir::list();
            if all.is_empty() {
                println!("No caches in {}", cachedir::root().map(|r| r.display().to_string()).unwrap_or_default());
                return Ok(());
            }
            let (mut rb, mut kept, mut old) = (0, 0, 0);
            println!("{:>9} {:>9}  {:<10}  Folder", "Rebuild", "Keep", "Opened");
            for c in &all {
                let opened = c.last_used.map(|t| chrono::DateTime::<chrono::Local>::from(t).format("%Y-%m-%d").to_string()).unwrap_or_else(|| "?".into());
                let name = c.folder.as_ref().map(|f| f.display().to_string()).unwrap_or_else(|| c.dir.display().to_string());
                println!("{:>9} {:>9}  {:<10}  {}{}", bytes(c.rebuildable), bytes(c.kept), opened, name, if c.folder_exists { "" } else { "  (folder gone)" });
                if let Some((l, n)) = &c.legacy {
                    println!("{:>9} {:>9}  {:<10}    old cache still in the data folder: {}", bytes(*n), "", "", l.display());
                    old += n;
                }
                rb += c.rebuildable;
                kept += c.kept;
            }
            println!("\n{} cache{} in {}: {} can be rebuilt, {} of decisions and journals", all.len(), if all.len() == 1 { "" } else { "s" }, cachedir::root().unwrap().display(), bytes(rb), bytes(kept));
            if old > 0 {
                println!("Old caches inside data folders: {} (crux cache remove-old --all)", bytes(old));
            }
        }
        CacheCmd::Empty { dirs, all } => {
            let targets: Vec<PathBuf> = if *all { cachedir::list().into_iter().map(|c| c.dir).collect() } else { dirs.iter().map(of).collect::<Result<_>>()? };
            anyhow::ensure!(!targets.is_empty(), "name a folder, or pass --all");
            let mut freed = 0;
            for t in targets {
                freed += cachedir::empty(&t)?;
            }
            println!("Freed {}", bytes(freed));
        }
        CacheCmd::Forget { dirs, missing, yes } => {
            let mut targets: Vec<PathBuf> = dirs.iter().map(of).collect::<Result<_>>()?;
            if *missing {
                targets.extend(cachedir::list().into_iter().filter(|c| !c.folder_exists).map(|c| c.dir));
            }
            anyhow::ensure!(!targets.is_empty(), "nothing to forget");
            for t in &targets {
                println!("  {}", t.display());
            }
            confirm(*yes, &format!("Remove {} cache(s), your decisions and move journals included?", targets.len()))?;
            let mut freed = 0;
            for t in targets {
                freed += cachedir::forget(&t)?;
            }
            println!("Freed {}", bytes(freed));
        }
        CacheCmd::RemoveOld { dirs, all, yes } => {
            let mut targets: Vec<PathBuf> = dirs.iter().map(|d| d.join(cachedir::LEGACY)).filter(|l| l.join("cache.json").is_file()).collect();
            if *all {
                targets.extend(cachedir::list().into_iter().filter_map(|c| c.legacy.map(|l| l.0)));
            }
            anyhow::ensure!(!targets.is_empty(), "no old caches found");
            for t in &targets {
                // Only once it has been copied across.
                let parent = t.parent().unwrap_or(t);
                anyhow::ensure!(cachedir::describe_for(parent).is_some(), "{} hasn't been opened with this version yet, so its old cache wasn't copied", parent.display());
                println!("  {} ({})", t.display(), bytes(cachedir::dir_size(t)));
            }
            confirm(*yes, &format!("Remove {} old cache folder(s) from your data?", targets.len()))?;
            let mut freed = 0;
            for t in targets {
                freed += cachedir::remove_legacy(&t)?;
            }
            println!("Freed {}", bytes(freed));
        }
    }
    Ok(())
}

/// Ask before moving anything, unless --yes. Refuses without a terminal.
fn confirm(yes: bool, question: &str) -> Result<()> {
    use std::io::{IsTerminal, Write};
    if yes {
        return Ok(());
    }
    if !std::io::stdin().is_terminal() {
        anyhow::bail!("{question} Not confirmed: run in a terminal, or pass --yes");
    }
    print!("\n{question} Type yes to go ahead: ");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    if line.trim().eq_ignore_ascii_case("yes") {
        Ok(())
    } else {
        anyhow::bail!("not confirmed; nothing moved")
    }
}

fn triage(cli: &Cli, dir: &Path, analyse: bool) -> Result<(Config, Triage)> {
    if !dir.is_dir() {
        anyhow::bail!("{} is not a folder", dir.display());
    }
    let (mut cfg, _) = Config::load(cli.config.as_deref(), Some(dir))?;
    if let Some(l) = &cli.library {
        cfg.library.use_folder(l);
    }
    let mut cache = if cli.no_cache {
        Cache::disabled()
    } else {
        Cache::load(&cache_dir(cli, dir))
    };
    // Your decisions from the GUI: threshold changes and pinned grades.
    let decisions = if cli.no_cache { None } else { Some(crux_core::decisions::Decisions::load(&cache_dir(cli, dir))?) };
    if let Some(d) = &decisions {
        d.apply_thresholds(&mut cfg);
        d.apply_layout(&mut cfg);
        if !d.thresholds.is_empty() || !d.overrides.is_empty() {
            let th: Vec<String> = d.thresholds.iter().map(|(k, v)| format!("{k} {v}")).collect();
            eprintln!(
                "Using your decisions ({}): {} pinned frame{}{}",
                d.file.as_ref().map(|f| f.display().to_string()).unwrap_or_default(),
                d.overrides.len(),
                if d.overrides.len() == 1 { "" } else { "s" },
                if th.is_empty() { String::new() } else { format!(", limits {}", th.join(", ")) }
            );
        }
    }
    let mut t = pipeline::run(dir, &cfg, &mut cache, analyse, &report::progress);
    report::clear_progress();
    if let Some(d) = &decisions {
        d.apply_overrides(&t.scan.frames, &mut t.reports);
        d.apply_flat_choices(dir, &mut t.cal, &t.grouping, t.library.as_ref());
    }
    if let Err(e) = cache.save() {
        eprintln!("warning: couldn't save cache: {e:#}");
    }
    Ok((cfg, t))
}
