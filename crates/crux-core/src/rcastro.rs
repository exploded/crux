//! RC Astro's stand-alone CLI (`rc-astro`: BlurXTerminator,
//! NoiseXTerminator, StarXTerminator) as an add-on, run on the user's own
//! images with their own licence, through the host interface RC Astro
//! documents in `README-DEVS.md` beside the CLI: the products document
//! (`rc-astro --json`), the NDJSON event stream (`--json` with an input),
//! exit code 77 for "not licensed", and `--host` (schema 6 and later).
//!
//! **Kept apart from crux's own networks.** RC Astro's licence (clause 10)
//! forbids using its outputs to train, test, benchmark or improve tools
//! like crux's denoiser. Every image this module writes carries
//! `RCASTRO` and a HISTORY line, and `collect::touched_by_rc_astro` refuses
//! such files, so they can never end up in a training bundle.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// The schema this module was written against.
pub const SCHEMA: u32 = 6;
/// Exit code for "product not licensed on this computer".
pub const EXIT_UNLICENSED: i32 = 77;

/// The CLI found on this computer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cli {
    pub exe: PathBuf,
    pub version: String,
    pub schema: u32,
    pub products: Vec<Product>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Product {
    /// "bxt", "nxt", "sxt".
    pub key: String,
    pub name: String,
    pub licensed: bool,
    /// The licence's own status line.
    pub status: String,
}

impl Cli {
    pub fn licensed(&self, key: &str) -> bool {
        self.products.iter().any(|p| p.key == key && p.licensed)
    }
}

/// Where the CLI is: `CRUX_RC_ASTRO`, the install folder, or the PATH.
pub fn find() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("CRUX_RC_ASTRO").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(p);
    }
    let exe = if cfg!(windows) { "rc-astro.exe" } else { "rc-astro" };
    let mut dirs: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        for v in ["ProgramFiles", "ProgramW6432"] {
            if let Some(pf) = std::env::var_os(v) {
                dirs.push(PathBuf::from(pf).join("RC-Astro").join("CLI"));
            }
        }
    } else if cfg!(target_os = "macos") {
        dirs.push(PathBuf::from("/Applications/RC-Astro"));
        dirs.push(PathBuf::from("/usr/local/bin"));
        dirs.push(PathBuf::from("/opt/homebrew/bin"));
    }
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    dirs.into_iter().map(|d| d.join(exe)).find(|p| p.is_file())
}

fn command(exe: &Path) -> Command {
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut c = Command::new(exe);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // No console window when called from the GUI.
        c.creation_flags(0x0800_0000);
    }
    c
}

/// Ask the CLI what it is and what's licensed (the products document).
pub fn probe(exe: &Path) -> Result<Cli> {
    let out = command(exe).arg("--json").stdin(Stdio::null()).output().with_context(|| format!("run {}", exe.display()))?;
    let doc: serde_json::Value = serde_json::from_slice(&out.stdout).with_context(|| format!("{} --json didn't print its products document", exe.display()))?;
    let schema = doc["schemaVersion"].as_u64().unwrap_or(0) as u32;
    let version = doc["cliVersion"].as_str().unwrap_or("").to_string();
    let products = doc["products"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|p| Product {
                    key: p["key"].as_str().unwrap_or("").to_string(),
                    name: p["displayName"].as_str().or(p["name"].as_str()).unwrap_or("").to_string(),
                    licensed: p["license"]["valid"].as_bool().unwrap_or(false),
                    status: p["license"]["message"].as_str().or(p["license"]["status"].as_str()).unwrap_or("").to_string(),
                })
                .filter(|p| !p.key.is_empty())
                .collect()
        })
        .unwrap_or_default();
    Ok(Cli { exe: exe.to_path_buf(), version, schema, products })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Bxt {
    /// Correct only: fix aberrations, no sharpening (RC Astro's mode).
    pub correct_only: bool,
    pub sharpen_stars: f32,
    pub adjust_halos: f32,
    pub sharpen_nonstellar: f32,
    /// PSF diameter in pixels; `None` = automatic.
    pub psf_diameter: Option<f32>,
}

impl Default for Bxt {
    fn default() -> Self {
        Bxt { correct_only: false, sharpen_stars: 0.5, adjust_halos: 0.0, sharpen_nonstellar: 0.5, psf_diameter: None }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Nxt {
    pub denoise: f32,
    pub iterations: u32,
    /// Pixel scale of the low/high frequency transition.
    pub frequency_scale: f32,
}

impl Default for Nxt {
    fn default() -> Self {
        Nxt { denoise: 0.9, iterations: 2, frequency_scale: 5.0 }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sxt {
    /// Also write the stars-only image.
    pub stars: bool,
    pub unscreen: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "tool")]
pub enum Step {
    Bxt(Bxt),
    Nxt(Nxt),
    Sxt(Sxt),
}

impl Step {
    pub fn key(&self) -> &'static str {
        match self {
            Step::Bxt(_) => "bxt",
            Step::Nxt(_) => "nxt",
            Step::Sxt(_) => "sxt",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Step::Bxt(_) => "BlurXTerminator",
            Step::Nxt(_) => "NoiseXTerminator",
            Step::Sxt(_) => "StarXTerminator",
        }
    }

    /// Command-line options (only values that differ from the defaults, as
    /// the guide's section 8 asks).
    pub fn args(&self) -> Vec<String> {
        let f = |v: f32| format!("{v:.2}");
        let mut a = Vec::new();
        match self {
            Step::Bxt(b) => {
                if b.correct_only {
                    a.push("--correct-only".into());
                } else {
                    let d = Bxt::default();
                    if b.sharpen_stars != d.sharpen_stars {
                        a.extend(["--ss".into(), f(b.sharpen_stars.clamp(0.0, 0.7))]);
                    }
                    if b.adjust_halos != d.adjust_halos {
                        a.extend(["--ash".into(), f(b.adjust_halos.clamp(-0.5, 0.5))]);
                    }
                    if b.sharpen_nonstellar != d.sharpen_nonstellar {
                        a.extend(["--sn".into(), f(b.sharpen_nonstellar.clamp(0.0, 1.0))]);
                    }
                    if let Some(p) = b.psf_diameter {
                        a.extend(["--ansp=false".into(), "--nsd".into(), f(p.clamp(0.0, 8.0))]);
                    }
                }
            }
            Step::Nxt(n) => {
                let d = Nxt::default();
                if n.denoise != d.denoise {
                    a.extend(["--dn".into(), f(n.denoise.clamp(0.0, 1.0))]);
                }
                if n.iterations != d.iterations {
                    a.extend(["--it".into(), n.iterations.clamp(1, 5).to_string()]);
                }
                if n.frequency_scale != d.frequency_scale {
                    a.extend(["--fs".into(), format!("{:.1}", n.frequency_scale.clamp(1.0, 100.0))]);
                }
            }
            Step::Sxt(s) => {
                if s.stars {
                    a.push("--stars".into());
                } else if s.unscreen {
                    a.push("--unscreen".into());
                }
            }
        }
        a
    }
}

/// What a run reports as it goes.
#[derive(Debug, Clone)]
pub enum Event {
    /// Percent done, 0–100.
    Progress(f32),
    /// "initializing", "saving", "complete", or a model download.
    Status(String),
    /// "gpu" or "cpu", and the accelerator's name.
    Device(String, String),
    Warning(String),
}

/// Run one tool on `input`, writing `output` (both FITS or XISF; the
/// format follows the extension). Returns every file written (SXT with
/// `stars` writes two). Each output is rewritten by crux with the RC Astro
/// marker (see the module doc).
pub fn run(cli: &Cli, step: &Step, input: &Path, output: &Path, device: Option<&str>, progress: &(dyn Fn(Event) + Sync), cancel: &AtomicBool) -> Result<Vec<PathBuf>> {
    anyhow::ensure!(cli.licensed(step.key()), "{} isn't licensed on this computer: run `rc-astro license` to activate it", step.name());
    if let Some(d) = output.parent() {
        std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))?;
    }
    let mut c = command(&cli.exe);
    c.arg(step.key()).arg(input).arg("-o").arg(output).arg("--json").arg("--overwrite");
    if cli.schema >= 6 {
        c.arg("--host").arg(format!("crux-{}", env!("CARGO_PKG_VERSION")));
    }
    if let Some(d) = device {
        c.arg("--device").arg(d);
    }
    c.args(step.args());
    let mut child = c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().with_context(|| format!("run {}", cli.exe.display()))?;
    let stdout = child.stdout.take().context("no stdout")?;
    let mut stderr = child.stderr.take().context("no stderr")?;
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(|l| l.ok()) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let err_reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });
    let mut written: Vec<PathBuf> = Vec::new();
    let mut error: Option<String> = None;
    let status = loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("cancelled");
        }
        match rx.recv_timeout(Duration::from_millis(150)) {
            Ok(line) => handle(&line, progress, &mut written, &mut error),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break child.wait()?,
        }
    };
    let _ = reader.join();
    for line in rx.try_iter() {
        handle(&line, progress, &mut written, &mut error);
    }
    let err_text = err_reader.join().unwrap_or_default();
    if status.code() == Some(EXIT_UNLICENSED) {
        anyhow::bail!("{} isn't licensed on this computer: run `rc-astro license` to activate it", step.name());
    }
    if !status.success() {
        let msg = error.or_else(|| err_text.lines().find(|l| !l.trim().is_empty()).map(|l| l.trim().to_string())).unwrap_or_else(|| format!("exit code {:?}", status.code()));
        anyhow::bail!("{}: {msg}", step.name());
    }
    if written.is_empty() && output.is_file() {
        written.push(output.to_path_buf());
    }
    for p in &written {
        mark(p, step, &cli.version).with_context(|| format!("mark {}", p.display()))?;
    }
    Ok(written)
}

fn handle(line: &str, progress: &(dyn Fn(Event) + Sync), written: &mut Vec<PathBuf>, error: &mut Option<String>) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { return };
    match v["event"].as_str() {
        Some("progress") => progress(Event::Progress(v["done"].as_f64().unwrap_or(0.0) as f32)),
        Some("status") => {
            let phase = v["phase"].as_str().unwrap_or("").to_string();
            if phase == "complete" {
                if let Some(o) = v["output"].as_str() {
                    written.push(PathBuf::from(o));
                }
            }
            progress(Event::Status(phase));
        }
        Some("warning") => progress(Event::Warning(v["message"].as_str().unwrap_or("").to_string())),
        Some("error") => *error = v["message"].as_str().map(str::to_string),
        Some("info") => match v["topic"].as_str() {
            Some("device") => progress(Event::Device(v["device"].as_str().unwrap_or("").to_string(), v["name"].as_str().unwrap_or("").to_string())),
            Some("modelDownload") => progress(Event::Status(format!("downloading the model ({})", v["status"].as_str().unwrap_or("")))),
            _ => {}
        },
        _ => {}
    }
}

/// Rewrite an output with crux's marker: keyword `RCASTRO` (the tools, in
/// order) and a HISTORY line naming the tool, its settings and the CLI.
fn mark(path: &Path, step: &Step, cli_version: &str) -> Result<()> {
    let (img, mut hdr) = crate::process::read_with_header(path)?;
    let before = hdr.get("RCASTRO").map(str::to_string);
    hdr.cards.retain(|c| !c.key.eq_ignore_ascii_case("RCASTRO"));
    let tools = match before {
        Some(b) if !b.is_empty() => format!("{b},{}", step.key()),
        _ => step.key().to_string(),
    };
    hdr.push("RCASTRO", &tools, "processed by RC Astro tools (not for training)");
    hdr.push("HISTORY", "", &format!("RC Astro {} (rc-astro {cli_version}) {}", step.name(), step.args().join(" ")));
    let tmp = path.with_extension(format!("{}.tmp", path.extension().and_then(|e| e.to_str()).unwrap_or("img")));
    let fits = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("fits") || e.eq_ignore_ascii_case("fit") || e.eq_ignore_ascii_case("fts"));
    if fits {
        crate::fits::write_image(&tmp, &img, -32, &hdr)?;
    } else {
        crate::xisf::write_image(&tmp, &img, &hdr, None)?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("replace {}", path.display()))?;
    Ok(())
}

/// Run steps in order: `input` → … → `output`, intermediate files beside
/// `output` (removed afterwards). SXT's stars image, if asked for, is kept
/// as `<output stem>_stars.<ext>`. Returns the outputs.
pub fn run_chain(cli: &Cli, steps: &[Step], input: &Path, output: &Path, device: Option<&str>, progress: &(dyn Fn(usize, Event) + Sync), cancel: &AtomicBool) -> Result<Vec<PathBuf>> {
    anyhow::ensure!(!steps.is_empty(), "no RC Astro tools chosen");
    let ext = output.extension().and_then(|e| e.to_str()).unwrap_or("xisf").to_string();
    let stem = output.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "out".into());
    let mut cur = input.to_path_buf();
    let mut temps: Vec<PathBuf> = Vec::new();
    let mut extra: Vec<PathBuf> = Vec::new();
    for (k, step) in steps.iter().enumerate() {
        let last = k + 1 == steps.len();
        let out = if last { output.to_path_buf() } else { output.with_file_name(format!("{stem}.step{}.{ext}", k + 1)) };
        let files = run(cli, step, &cur, &out, device, &|e| progress(k, e), cancel)?;
        for f in files {
            if f != out {
                // SXT's stars image: give it a stable name.
                let keep = output.with_file_name(format!("{stem}_stars.{ext}"));
                let _ = std::fs::remove_file(&keep);
                std::fs::rename(&f, &keep).with_context(|| format!("rename {}", f.display()))?;
                extra.push(keep);
            }
        }
        if !last {
            temps.push(out.clone());
        }
        cur = out;
    }
    for t in temps {
        let _ = std::fs::remove_file(t);
    }
    let mut all = vec![output.to_path_buf()];
    all.extend(extra);
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_follow_the_guide() {
        assert!(Step::Bxt(Bxt::default()).args().is_empty());
        assert_eq!(Step::Bxt(Bxt { correct_only: true, ..Default::default() }).args(), vec!["--correct-only"]);
        assert_eq!(Step::Bxt(Bxt { sharpen_stars: 0.3, psf_diameter: Some(4.0), ..Default::default() }).args(), vec!["--ss", "0.30", "--ansp=false", "--nsd", "4.00"]);
        assert_eq!(Step::Nxt(Nxt { denoise: 0.6, ..Default::default() }).args(), vec!["--dn", "0.60"]);
        assert_eq!(Step::Sxt(Sxt { stars: true, unscreen: true }).args(), vec!["--stars"]);
    }

    #[test]
    fn events_are_parsed() {
        let seen = std::sync::Mutex::new(Vec::new());
        let (mut w, mut e) = (Vec::new(), None);
        let log = |ev: Event| seen.lock().unwrap().push(format!("{ev:?}"));
        handle(r#"{"event":"progress","done":12.5,"mpPerSec":3.4,"eta":7.8}"#, &log, &mut w, &mut e);
        handle(r#"{"event":"status","phase":"complete","message":"Done","output":"C:/x/out.xisf"}"#, &log, &mut w, &mut e);
        handle(r#"{"event":"error","message":"output file already exists"}"#, &log, &mut w, &mut e);
        handle(r#"{"event":"info","topic":"device","device":"gpu","name":"RTX","provider":"DirectML","runtime":""}"#, &log, &mut w, &mut e);
        handle("not json", &log, &mut w, &mut e);
        assert_eq!(w, vec![PathBuf::from("C:/x/out.xisf")]);
        assert_eq!(e.as_deref(), Some("output file already exists"));
        let s = seen.lock().unwrap();
        assert!(s[0].contains("12.5") && s.iter().any(|x| x.contains("RTX")));
    }
}
