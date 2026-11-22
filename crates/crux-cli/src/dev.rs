//! `crux dev …`: validation tools for processing (compare against
//! PixInsight's outputs). Hidden from the main help.

use anyhow::Result;
use clap::Subcommand;
use crux_core::process::{self, calibrate, integrate};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Subcommand)]
pub enum DevCmd {
    /// Difference statistics between two images.
    Compare { a: PathBuf, b: PathBuf },
    /// Calibrate one light with the given masters.
    Calibrate {
        light: PathBuf,
        #[arg(long)]
        dark: Option<PathBuf>,
        #[arg(long)]
        flat: Option<PathBuf>,
        #[arg(long)]
        bias: Option<PathBuf>,
        /// Hot-pixel sigma (0 = off).
        #[arg(long, default_value_t = 10.0)]
        hot: f32,
        /// No output pedestal.
        #[arg(long)]
        no_pedestal: bool,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Compare the result with this file (e.g. WBPP's _c.xisf).
        #[arg(long)]
        compare: Option<PathBuf>,
    },
    /// Stack raw frames into a master (flats: bias-subtracted, multiplicative).
    Stack {
        files: Vec<PathBuf>,
        #[arg(long)]
        bias: Option<PathBuf>,
        /// flat, dark or bias.
        #[arg(long, default_value = "flat")]
        kind: String,
        /// esd, winsor, sigma, none, auto.
        #[arg(long, default_value = "auto")]
        rejection: String,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        compare: Option<PathBuf>,
    },
    /// Integrate registered lights (milestone 12) and compare with another
    /// master (e.g. WBPP's masterLight from the same frames).
    Integrate {
        files: Vec<PathBuf>,
        /// Reference frame: the file whose name contains this (default: first).
        #[arg(long)]
        reference: Option<String>,
        /// local or global.
        #[arg(long, default_value = "local")]
        norm: String,
        /// snr, equal or exposure.
        #[arg(long, default_value = "snr")]
        weights: String,
        /// esd, winsor, sigma, percentile, none, auto.
        #[arg(long, default_value = "auto")]
        rejection: String,
        #[arg(long, default_value_t = 256)]
        cell: usize,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        compare: Option<PathBuf>,
    },
    /// Register an image (e.g. another program's master) onto a reference
    /// by its stars, warp it there (Lanczos-3) and compare the two.
    Align {
        input: PathBuf,
        #[arg(long)]
        to: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Dump a target's channel statistics, mixes and rendered proxy pixels
    /// as JSON, to check the palette engine against the PixInsight
    /// mixer's own JavaScript (run with node).
    MixerDump {
        /// Processing output folder (with crux-integration.json).
        out: PathBuf,
        #[arg(long)]
        object: Option<String>,
        #[arg(long)]
        json: PathBuf,
        /// Every n-th proxy pixel.
        #[arg(long, default_value_t = 7)]
        step: usize,
    },
    /// Render many recipes of one target as previews (the Image stage's
    /// proxy, at most 1,200 px), preparing once per gradient and star
    /// setting: for fitting stretch and colour settings. Also writes each
    /// prepared target's measurements (`stats.json`).
    RenderMany {
        /// The data folder (with `--masters`: masters stacked elsewhere).
        dir: PathBuf,
        #[arg(long)]
        masters: bool,
        #[arg(long)]
        workspace: PathBuf,
        /// A folder of recipe files (`*.toml`); each renders to
        /// `<out>/<file stem>.png`.
        #[arg(long)]
        recipes: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Inject a known gradient into a real master and measure how much of
    /// it the gradient fit recovers (the master's own background fitted
    /// first and taken off the comparison).
    GradientCheck {
        master: PathBuf,
        /// Gradient size, as a fraction of the background across the frame.
        #[arg(long, default_value_t = 0.2)]
        amount: f64,
        #[arg(long, default_value_t = 0)]
        degree: u32,
    },
    /// Run the trained denoiser (tools/ml) on an image: writes the result,
    /// prints the noise before and after, and optionally a PNG of the
    /// centre, before | after with one stretch.
    #[cfg(feature = "ml")]
    Denoise {
        image: PathBuf,
        /// A network's .onnx (with its .json beside it); default: the
        /// built-in denoiser.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Values for the network's control channels (e.g. sharpen 0 or 1).
        #[arg(long, num_args = 1..)]
        control: Vec<f32>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1.0)]
        strength: f32,
        #[arg(long, default_value_t = 512)]
        tile: usize,
        #[arg(long)]
        png: Option<PathBuf>,
        /// Centre of the PNG crop (default: the image centre).
        #[arg(long, num_args = 2)]
        at: Option<Vec<usize>>,
        /// Use these instead of measuring (normalised units; FWHM in px).
        #[arg(long)]
        noise: Option<f32>,
        #[arg(long)]
        sky: Option<f32>,
        #[arg(long)]
        fwhm: Option<f32>,
        /// PNG of input − output over the whole frame (4×4 block means,
        /// ±3σ of pure noise): tile seams or removed structure show here.
        #[arg(long)]
        residual: Option<PathBuf>,
    },
    /// Write images as Float32 FITS (first stored row first), e.g. masters
    /// for the training scripts, which read FITS only.
    ExportFits {
        files: Vec<PathBuf>,
        #[arg(long)]
        out_dir: PathBuf,
        /// File name prefix.
        #[arg(long, default_value = "")]
        prefix: String,
    },
    /// crux's classic starless (the Image stage's push-pull fill) of each
    /// image, as Float32 FITS: backgrounds for training the star remover.
    Starless {
        files: Vec<PathBuf>,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long, default_value = "")]
        prefix: String,
    },
    /// Plate-solve a master and fit its lens distortion, as the mosaic does.
    MosaicSolve { master: PathBuf },
    /// Work out which pixels WBPP's cosmetic correction replaced and how.
    HotProbe { light: PathBuf, dark: PathBuf, flat: PathBuf, reference: PathBuf },
    /// Warp an image with a PixInsight matrix (from an .xdrz file).
    Warp {
        input: PathBuf,
        #[arg(long)]
        xdrz: PathBuf,
        /// Treat the matrix as source → reference (default: reference → source).
        #[arg(long)]
        invert: bool,
        /// Lanczos clamping threshold (0 = off).
        #[arg(long, default_value_t = 0.15)]
        clamp: f32,
        #[arg(long)]
        compare: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Register a folder's good lights from cached stars; optionally compare
    /// with the matrices in a WBPP log.
    Register {
        dir: PathBuf,
        /// File-name fragment of the reference (default: crux's choice).
        #[arg(long)]
        reference: Option<String>,
        #[arg(long)]
        wbpp_log: Option<PathBuf>,
        /// Folder tree with WBPP .xdrz files (full-precision matrices).
        #[arg(long)]
        xdrz_dir: Option<PathBuf>,
        /// Only lights whose name contains this.
        #[arg(long)]
        only: Option<String>,
    },
    /// Detect stars in registered frames and measure their offset from the reference.
    StarCheck { reference: PathBuf, files: Vec<PathBuf> },
    /// Flat scaling factor (mean) of a master flat.
    FlatMean { file: PathBuf },
    /// One-shot-colour frames: Bayer pattern check (do the two greens agree
    /// as read, and for each 1 px shift?) and an optional debayered preview.
    Cfa {
        files: Vec<PathBuf>,
        /// Write a debayered, auto-stretched colour PNG of the first file.
        #[arg(long)]
        png: Option<PathBuf>,
    },
}

/// Two masters of the same field: linear relation, residuals, large-scale
/// difference, star photometry and SNR.
fn compare_masters(a: &std::path::Path, b: &std::path::Path, stars: &[(f32, f32)], fwhm: f32, b_noise: Option<f64>) -> Result<()> {
    use crux_core::process::combine::noise_of;
    let ia = crux_core::read_image(a)?;
    let ib = crux_core::read_image(b)?;
    anyhow::ensure!(ia.width == ib.width && ia.height == ib.height, "sizes differ");
    let (w, h) = (ia.width, ia.height);
    let r = (2.0 * fwhm).clamp(3.0, 15.0);
    let flux = |img: &crux_core::Image, x: f32, y: f32| -> Option<f64> {
        let (cx, cy) = (x.round() as i64, y.round() as i64);
        let ro = (r + 6.0).ceil() as i64;
        if cx < ro || cy < ro || cx + ro >= w as i64 || cy + ro >= h as i64 {
            return None;
        }
        let mut ann = Vec::new();
        let mut inside = Vec::new();
        for yy in cy - ro..=cy + ro {
            for xx in cx - ro..=cx + ro {
                let d = ((xx as f32 - x).powi(2) + (yy as f32 - y).powi(2)).sqrt();
                let v = img.data[yy as usize * w + xx as usize];
                if d <= r {
                    inside.push(v as f64);
                } else if d >= r + 2.0 && d <= r + 6.0 {
                    ann.push(v);
                }
            }
        }
        let bg = crux_core::stats::median_mut(&mut ann) as f64;
        let s: f64 = inside.iter().map(|v| v - bg).sum();
        (s > 0.0).then_some(s)
    };
    let mut ratios = Vec::new();
    for &(x, y) in stars.iter().take(250) {
        if let (Some(p), Some(q)) = (flux(&ia, x, y), flux(&ib, x, y)) {
            ratios.push(q / p);
        }
    }
    let k = crux_core::stats::median_f64(&ratios).unwrap_or(1.0);
    let dev: Vec<f64> = ratios.iter().map(|r| (r / k - 1.0).abs()).collect();
    let spread = crux_core::stats::median_f64(&dev).unwrap_or(0.0) * 1.4826;
    let med = |img: &crux_core::Image| crux_core::process::integrate::sample_median(&img.data, 7) as f64;
    let off = med(&ib) - k * med(&ia);
    // b may have been resampled here (dev align), which lowers its per-pixel
    // noise: use its noise from before when given.
    let (na, nb) = (noise_of(&ia) as f64, b_noise.unwrap_or_else(|| noise_of(&ib) as f64));
    println!("Against {}:", b.display());
    println!("  stars: b = {k:.4} x a ({} stars, scatter {:.2}%); offset {:+.2} ADU", ratios.len(), spread * 100.0, off * 65535.0);
    println!("  noise: a {:.2} ADU, b {:.2} ADU; in a's units b is {:.2} ADU; SNR a/b = {:.3}", na * 65535.0, nb * 65535.0, nb / k * 65535.0, (nb / k) / na);
    let cell = 128usize;
    let (nx, ny) = (w / cell, h / cell);
    let mut cells = Vec::new();
    let mut sig = Vec::new();
    for cy in 0..ny {
        for cx in 0..nx {
            let mut v = Vec::new();
            for y in (cy * cell..(cy + 1) * cell).step_by(2) {
                for x in (cx * cell..(cx + 1) * cell).step_by(2) {
                    let (p, q) = (ia.data[y * w + x] as f64, ib.data[y * w + x] as f64);
                    if p > 0.0 && q > 0.0 {
                        v.push((q - (k * p + off)) as f32);
                    }
                }
            }
            if v.len() > cell * cell / 8 {
                let (m, s) = crux_core::stats::clipped_median_sigma(&mut v, 3.0, 5);
                cells.push(m as f64 * 65535.0);
                sig.push(s as f64 * 65535.0);
            }
        }
    }
    let p = |v: &[f64], q: f64| crux_core::stats::percentile_f64(v, q).unwrap_or(0.0);
    println!(
        "  residual b - (k a + offset): per-pixel sigma {:.2} ADU (median over 128 px cells); large-scale cell medians 5-95% {:+.2} .. {:+.2} ADU (p-p {:.2})",
        crux_core::stats::median_f64(&sig).unwrap_or(0.0),
        p(&cells, 0.05),
        p(&cells, 0.95),
        p(&cells, 0.95) - p(&cells, 0.05)
    );
    Ok(())
}

fn print_diff(d: &process::Diff, scale: f64) {
    println!(
        "  mean {:.6} vs {:.6}; a−b median {:+.3e} σ {:.3e} rms {:.3e} max {:.3e} ({:.2} ADU σ, {:.1} ADU max); {} px over 1e-4 ({:.4}%); b = {:.6}·a {:+.3e}",
        d.mean_a,
        d.mean_b,
        d.median,
        d.sigma,
        d.rms,
        d.max_abs,
        d.sigma * scale,
        d.max_abs * scale,
        d.over,
        d.over as f64 * 100.0 / d.n as f64,
        d.gain,
        d.offset
    );
}

pub fn run(cmd: &DevCmd, triage: &dyn Fn(&std::path::Path) -> Result<crux_core::pipeline::Triage>) -> Result<()> {
    match cmd {
        DevCmd::Compare { a, b } => {
            let (x, y) = (crux_core::read_image(a)?, crux_core::read_image(b)?);
            print_diff(&process::diff(&x, &y)?, 65535.0);
        }
        DevCmd::HotProbe { light, dark, flat, reference } => {
            let (raw, hdr) = process::read_with_header(light)?;
            let d = process::load_master(dark, "dark")?;
            let c = calibrate::Calibrator::new(None, Some(d.clone()), Some(process::load_master(flat, "flat")?), hdr.f64("EXPTIME"), calibrate::CalParams { hot_sigma: None, pedestal_limit: None, ..Default::default() })?;
            let (img, _) = c.apply(&raw)?;
            let r = crux_core::read_image(reference)?;
            let w = img.width;
            let dd = &d.image.data;
            let mut changed: Vec<usize> = (0..img.data.len()).filter(|&i| (img.data[i] - r.data[i]).abs() > 2e-6 && img.data[i] > 0.0 && img.data[i] < 1.0).collect();
            println!("{} pixels changed", changed.len());
            let mut dv: Vec<f32> = changed.iter().map(|&i| dd[i]).collect();
            dv.sort_by(|a, b| a.total_cmp(b));
            println!("dark values of changed pixels: min {:.7} p1 {:.7} median {:.7}", dv[0], dv[dv.len() / 100], dv[dv.len() / 2]);
            let mut all: Vec<f32> = dd.iter().copied().collect();
            let (m, s) = crux_core::stats::median_sigma_mut(&mut all.clone());
            all.sort_by(|a, b| a.total_cmp(b));
            let n = all.len();
            let mean = all.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
            let sd = (all.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
            println!("dark median {m:.7} MADσ {s:.7} mean {mean:.7} sd {sd:.7}; ADU: median {:.2} MADσ {:.3} sd {:.3}", m * 65535.0, s * 65535.0, sd * 65535.0);
            let t = dv[0];
            let above = all.iter().filter(|x| **x >= t).count();
            println!("pixels with dark >= min changed: {above}; implied k = (t−median)/MADσ {:.2}, /sd {:.2}", (t - m) / s, (t as f64 - mean) / sd);
            // Replacement: compare with neighbour statistics.
            changed.retain(|&i| i % w > 1 && i % w < w - 2 && i / w > 1 && i / w < img.height - 2);
            let hot: std::collections::HashSet<usize> = (0..dd.len()).filter(|&i| dd[i] >= t).collect();
            let k = changed.len().min(20000);
            let mut errs = [0f64; 6];
            for &i in changed.iter().take(k) {
                let mut nb: Vec<f32> = Vec::new();
                let mut nbc: Vec<f32> = Vec::new();
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let j = (i as i64 + dy * w as i64 + dx) as usize;
                        if dx == 0 && dy == 0 { nbc.push(img.data[j]); continue; }
                        nb.push(img.data[j]);
                        nbc.push(img.data[j]);
                    }
                }
                let mut s8 = nb.clone(); s8.sort_by(|a, b| a.total_cmp(b));
                let mut s9 = nbc.clone(); s9.sort_by(|a, b| a.total_cmp(b));
                let clean: Vec<f32> = { let mut v = Vec::new(); for dy in -1i64..=1 { for dx in -1i64..=1 { let j = (i as i64 + dy * w as i64 + dx) as usize; if (dx, dy) != (0, 0) && !hot.contains(&j) { v.push(img.data[j]); } } } v };
                let mut sc = clean.clone(); sc.sort_by(|a, b| a.total_cmp(b));
                let cands = [
                    0.5 * (s8[3] + s8[4]),
                    s9[4],
                    nb.iter().sum::<f32>() / 8.0,
                    if sc.is_empty() { 0.0 } else { sc[sc.len() / 2] },
                    if sc.is_empty() { 0.0 } else if sc.len() % 2 == 1 { sc[sc.len() / 2] } else { 0.5 * (sc[sc.len() / 2 - 1] + sc[sc.len() / 2]) },
                    if clean.is_empty() { 0.0 } else { clean.iter().sum::<f32>() / clean.len() as f32 },
                ];
                for (e, c) in errs.iter_mut().zip(cands) { *e += (r.data[i] - c).abs() as f64; }
            }
            let names = ["median8 avg", "median9", "mean8", "clean upper median", "clean median avg", "clean mean"];
            for (n, e) in names.iter().zip(errs) { println!("  {n}: {:.3e}", e / k as f64); }
            // Sigma estimators.
            let mut v: Vec<f64> = dd.iter().map(|x| *x as f64).collect();
            for kk in [5.0, 3.0] {
                let mut lo = m as f64 - kk * s as f64; let mut hi = m as f64 + kk * s as f64;
                let mut sdv = 0.0; let mut mu = 0.0;
                for _ in 0..10 {
                    let sel: Vec<f64> = v.iter().copied().filter(|x| *x >= lo && *x <= hi).collect();
                    mu = sel.iter().sum::<f64>() / sel.len() as f64;
                    sdv = (sel.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / sel.len() as f64).sqrt();
                    lo = mu - kk * sdv; hi = mu + kk * sdv;
                }
                let tt = m as f64 + 10.0 * sdv;
                println!("  clipped {kk}σ: mean {:.3} sd {:.4} ADU -> count above median+10sd {}", mu * 65535.0, sdv * 65535.0, v.iter().filter(|x| **x > tt).count());
            }
            v.clear();
        }
        DevCmd::Warp { input, xdrz, invert, clamp, compare, out } => {
            let (m, origin) = read_xdrz(xdrz)?;
            let src = crux_core::read_image(input)?;
            let h = process::resample::Homography::from_pixinsight(m, origin);
            let h = if *invert { h.inverse().unwrap() } else { h };

            let t = Instant::now();
            let img = process::resample::warp(&src, &h, src.width, src.height, if *clamp > 0.0 { Some(*clamp) } else { None });
            println!("warped in {:.2} s", t.elapsed().as_secs_f64());
            if let Some(o) = out {
                crux_core::xisf::write_image(o, &img, &Default::default(), None)?;
            }
            if let (Some(r), true) = (compare, std::env::var("CRUX_PROBE").is_ok()) {
                let rf = crux_core::read_image(r)?;
                let none = process::resample::warp(&src, &h, src.width, src.height, None);
                let bil = process::resample::warp_bilinear(&src, &h, src.width, src.height);
                let w = src.width;
                let mut n = 0usize;
                let (mut e0, mut e1, mut eb) = (0f64, 0f64, 0f64);
                let mut samples = 0;
                for y in 50..src.height - 50 {
                    for x in 50..w - 50 {
                        let i = y * w + x;
                        if (none.data[i] - rf.data[i]).abs() > 1e-4 {
                            n += 1;
                            e0 += (none.data[i] - rf.data[i]).abs() as f64;
                            e1 += (img.data[i] - rf.data[i]).abs() as f64;
                            eb += (bil.data[i] - rf.data[i]).abs() as f64;
                            if samples < 12 && n % 997 == 0 {
                                samples += 1;
                                let (sx, sy) = h.apply(x as f64, y as f64);
                                let (ix, iy) = (sx.round() as i64, sy.round() as i64);
                                let nb: Vec<String> = (-1i64..=1).map(|dy| (-1i64..=1).map(|dx| format!("{:.5}", src.data[((iy + dy).clamp(0, src.height as i64 - 1) * w as i64 + (ix + dx).clamp(0, w as i64 - 1)) as usize])).collect::<Vec<_>>().join(" ")).collect();
                                println!("  ({x},{y}) wbpp {:.5} lanczos {:.5} clamp {:.5} bilinear {:.5}  src 3x3: {}", rf.data[i], none.data[i], img.data[i], bil.data[i], nb.join(" | "));
                            }
                        }
                    }
                }
                println!("  {n} px differ from unclamped Lanczos; mean |err| unclamped {:.3e}, clamped {:.3e}, bilinear {:.3e}", e0 / n as f64, e1 / n as f64, eb / n as f64);
            }
            if let Some(r) = compare {
                let rf = crux_core::read_image(r)?;
                print_diff(&process::diff(&img, &rf)?, 65535.0);
                // Interior only (skip 50 px borders and no-data pixels).
                let (w, hh) = (img.width, img.height);
                let mut d: Vec<f32> = Vec::new();
                let mut big = 0usize;
                for y in 50..hh - 50 {
                    for x in 50..w - 50 {
                        let (a, b) = (img.data[y * w + x], rf.data[y * w + x]);
                        if a > 0.0 && b > 0.0 {
                            d.push(a - b);
                            if (a - b).abs() > 1e-4 { big += 1; }
                        }
                    }
                }
                let n = d.len();
                let rms = (d.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
                let (md, sd) = crux_core::stats::median_sigma_mut(&mut d);
                println!("  interior: median {md:+.3e} σ {sd:.3e} rms {rms:.3e} ({:.3} ADU rms); {big} px over 1e-4 ({:.4}%)", rms * 65535.0, big as f64 * 100.0 / n as f64);
            }
        }
        DevCmd::Register { dir, reference, wbpp_log, xdrz_dir, only } => {
            use crux_core::grade::Grade;
            use crux_core::process::register;
            let t = triage(dir)?;
            let lights: Vec<usize> = t
                .reports
                .iter()
                .filter(|r| r.grade != Grade::Reject)
                .map(|r| r.frame)
                .filter(|i| t.metrics.contains_key(i))
                .filter(|&i| only.as_ref().is_none_or(|o| t.scan.frames[i].meta.path.to_string_lossy().contains(o.as_str())))
                .collect();
            let score = |i: usize| {
                let m = &t.metrics[&i];
                let tr = t.report_for(i).and_then(|r| r.transparency.as_ref()).map(|x| x.value);
                register::reference_score(m.n_stars, m.fwhm, m.ecc, tr)
            };
            let name = |i: usize| t.scan.frames[i].meta.file_name().to_string();
            let r0 = match reference {
                Some(f) => *lights.iter().find(|&&i| name(i).contains(f.as_str())).ok_or_else(|| anyhow::anyhow!("no light matches {f}"))?,
                None => *lights.iter().max_by(|&&a, &&b| score(a).total_cmp(&score(b))).ok_or_else(|| anyhow::anyhow!("no lights"))?,
            };
            let mut best: Vec<usize> = lights.clone();
            best.sort_by(|&a, &b| score(b).total_cmp(&score(a)));
            println!("reference {} (score {:.0}); next: {}", name(r0), score(r0), best.iter().take(4).map(|&i| format!("{} {:.0}", name(i), score(i))).collect::<Vec<_>>().join(", "));
            let mut wb = match wbpp_log { Some(p) => parse_wbpp_matrices(p)?, None => Vec::new() };
            if let Some(d) = xdrz_dir {
                for e in walkdir::WalkDir::new(d).into_iter().flatten() {
                    if e.path().extension().is_some_and(|x| x == "xdrz") {
                        let (m, _) = read_xdrz(e.path())?;
                        wb.push((e.path().to_string_lossy().to_string(), m));
                    }
                }
            }
            let rm = &t.metrics[&r0];
            let (w, h) = (rm.width as f64, rm.height as f64);
            let t0 = Instant::now();
            let mut diffs = Vec::new();
            for &i in &lights {
                let m = &t.metrics[&i];
                let reg = register::register(&rm.stars, &m.stars, w, h);
                let n = name(i);
                let stem = n.trim_end_matches(".fits").trim_end_matches(".xisf").to_string();
                match reg {
                    Some(r) => {
                        let cmp = wb.iter().find(|(k, _)| k.contains(&stem)).map(|(_, mat)| {
                            // WBPP: reference → target, origin 0.5.
                            let to_src = process::resample::Homography::from_pixinsight(*mat, 0.5);
                            let to_ref = to_src.inverse().unwrap();
                            let d = register::max_difference(&r.to_ref, &to_ref, w, h);
                            diffs.push(d);
                            // Both models on crux's star pairs.
                            let mut e = (0f64, 0f64, 0usize);
                            let mut ec = [(0f64, 0usize); 2];
                            for s in &m.stars {
                                let (x, y) = (s.x as f64, s.y as f64);
                                let q = r.to_ref.apply(x, y);
                                if let Some(rs) = rm.stars.iter().find(|t| ((t.x as f64 - q.0).powi(2) + (t.y as f64 - q.1).powi(2)) < 2.25) {
                                    let (rx, ry) = (rs.x as f64, rs.y as f64);
                                    let q2 = to_ref.apply(x, y);
                                    let a = (q.0 - rx).powi(2) + (q.1 - ry).powi(2);
                                    let b = (q2.0 - rx).powi(2) + (q2.1 - ry).powi(2);
                                    e.0 += a; e.1 += b; e.2 += 1;
                                    let edge = ((x - w / 2.0).abs() > w * 0.35 || (y - h / 2.0).abs() > h * 0.35) as usize;
                                    ec[edge].0 += b; ec[edge].1 += 1;
                                }
                            }
                            let n = e.2.max(1) as f64;
                            format!("  vs WBPP max {d:.3} px; on {} pairs rms crux {:.3} WBPP {:.3} (WBPP centre {:.3}, edges {:.3})", e.2, (e.0 / n).sqrt(), (e.1 / n).sqrt(), (ec[0].0 / ec[0].1.max(1) as f64).sqrt(), (ec[1].0 / ec[1].1.max(1) as f64).sqrt())
                        }).unwrap_or_default();
                        println!("  {n}: {} stars, rms {:.3} px, shift ({:+.1}, {:+.1}) rot {:+.3}° scale {:.5}{}{cmp}", r.matched, r.rms, r.shift.0, r.shift.1, r.rotation, r.scale, if r.flipped { " flipped" } else { "" });
                    }
                    None => println!("  {n}: no match"),
                }
            }
            println!("{} frames in {:.2} s", lights.len(), t0.elapsed().as_secs_f64());
            if !diffs.is_empty() {
                diffs.sort_by(|a, b| a.total_cmp(b));
                println!("vs WBPP: {} frames, median max-difference {:.3} px, worst {:.3} px", diffs.len(), diffs[diffs.len() / 2], diffs[diffs.len() - 1]);
            }
        }
        DevCmd::StarCheck { reference, files } => {
            let cfg = crux_core::Config::default();
            let r = crux_core::analysis::analyze_light(&crux_core::read_image(reference)?, &cfg);
            println!("reference: {} stars, FWHM {:.2}", r.stars.len(), r.fwhm.unwrap_or(0.0));
            for f in files {
                let m = crux_core::analysis::analyze_light(&crux_core::read_image(f)?, &cfg);
                match crux_core::process::register::register(&r.stars, &m.stars, r.width as f64, r.height as f64) {
                    Some(x) => {
                        let id = process::resample::Homography::identity();
                        println!(
                            "  {}: {} stars matched, residual {:.3} px; offset from the reference: shift ({:+.3}, {:+.3}) px, rotation {:+.4}°, scale {:.5}, max over the frame {:.3} px; FWHM {:.2}",
                            f.file_name().unwrap().to_string_lossy(), x.matched, x.rms, x.shift.0, x.shift.1, x.rotation, x.scale,
                            crux_core::process::register::max_difference(&x.to_ref, &id, r.width as f64, r.height as f64), m.fwhm.unwrap_or(0.0)
                        );
                    }
                    None => println!("  {}: no match", f.display()),
                }
            }
        }
        DevCmd::Cfa { files, png } => {
            for f in files {
                let (img, hdr) = process::read_with_header(f)?;
                let Some(b) = crux_core::cfa::from_header(&hdr) else {
                    println!("{}: no Bayer pattern in the header", f.display());
                    continue;
                };
                let shifts: Vec<String> = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .iter()
                    .map(|&(dx, dy)| {
                        let s = crux_core::cfa::Bayer::parse(&b.name(), dx, dy).unwrap();
                        format!("{} {:.4}", s.name(), crux_core::cfa::green_mismatch(&img, s))
                    })
                    .collect();
                let hot = crux_core::cfa::hot_pixels(&img, 8.0).len();
                println!("{}: {} as read; green mismatch {}; {hot} hot pixels", f.file_name().unwrap_or_default().to_string_lossy(), b.name(), shifts.join(", "));
            }
            if let (Some(p), Some(f)) = (png, files.first()) {
                let (img, hdr) = process::read_with_header(f)?;
                let b = crux_core::cfa::from_header(&hdr).ok_or_else(|| anyhow::anyhow!("no Bayer pattern"))?;
                let mut clean = img.clone();
                for (x, y) in crux_core::cfa::hot_pixels(&img, 8.0) {
                    crux_core::cfa::replace_with_neighbours(&mut clean, x, y);
                }
                let planes = crux_core::cfa::debayer(&clean, b);
                // One stretch per channel (crude colour balance), binned to ≤ 1,600 px.
                let k = (img.width.max(img.height)).div_ceil(1600).max(1);
                let (w, h) = (img.width / k, img.height / k);
                let bin = |im: &crux_core::Image| {
                    let mut d = vec![0f32; w * h];
                    for y in 0..h {
                        for x in 0..w {
                            let mut s = 0.0;
                            for yy in 0..k {
                                for xx in 0..k {
                                    s += im.get(x * k + xx, y * k + yy);
                                }
                            }
                            d[y * w + x] = s / (k * k) as f32;
                        }
                    }
                    crux_core::Image::from_data(w, h, d)
                };
                let small: Vec<crux_core::Image> = planes.iter().map(bin).collect();
                let stfs: Vec<crux_core::stretch::Stf> = small.iter().map(crux_core::stretch::Stf::auto).collect();
                let mut px = vec![0u8; w * h * 3];
                for i in 0..w * h {
                    for c in 0..3 {
                        px[i * 3 + c] = (stfs[c].apply(small[c].data[i]) * 255.0).round().clamp(0.0, 255.0) as u8;
                    }
                }
                let out = std::io::BufWriter::new(std::fs::File::create(p)?);
                let mut e = png::Encoder::new(out, w as u32, h as u32);
                e.set_color(png::ColorType::Rgb);
                e.set_depth(png::BitDepth::Eight);
                e.write_header()?.write_image_data(&px)?;
                println!("wrote {} ({w} x {h}, binned {k}x)", p.display());
            }
        }
        DevCmd::FlatMean { file } => {
            let f = crux_core::read_image(file)?;
            println!("mean {:.6}", calibrate::flat_mean(&f));
            let mut v: Vec<f32> = f.data.iter().copied().filter(|x| x.is_finite()).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            let n = v.len();
            for c in [0.01, 0.02, 0.05, 0.1] {
                let lo = (n as f64 * c) as usize;
                let m1: f64 = v[lo..].iter().map(|x| *x as f64).sum::<f64>() / (n - lo) as f64;
                let m2: f64 = v[lo..n - lo].iter().map(|x| *x as f64).sum::<f64>() / (n - 2 * lo) as f64;
                println!("  trim low {c}: {m1:.6}; both: {m2:.6}");
            }
            let mx = v[n - 1] as f64;
            for c in [0.05] {
                let kept: Vec<f64> = v.iter().map(|x| *x as f64).filter(|x| *x >= c * mx).collect();
                println!("  >= {c}·max: {:.6}", kept.iter().sum::<f64>() / kept.len() as f64);
            }
            println!("  median {:.6}", v[n / 2]);
        }
        DevCmd::Calibrate { light, dark, flat, bias, hot, no_pedestal, out, compare } => {
            let t0 = Instant::now();
            let ld = |p: &Option<PathBuf>, l: &str| -> Result<Option<calibrate::Master>> { p.as_ref().map(|p| process::load_master(p, l)).transpose() };
            let (raw, hdr) = process::read_with_header(light)?;
            let params = calibrate::CalParams { hot_sigma: if *hot > 0.0 { Some(*hot) } else { None }, pedestal_limit: if *no_pedestal { None } else { Some(0.0001) }, ..Default::default() };
            let c = calibrate::Calibrator::new(ld(bias, "bias")?, ld(dark, "dark")?, ld(flat, "flat")?, hdr.f64("EXPTIME"), params)?;
            println!("loaded in {:.2} s; flat mean {:.6}", t0.elapsed().as_secs_f64(), c.flat_mean);
            for n in &c.notes {
                println!("  {n}");
            }
            let t1 = Instant::now();
            let (img, st) = c.apply(&raw)?;
            println!("calibrated in {:.3} s: {:?}", t1.elapsed().as_secs_f64(), st);
            if let Some(o) = out {
                crux_core::xisf::write_image(o, &img, &hdr, None)?;
                println!("wrote {}", o.display());
            }
            if let Some(r) = compare {
                let rf = crux_core::read_image(r)?;
                print_diff(&process::diff(&img, &rf)?, 65535.0);
            }
        }
        DevCmd::Integrate { files, reference, norm, weights, rejection, cell, out, compare } => {
            use crux_core::process::combine;
            let ri = reference.as_ref().and_then(|r| files.iter().position(|f| f.to_string_lossy().contains(r.as_str()))).unwrap_or(0);
            let t0 = Instant::now();
            let rimg = crux_core::read_image(&files[ri])?;
            let cfg = crux_core::config::Config::default();
            let m = crux_core::analysis::analyze_light(&rimg, &cfg);
            let stars: Vec<(f32, f32)> = m.stars.iter().filter(|s| s.flags == 0).map(|s| (s.x, s.y)).collect();
            println!("reference {}: {} stars, FWHM {:.2} ({:.1} s)", files[ri].display(), stars.len(), m.fwhm.unwrap_or(0.0), t0.elapsed().as_secs_f64());
            let inputs: Vec<combine::Input> = files
                .iter()
                .map(|p| combine::Input {
                    path: p.clone(),
                    name: p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
                    exptime: crux_core::meta::read_meta(p).ok().and_then(|m| m.exptime).unwrap_or(1.0),
                    fwhm: m.fwhm,
                    night: None,
                    grade: crux_core::grade::Grade::Good,
                })
                .collect();
            let st = combine::Settings {
                normalisation: if norm == "global" { combine::Normalisation::Global } else { combine::Normalisation::Local },
                weighting: match weights.as_str() {
                    "equal" => combine::Weighting::Equal,
                    "exposure" => combine::Weighting::Exposure,
                    _ => combine::Weighting::Snr,
                },
                rejection: match rejection.as_str() {
                    "esd" => Some(integrate::Rejection::Esd { fraction: 0.3, alpha: 0.05 }),
                    "winsor" => Some(integrate::Rejection::Winsorized { low: 4.0, high: 3.0 }),
                    "sigma" => Some(integrate::Rejection::Sigma { low: 4.0, high: 3.0 }),
                    "percentile" => Some(integrate::Rejection::Percentile { low: 0.2, high: 0.1 }),
                    "none" => Some(integrate::Rejection::None),
                    _ => None,
                },
                local_cell: *cell,
            };
            let r = combine::integrate(&inputs, ri, &stars, &st, "dev", "", out, &Default::default(), &|_| {}, &std::sync::atomic::AtomicBool::new(false))?;
            println!("integrated {} frames with {} in {:.1} s; rejected {:.3}%, clamped {} px", r.frames.len(), r.rejection, r.seconds, r.rejected_total * 100.0, r.clamped);
            println!("  noise: reference {:.2} ADU, median frame {:.2}, master {:.2}; gain x{:.2} (weights predict x{:.2}), worth {:.1} frames", r.noise_reference, r.noise_typical, r.noise_master, r.snr_gain, r.snr_gain_expected, r.equivalent_frames);
            for f in &r.frames {
                println!("  {:<48} signal {:.3} ({:>3} stars)  noise {:6.2}  snr {:.3}  weight {:.3}  rejected {:.3}% low {:.3}% high", f.name, f.signal, f.stars, f.noise, f.snr, f.weight, f.rejected_low * 100.0, f.rejected_high * 100.0);
            }
            for w in &r.what_ifs {
                println!("  what if {}: SNR {:+.1}%{}", w.label, w.snr_change * 100.0, w.fwhm.map(|f| format!(", FWHM {f:.2} px")).unwrap_or_default());
            }
            if let Some(c) = compare {
                compare_masters(out, c, &stars, m.fwhm.unwrap_or(3.0), None)?;
            }
        }
        DevCmd::RenderMany { dir, masters, workspace, recipes, out } => render_many(dir, *masters, workspace, recipes, out)?,
        DevCmd::MixerDump { out, object, json, step } => {
            use crux_core::render::{self, palette, recipe::Recipe};
            let t = render::targets(out).into_iter().find(|t| object.as_ref().is_none_or(|o| &t.object == o)).ok_or_else(|| anyhow::anyhow!("no target"))?;
            let keep = render::stars::StarSettings { mode: render::stars::StarMode::Keep, ..Default::default() };
            let p = render::prepare(&t, &render::gradient::GradientSettings::default(), &keep, false, &std::sync::atomic::AtomicBool::new(false))?;
            let names = p.names();
            let stats: serde_json::Map<String, serde_json::Value> = p
                .channels
                .iter()
                .map(|c| {
                    let s = &c.stats;
                    (c.name.clone(), serde_json::json!({"median": s.median, "mad": s.mad, "noise": s.noise, "pMedian": s.p_median, "pHigh": s.p_high, "sorted": s.sorted}))
                })
                .collect();
            let idx: Vec<usize> = (0..p.channels[0].proxy.data.len()).step_by(*step).collect();
            let px: serde_json::Map<String, serde_json::Value> = p.channels.iter().map(|c| (c.name.clone(), serde_json::json!(idx.iter().map(|&i| c.proxy.data[i]).collect::<Vec<f32>>()))).collect();
            let mut cases = Vec::new();
            let opts = [
                palette::Stretching::default(),
                palette::Stretching { neutral_sky: true, black: 0.7, background: 0.10, keep_colour: 0.3, ..Default::default() },
                palette::Stretching { dark_ref: 0.3, keep_colour: 1.0, clip: 5.0, ..Default::default() },
            ];
            let fins = [palette::Finishing::default(), palette::Finishing { luminance: 0.8, green: 0.6, saturation: 1.5, detail: 0.0, chroma: 0.0, neutral: 0.0 }, palette::Finishing { luminance: 0.0, green: 0.0, saturation: 1.0, detail: 0.0, chroma: 0.0, neutral: 0.0 }];
            for (k, pr) in palette::presets(p.kind, &names).iter().enumerate() {
                let mut r = Recipe::for_channels(&names);
                r.stars.mode = render::stars::StarMode::Keep;
                r.apply_preset(pr, &names);
                r.stretch = opts[k % 3].clone();
                r.finish = fins[k % 3].clone();
                let coef = render::coefficients(&p, &r);
                let planes: Vec<Vec<f32>> = p.channels.iter().map(|c| idx.iter().map(|&i| c.proxy.data[i]).collect()).collect();
                let refs: Vec<&[f32]> = planes.iter().map(|v| v.as_slice()).collect();
                let n = idx.len();
                let (mut rr, mut gg, mut bb) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
                palette::process(&refs, &coef, &mut rr, &mut gg, &mut bb);
                let w: serde_json::Map<String, serde_json::Value> = pr.weights.iter().map(|(c, v)| (c.to_string(), serde_json::json!(v))).collect();
                cases.push(serde_json::json!({
                    "preset": pr.name, "weights": w,
                    "opts": {"linear": r.stretch.linear, "normalise": true, "boostLimit": palette::BOOST_NOISE_LIMIT, "target": r.stretch.background, "black": r.stretch.black, "clip": r.stretch.clip, "neutralSky": r.stretch.neutral_sky, "darkRef": r.stretch.dark_ref, "keepColour": r.stretch.keep_colour, "haLum": r.finish.luminance, "green": r.finish.green, "sat": r.finish.saturation},
                    "gain": coef.gain, "r": rr, "g": gg, "b": bb,
                }));
            }
            let doc = serde_json::json!({"object": t.object, "channels": names, "stats": stats, "pixels": px, "cases": cases});
            std::fs::write(json, serde_json::to_vec(&doc)?)?;
            println!("{}: {} channels, {} pixels, {} cases → {}", t.object, names.len(), idx.len(), doc["cases"].as_array().map(|a| a.len()).unwrap_or(0), json.display());
        }
        DevCmd::MosaicSolve { master } => {
            let cat = crux_core::catalog::Catalog::find().ok_or_else(|| anyhow::anyhow!("no star database"))?;
            let learned = std::sync::Mutex::new(crux_core::solve::Learned::default());
            let (g, stars) = crux_core::process::mosaic::solve_panel(master, &cat, &learned, true)?;
            println!("{} clean stars; solved at {:.4}\"/px, rotation {:.2}°, {} matched", stars.len(), g.solution.scale, g.solution.rotation, g.matched);
            println!("residual {:.3} px linear, {:.3} px with distortion (up to {:.1} px)", g.rms_linear, g.rms, g.distortion_max);
            // The true local plate solution at the centre, from the full model.
            {
                let (cx, cy) = (g.width as f64 / 2.0, g.height as f64 / 2.0);
                let (ra0, de0) = g.pixel_to_sky(cx, cy);
                let st = |x: f64, y: f64| { let (r, d) = g.pixel_to_sky(x, y); crux_core::catalog::project(ra0, de0, r, d).unwrap() };
                let h = 50.0;
                let (xp, xm, yp, ym) = (st(cx + h, cy), st(cx - h, cy), st(cx, cy + h), st(cx, cy - h));
                let cd = [(xp.0 - xm.0) / (2.0 * h), (yp.0 - ym.0) / (2.0 * h), (xp.1 - xm.1) / (2.0 * h), (yp.1 - ym.1) / (2.0 * h)];
                let sc = (cd[0] * cd[3] - cd[1] * cd[2]).abs().sqrt() * 3600.0;
                println!("true centre: {:.5} {:.5}, {:.4}\"/px, rotation {:.3}° (solver: {:.5} {:.5}, {:.4}\"/px, {:.3}°; centre {:.2}\" apart)", ra0, de0, sc, crux_core::solve::rotation_of(&cd), g.solution.ra, g.solution.dec, g.solution.scale, g.solution.rotation, crux_core::catalog::separation(ra0, de0, g.solution.ra, g.solution.dec) * 3600.0);
                // Distortion left once the best straight-line model is taken out.
                let mut rows = Vec::new();
                for gy in 0..=8 {
                    for gx in 0..=8 {
                        let (x, y) = (gx as f64 * g.width as f64 / 8.0, gy as f64 * g.height as f64 / 8.0);
                        rows.push((x, y, st(x, y)));
                    }
                }
                let fit = |f: &dyn Fn(&(f64, f64)) -> f64| {
                    let mut m = [[0f64; 4]; 3];
                    for (x, y, p) in &rows {
                        let r = [1.0, x - cx, y - cy];
                        for i in 0..3 { for j in 0..3 { m[i][j] += r[i] * r[j]; } m[i][3] += r[i] * f(p); }
                    }
                    for c in 0..3 { for r in 0..3 { if r != c { let k = m[r][c] / m[c][c]; for j in 0..4 { m[r][j] -= k * m[c][j]; } } } }
                    [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]]
                };
                let (a, b) = (fit(&|p| p.0), fit(&|p| p.1));
                let px = sc / 3600.0;
                let worst = rows.iter().map(|(x, y, p)| ((a[0] + a[1] * (x - cx) + a[2] * (y - cy) - p.0).hypot(b[0] + b[1] * (x - cx) + b[2] * (y - cy) - p.1)) / px).fold(0.0, f64::max);
                println!("non-linear distortion: up to {worst:.1} px at the edges");
            }
            if let Some(d) = &g.distortion {
                for y in [0.0, 0.5, 1.0] {
                    let row: Vec<String> = [0.0, 0.5, 1.0].iter().map(|x| { let (dx, dy) = d.at(x * g.width as f64, y * g.height as f64); format!("({dx:+6.1},{dy:+6.1})") }).collect();
                    println!("  {}", row.join(" "));
                }
            }
        }
        #[cfg(feature = "ml")]
        DevCmd::Denoise { image, model, control, out, strength, tile, png, at, noise: noise_in, sky: sky_in, fwhm: fwhm_in, residual } => {
            use crux_core::process::combine;
            use crux_core::render::net;
            let t0 = Instant::now();
            let img = crux_core::read_image(image)?;
            let noise = noise_in.unwrap_or_else(|| combine::noise_of(&img));
            let sky = if let Some(s) = sky_in { *s } else {
                let mut v: Vec<f32> = img.data.iter().step_by(11).copied().filter(|v| *v > 0.0).collect();
                crux_core::stats::median_mut(&mut v)
            };
            let mut cfg = crux_core::config::Config::default();
            cfg.analysis.trails = false;
            let fwhm = fwhm_in.unwrap_or_else(|| crux_core::analysis::analyze_light(&img, &cfg).fwhm.unwrap_or(3.0));
            println!("{}: {}×{}, noise {:.2} ADU, sky {:.1} ADU, FWHM {fwhm:.2} px ({:.1} s)", image.display(), img.width, img.height, noise * img.adu_scale, sky * img.adu_scale, t0.elapsed().as_secs_f64());
            let t1 = Instant::now();
            let d = match model {
                Some(m) => net::Network::load(m, *tile)?,
                None => net::Network::builtin(net::Kind::Denoise, *tile)?,
            };
            println!("model loaded in {:.1} s, tiles {} px", t1.elapsed().as_secs_f64(), d.tile);
            let t2 = Instant::now();
            let st = net::Settings { strength: *strength, ..Default::default() };
            let res = d.run(&img, noise, sky, fwhm, control, &st, &|k, n| if k % 20 == 0 || k == n { eprintln!("  tile {k}/{n}") }, &std::sync::atomic::AtomicBool::new(false))?;
            println!("denoised in {:.1} s; noise now {:.2} ADU (×{:.2} lower)", t2.elapsed().as_secs_f64(), combine::noise_of(&res) * res.adu_scale, noise / combine::noise_of(&res).max(1e-12));
            if out.extension().is_some_and(|e| e.eq_ignore_ascii_case("fits") || e.eq_ignore_ascii_case("fit")) {
                crux_core::fits::write_image(out, &res, -32, &crux_core::header::Header::default())?;
            } else {
                crux_core::xisf::write_image(out, &res, &crux_core::header::Header::default(), None)?;
            }
            println!("wrote {}", out.display());
            if let Some(p) = residual {
                let k = 4usize;
                let (bw, bh) = (img.width / k, img.height / k);
                let scale = 127.0 / (3.0 * noise / k as f32);
                let mut px = vec![128u8; bw * bh];
                for by in 0..bh {
                    for bx in 0..bw {
                        let mut sum = 0f32;
                        for y in by * k..by * k + k {
                            for x in bx * k..bx * k + k {
                                let i = y * img.width + x;
                                sum += img.data[i] - res.data[i];
                            }
                        }
                        px[by * bw + bx] = (128.0 + sum / (k * k) as f32 * scale).round().clamp(0.0, 255.0) as u8;
                    }
                }
                let f = std::io::BufWriter::new(std::fs::File::create(p)?);
                let mut e = png::Encoder::new(f, bw as u32, bh as u32);
                e.set_color(png::ColorType::Grayscale);
                e.set_depth(png::BitDepth::Eight);
                e.write_header()?.write_image_data(&px)?;
                println!("wrote {} (residual, 1/{k} scale)", p.display());
            }
            if let Some(p) = png {
                let (cw, ch) = (900usize.min(img.width), 700usize.min(img.height));
                let (cx, cy) = at.as_ref().map(|v| (v[0], v[1])).unwrap_or((img.width / 2, img.height / 2));
                let x0 = cx.saturating_sub(cw / 2).min(img.width - cw);
                let y0 = cy.saturating_sub(ch / 2).min(img.height - ch);
                let cut = |im: &crux_core::image::Image| crux_core::image::Image::from_data(cw, ch, (y0..y0 + ch).flat_map(|y| im.data[y * im.width + x0..y * im.width + x0 + cw].iter().copied()).collect());
                let (a, b) = (cut(&img), cut(&res));
                let stf = crux_core::stretch::Stf::auto(&a);
                let gap = 8;
                let w = cw * 2 + gap;
                let mut px = vec![60u8; w * ch];
                for y in 0..ch {
                    for x in 0..cw {
                        px[y * w + x] = (stf.apply(a.data[y * cw + x]) * 255.0).round().clamp(0.0, 255.0) as u8;
                        px[y * w + cw + gap + x] = (stf.apply(b.data[y * cw + x]) * 255.0).round().clamp(0.0, 255.0) as u8;
                    }
                }
                let f = std::io::BufWriter::new(std::fs::File::create(p)?);
                let mut e = png::Encoder::new(f, w as u32, ch as u32);
                e.set_color(png::ColorType::Grayscale);
                e.set_depth(png::BitDepth::Eight);
                e.write_header()?.write_image_data(&px)?;
                println!("wrote {} (before | after, x {x0}–{}, y {y0}–{})", p.display(), x0 + cw, y0 + ch);
            }
        }
        DevCmd::ExportFits { files, out_dir, prefix } => {
            std::fs::create_dir_all(out_dir)?;
            for f in files {
                let img = crux_core::read_image(f)?;
                let stem = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                let out = out_dir.join(format!("{prefix}{stem}.fits"));
                let mut h = crux_core::header::Header::default();
                h.push("ADUSCALE", format!("{}", img.adu_scale), "ADU = value x ADUSCALE");
                crux_core::fits::write_image(&out, &img, -32, &h)?;
                println!("{} → {}", f.display(), out.display());
            }
        }
        DevCmd::Starless { files, out_dir, prefix } => {
            use crux_core::render::stars;
            std::fs::create_dir_all(out_dir)?;
            let st = stars::StarSettings::default();
            for f in files {
                let t0 = Instant::now();
                let img = crux_core::read_image(f)?;
                let discs = stars::dedupe(stars::detect(&img, &st));
                let m = stars::mask(&discs, img.width, img.height);
                let noise = crux_core::process::combine::noise_of(&img);
                let sl = stars::starless(&img, &m, noise, 1);
                let stem = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                let out = out_dir.join(format!("{prefix}{stem}.fits"));
                crux_core::fits::write_image(&out, &sl, -32, &crux_core::header::Header::default())?;
                println!("{}: {} stars removed in {:.1} s → {}", f.display(), discs.len(), t0.elapsed().as_secs_f64(), out.display());
            }
        }
        DevCmd::GradientCheck { master, amount, degree } => {
            use crux_core::render::gradient;
            let img = crux_core::read_image(master)?;
            let s = gradient::GradientSettings { degree: *degree, ..Default::default() };
            let (base, br) = gradient::fit(&img, &s).ok_or_else(|| anyhow::anyhow!("no fit"))?;
            let d = br.degree;
            let fixed = gradient::GradientSettings { degree: d, ..s.clone() };
            let (w, h) = (img.width, img.height);
            // A tilt plus a curve, `amount` of the background across the frame.
            let shapes: [(&str, Box<dyn Fn(f64, f64) -> f64>); 3] = [
                ("left-right tilt", Box::new(|u: f64, _v: f64| 0.5 * u)),
                ("diagonal tilt", Box::new(|u: f64, v: f64| 0.25 * (u + v))),
                ("corner glow", Box::new(|u: f64, v: f64| 0.25 * ((u + 1.0).powi(2) + (v + 1.0).powi(2)) / 4.0 - 0.25)),
            ];
            println!("{}: {}×{}, background {:.5}, own surface degree {d} ({:.1}% across, {} of {} boxes kept)", master.display(), w, h, base.level, br.amplitude_frac * 100.0, br.kept, br.boxes);
            for (name, f) in shapes.iter() {
                if d < 2 && name.contains("glow") {
                    continue;
                }
                let mut test = img.clone();
                let a = amount * base.level;
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        if test.data[i] != 0.0 {
                            let (u, v) = (2.0 * x as f64 / w as f64 - 1.0, 2.0 * y as f64 / h as f64 - 1.0);
                            test.data[i] += (a * f(u, v)) as f32;
                        }
                    }
                }
                let (m, r) = gradient::fit(&test, &fixed).ok_or_else(|| anyhow::anyhow!("no fit"))?;
                let mut worst: f64 = 0.0;
                let (mut lo, mut hi) = (f64::MAX, f64::MIN);
                for j in 0..33 {
                    for i in 0..33 {
                        let (x, y) = (i as f64 * w as f64 / 32.0, j as f64 * h as f64 / 32.0);
                        let (u, v) = (2.0 * x / w as f64 - 1.0, 2.0 * y / h as f64 - 1.0);
                        let inj = a * f(u, v);
                        lo = lo.min(inj);
                        hi = hi.max(inj);
                        let got = (m.at(x, y) - m.level) - (base.at(x, y) - base.level);
                        worst = worst.max((got - (inj - 0.0)).abs());
                    }
                }
                // Offsets don't matter (the level is kept): compare shapes.
                let _ = r;
                println!("  {name:<16} injected {:.1}% of the background: worst error {:.2}% of the background ({:.1}% of the gradient)", amount * 100.0, worst / base.level * 100.0, worst / (hi - lo) * 100.0);
            }
        }
        DevCmd::Align { input, to, out } => {
            let a = crux_core::read_image(input)?;
            let r = crux_core::read_image(to)?;
            let cfg = crux_core::config::Config::default();
            let ma = crux_core::analysis::analyze_light(&a, &cfg);
            let mr = crux_core::analysis::analyze_light(&r, &cfg);
            let reg = process::register::register(&mr.stars, &ma.stars, r.width as f64, r.height as f64).ok_or_else(|| anyhow::anyhow!("couldn't match the stars"))?;
            let (dx, dy, rot, scale) = reg.to_ref.summary();
            println!("registered: {} stars, {:.3} px rms; shift ({dx:+.1}, {dy:+.1}), rotation {rot:+.3} deg, scale {scale:.5}", reg.matched, reg.rms);
            let inv = reg.to_ref.inverse().ok_or_else(|| anyhow::anyhow!("singular transform"))?;
            let img = process::resample::warp(&a, &inv, r.width, r.height, None);
            crux_core::xisf::write_image(out, &img, &Default::default(), None)?;
            let stars: Vec<(f32, f32)> = mr.stars.iter().filter(|s| s.flags == 0).map(|s| (s.x, s.y)).collect();
            compare_masters(to, out, &stars, mr.fwhm.unwrap_or(3.0), Some(crux_core::process::combine::noise_of(&a) as f64))?;
        }
        DevCmd::Stack { files, bias, kind, rejection, out, compare } => {
            let t0 = Instant::now();
            let b = bias.as_ref().map(|p| crux_core::read_image(p)).transpose()?;
            let mut frames = Vec::new();
            for f in files {
                let mut img = crux_core::read_image(f)?;
                if let Some(b) = &b {
                    for (v, bv) in img.data.iter_mut().zip(&b.data) {
                        *v -= bv;
                    }
                }
                frames.push(img);
            }
            println!("read {} frames in {:.1} s", frames.len(), t0.elapsed().as_secs_f64());
            let rej = match rejection.as_str() {
                "esd" => integrate::Rejection::Esd { fraction: 0.3, alpha: 0.05 },
                "winsor" => integrate::Rejection::Winsorized { low: 4.0, high: 3.0 },
                "sigma" => integrate::Rejection::Sigma { low: 4.0, high: 3.0 },
                "none" => integrate::Rejection::None,
                _ => integrate::Rejection::auto(frames.len()),
            };
            let norm = match kind.as_str() {
                "flat" => integrate::Normalise::Multiplicative,
                _ => integrate::Normalise::Additive,
            };
            let t1 = Instant::now();
            let s = integrate::stack(&frames, &integrate::Params { normalise: norm, rejection: rej })?;
            println!("stacked with {} in {:.1} s; rejected {:.3}%", rej.name(), t1.elapsed().as_secs_f64(), s.rejected_total * 100.0);
            if let Some(o) = out {
                crux_core::xisf::write_image(o, &s.image, &Default::default(), None)?;
            }
            if let Some(r) = compare {
                let rf = crux_core::read_image(r)?;
                // Masters differ in overall level; compare normalised to the mean.
                let (ma, mb) = (calibrate::flat_mean(&s.image) as f64, calibrate::flat_mean(&rf) as f64);
                let a = crux_core::Image::from_data(s.image.width, s.image.height, s.image.data.iter().map(|v| (*v as f64 / ma) as f32).collect());
                let bb = crux_core::Image::from_data(rf.width, rf.height, rf.data.iter().map(|v| (*v as f64 / mb) as f32).collect());
                println!("levels {:.6} vs {:.6}; normalised to 1:", ma, mb);
                print_diff(&process::diff(&a, &bb)?, 1.0);
            }
        }
    }
    Ok(())
}

/// AlignmentMatrix and AlignmentOrigin from a PixInsight .xdrz file.
pub fn read_xdrz(p: &std::path::Path) -> Result<([f64; 9], f64)> {
    let s = std::fs::read_to_string(p)?;
    let grab = |tag: &str| -> Option<String> {
        let a = s.find(&format!("<{tag}"))?;
        let rest = &s[a..];
        let b = rest.find('>')?;
        let e = rest.find(&format!("</{tag}>")).unwrap_or(b);
        Some(rest[..e].to_string())
    };
    let m = grab("AlignmentMatrix").ok_or_else(|| anyhow::anyhow!("no AlignmentMatrix"))?;
    let m = &m[m.find('>').unwrap() + 1..];
    let v: Vec<f64> = m.split(',').map(|x| x.trim().parse::<f64>()).collect::<Result<_, _>>()?;
    anyhow::ensure!(v.len() == 9, "matrix has {} values", v.len());
    let origin = grab("AlignmentOrigin").and_then(|o| {
        let i = o.find("x=\"")? + 3;
        o[i..].split('"').next()?.parse::<f64>().ok()
    }).unwrap_or(0.0);
    Ok((v.try_into().unwrap(), origin))
}

/// "Loading target file: <path>" followed by "Projective transformation matrix:".
pub fn parse_wbpp_matrices(p: &std::path::Path) -> Result<Vec<(String, [f64; 9])>> {
    let s = std::fs::read_to_string(p)?;
    let lines: Vec<&str> = s.lines().collect();
    let mut out = Vec::new();
    let mut cur: Option<String> = None;
    for (i, l) in lines.iter().enumerate() {
        if let Some(k) = l.find("Loading target file: ") {
            cur = Some(l[k + 21..].trim().to_string());
        }
        if l.contains("Projective transformation matrix:") && i + 3 < lines.len() {
            let mut m = Vec::new();
            for r in &lines[i + 1..i + 4] {
                let body = r.split("] ").nth(1).unwrap_or(r);
                m.extend(body.split_whitespace().filter_map(|x| x.parse::<f64>().ok()));
            }
            if let (Some(c), true) = (&cur, m.len() == 9) {
                out.push((c.clone(), m.try_into().unwrap()));
            }
        }
    }
    Ok(out)
}

/// `crux dev render-many`.
fn render_many(dir: &std::path::Path, masters: bool, workspace: &std::path::Path, recipes: &std::path::Path, out_dir: &std::path::Path) -> Result<()> {
    use crux_core::render::{self, prepcache, recipe::Recipe};
    let named = if masters && dir.file_name().is_some_and(|n| n.eq_ignore_ascii_case("master")) { dir.parent().unwrap_or(dir) } else { dir };
    let out = process::job::out_dir(named, workspace);
    let t = if masters {
        render::targets_from_masters(dir, &named.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()).into_iter().next()
    } else {
        render::targets(&out).into_iter().next()
    }
    .ok_or_else(|| anyhow::anyhow!("no masters for {}", dir.display()))?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(recipes)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "toml")).collect();
    files.sort();
    std::fs::create_dir_all(out_dir)?;
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let names = t.channels();
    let mut cache: Option<(String, render::Prepared)> = None;
    let t0 = Instant::now();
    let mut stats = serde_json::Map::new();
    for f in &files {
        let mut r = Recipe::load(f)?;
        r.fit_to(&names);
        let te = render::for_recipe(&t, &out, &r, &|_| {}, &cancel)?;
        let sig = prepcache::signature(&te, &r.gradient, &r.stars);
        if cache.as_ref().is_none_or(|(s, _)| *s != sig) {
            let p = prepcache::prepare(&te, &out, &r.gradient, &r.stars, &cancel)?;
            let ch: Vec<serde_json::Value> = p
                .channels
                .iter()
                .map(|c| {
                    let s = &c.stats;
                    let q = |v: f64| crux_core::render::palette::quantile_of_sorted(&s.sorted, v);
                    serde_json::json!({
                        "name": c.name, "median": s.median, "mad": s.mad, "noise": s.noise, "sky_noise": s.sky_noise,
                        "p_median": s.p_median, "p_high": s.p_high, "q005": q(0.005), "q05": q(0.05), "q25": q(0.25), "q75": q(0.75), "q99": q(0.99), "q999": q(0.999),
                        "gradient": c.report.as_ref().map(|g| serde_json::json!({"degree": g.degree, "amplitude": g.amplitude_frac, "residual": g.residual_frac, "kept": g.kept, "usable": g.samples.len()})),
                    })
                })
                .collect();
            stats.insert(sig.clone(), serde_json::json!({"kind": format!("{:?}", p.kind), "stars": p.n_stars, "star_cover": p.star_cover, "channels": ch}));
            cache = Some((sig, p));
        }
        let p = &cache.as_ref().unwrap().1;
        let mut rgb = render::render_proxy(p, &r);
        // The recipe's crop, else crux's own (empty and thinly covered edges).
        if let Some(c) = r.crop.or_else(|| render::auto_crop(p)) {
            let (x, y, w, h) = c.pixels(rgb.width, rgb.height);
            let cut = |v: &[f32]| -> Vec<f32> { (y..y + h).flat_map(|yy| v[yy * rgb.width + x..yy * rgb.width + x + w].to_vec()).collect() };
            rgb = render::Rgb { width: w, height: h, r: cut(&rgb.r), g: cut(&rgb.g), b: cut(&rgb.b) };
        }
        let stem = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        render::write_png8(&out_dir.join(format!("{stem}.png")), &rgb)?;
    }
    std::fs::write(out_dir.join("stats.json"), serde_json::to_vec_pretty(&serde_json::Value::Object(stats))?)?;
    println!("{}: {} renders in {:.1} s", t.object, files.len(), t0.elapsed().as_secs_f64());
    Ok(())
}
