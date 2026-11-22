//! crux's trained networks (`tools/ml`), run on the CPU with tract: the
//! denoiser and, when trained, the star corrector / sharpener and star
//! remover. Built-in models are compiled into the binary (`models/`).
//!
//! Every network works in a noise-normalised domain: channel 0 is
//! `asinh((v − sky) / (k · noise))`, channel 1 holds FWHM ÷ `fwhm_scale`,
//! and any further channels are constant controls (e.g. sharpen 0/1), as
//! the model's JSON lists in `inputs`. The output is channel 0's domain.
//! Images go through in square tiles (one shape, so tract optimises the
//! model once), overlapping and blended with a cosine ramp. Empty (zero)
//! pixels stay empty.

use crate::image::Image;
use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::Deserialize;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tract_onnx::prelude::*;

#[derive(Debug, Clone, Deserialize)]
pub struct Meta {
    pub format: String,
    pub version: u32,
    pub k: f32,
    pub fwhm_scale: f32,
    pub multiple: usize,
    #[serde(default)]
    pub inputs: Vec<String>,
}

impl Meta {
    pub fn channels(&self) -> usize {
        self.inputs.len().max(2)
    }
}

/// Which network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Denoise,
    Sharpen,
    Starless,
}

impl Kind {
    pub fn format(self) -> &'static str {
        match self {
            Kind::Denoise => "crux-denoise",
            Kind::Sharpen => "crux-sharpen",
            Kind::Starless => "crux-starless",
        }
    }
}

/// The models compiled into crux: (ONNX, JSON, a version tag for caches).
pub fn builtin(kind: Kind) -> Option<(&'static [u8], &'static str, &'static str)> {
    match kind {
        Kind::Denoise => Some((include_bytes!("../../models/denoise-2.onnx"), include_str!("../../models/denoise-2.json"), "denoise-2")),
        Kind::Sharpen => builtin_sharpen(),
        Kind::Starless => builtin_starless(),
    }
}

// Filled in as the networks are trained (see models/).
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/extra.rs"));

type Plan = std::sync::Arc<TypedRunnableModel>;

pub struct Network {
    onnx: std::borrow::Cow<'static, [u8]>,
    /// tract's plan for `tile`, made on first use (not needed while the
    /// GPU runs the network).
    plan: std::sync::OnceLock<std::result::Result<Plan, String>>,
    #[cfg(feature = "gpu")]
    gpu: std::sync::OnceLock<Option<std::sync::Arc<crate::gpu::nn::Program>>>,
    pub meta: Meta,
    pub tile: usize,
    /// Version tag, for caches.
    pub tag: String,
}

#[derive(Debug, Clone, Copy)]
pub struct Settings {
    /// 0 = untouched, 1 = the network's output.
    pub strength: f32,
    /// Overlap between tiles, pixels.
    pub overlap: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { strength: 1.0, overlap: 32 }
    }
}

fn tract_plan(onnx: &[u8], channels: usize, tile: usize) -> Result<Plan> {
    Ok(tract_onnx::onnx().model_for_read(&mut std::io::Cursor::new(onnx))?.with_input_fact(0, f32::fact([1, channels, tile, tile]).into())?.into_optimized()?.into_runnable()?)
}

impl Network {
    /// A built-in model.
    pub fn builtin(kind: Kind, tile: usize) -> Result<Network> {
        let (onnx, json, tag) = builtin(kind).ok_or_else(|| anyhow::anyhow!("this build of crux has no {} network", kind.format()))?;
        let meta: Meta = serde_json::from_str(json).context("built-in model metadata")?;
        Self::from_parts(std::borrow::Cow::Borrowed(onnx), meta, kind, tile, tag.to_string())
    }

    /// The built-in model with 512 px tiles, made once per process and
    /// kept (tract's plan and the GPU's copy with it).
    pub fn shared(kind: Kind) -> Result<std::sync::Arc<Network>> {
        static NETS: std::sync::Mutex<Vec<(Kind, std::sync::Arc<Network>)>> = std::sync::Mutex::new(Vec::new());
        let mut nets = NETS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, n)) = nets.iter().find(|(k, _)| *k == kind) {
            return Ok(n.clone());
        }
        let n = std::sync::Arc::new(Network::builtin(kind, 512)?);
        nets.push((kind, n.clone()));
        Ok(n)
    }

    /// `onnx` with its `.json` beside it; `tile` is rounded up to the
    /// model's multiple.
    pub fn load(onnx: &Path, tile: usize) -> Result<Network> {
        let meta_path = onnx.with_extension("json");
        let meta: Meta = serde_json::from_slice(&std::fs::read(&meta_path).with_context(|| format!("read {}", meta_path.display()))?)
            .with_context(|| format!("parse {}", meta_path.display()))?;
        let kind = [Kind::Denoise, Kind::Sharpen, Kind::Starless].into_iter().find(|k| k.format() == meta.format).ok_or_else(|| anyhow::anyhow!("{}: not a crux network ({})", meta_path.display(), meta.format))?;
        let bytes = std::fs::read(onnx).with_context(|| format!("read {}", onnx.display()))?;
        let tag = onnx.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        Self::from_parts(std::borrow::Cow::Owned(bytes), meta, kind, tile, tag)
    }

    fn from_parts(onnx: std::borrow::Cow<'static, [u8]>, meta: Meta, kind: Kind, tile: usize, tag: String) -> Result<Network> {
        anyhow::ensure!(meta.format == kind.format() && meta.version == 1, "not a {} network this version understands", kind.format());
        let m = meta.multiple.max(1);
        let tile = tile.div_ceil(m) * m;
        Ok(Network {
            onnx,
            plan: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            gpu: std::sync::OnceLock::new(),
            meta,
            tile,
            tag,
        })
    }

    fn plan(&self) -> Result<&Plan> {
        self.plan.get_or_init(|| tract_plan(&self.onnx, self.meta.channels(), self.tile).map_err(|e| format!("{e:#}"))).as_ref().map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// The network on the GPU, when there is one and it gives the same
    /// answers as tract (checked once per network per process on a small
    /// tile).
    #[cfg(feature = "gpu")]
    fn gpu_program(&self) -> Option<std::sync::Arc<crate::gpu::nn::Program>> {
        let gpu = crate::gpu::get()?;
        self.gpu
            .get_or_init(|| {
                let made = (|| -> Result<crate::gpu::nn::Program> {
                    self_test(gpu, &self.onnx, self.meta.channels(), self.meta.multiple.max(1))?;
                    crate::gpu::nn::Program::new(gpu, &self.onnx, self.meta.channels(), self.tile)
                })();
                match made {
                    Ok(p) => Some(std::sync::Arc::new(p)),
                    Err(e) => {
                        crate::gpu::fail(&format!("{}: {e:#}", self.tag));
                        None
                    }
                }
            })
            .clone()
    }

    /// Which engine `run` would use now: "GPU" or "CPU".
    pub fn engine(&self) -> &'static str {
        #[cfg(feature = "gpu")]
        if crate::gpu::wanted() && self.gpu_program().is_some() {
            return "GPU";
        }
        "CPU"
    }

    /// Run on `img`, whose background noise is `noise` and sky level `sky`
    /// (normalised units, as `combine::noise_of` and the median give), with
    /// stars of `fwhm` pixels; `controls` fill channels 2… (missing ones are
    /// zero). On the GPU when there is one, else (or if it fails) on the CPU.
    #[allow(clippy::too_many_arguments)]
    pub fn run(&self, img: &Image, noise: f32, sky: f32, fwhm: f32, controls: &[f32], s: &Settings, progress: &(dyn Fn(usize, usize) + Sync), cancel: &AtomicBool) -> Result<Image> {
        self.run_on(img, noise, sky, fwhm, controls, s, progress, cancel, false)
    }

    /// `run`, on the CPU only when `cpu_only`.
    #[allow(clippy::too_many_arguments)]
    fn run_on(&self, img: &Image, noise: f32, sky: f32, fwhm: f32, controls: &[f32], s: &Settings, progress: &(dyn Fn(usize, usize) + Sync), cancel: &AtomicBool, cpu_only: bool) -> Result<Image> {
        anyhow::ensure!(noise > 0.0, "noise must be positive");
        let laps = crate::timing::Laps::new(format!("network {}", self.tag));
        let (w, h, t) = (img.width, img.height, self.tile);
        let nch = self.meta.channels();
        let ov = s.overlap.min(t / 4);
        let step = t - 2 * ov;
        let starts = |n: usize| -> Vec<usize> {
            if n <= t {
                return vec![0];
            }
            let mut v: Vec<usize> = (0..).map(|k| k * step).take_while(|&x| x + t < n).collect();
            v.push(n - t);
            v
        };
        let (xs, ys) = (starts(w), starts(h));
        let tiles: Vec<(usize, usize)> = ys.iter().flat_map(|&y| xs.iter().map(move |&x| (x, y))).collect();
        let kn = self.meta.k * noise;
        let mut consts = vec![fwhm / self.meta.fwhm_scale];
        consts.extend((0..nch.saturating_sub(2)).map(|i| controls.get(i).copied().unwrap_or(0.0)));
        // Ramp: 0 at a tile's edge (except the image's own edges), 1 inside.
        let ramp = |i: usize, start: usize, n: usize| -> f32 {
            let lo = if start == 0 { 1.0 } else { edge(i, ov) };
            let hi = if start + t >= n { 1.0 } else { edge(t - 1 - i, ov) };
            lo * hi
        };
        let (tw, th) = (t.min(w), t.min(h));
        // Tile k's input: asinh-stretched pixels, then the constant channels.
        let fill = |k: usize, input: &mut [f32]| {
            let (x0, y0) = tiles[k];
            for y in 0..t {
                // Reflect into the image when it's smaller than a tile.
                let sy = reflect(y0 + y, h);
                let row = &mut input[y * t..(y + 1) * t];
                for (x, o) in row.iter_mut().enumerate() {
                    let sx = reflect(x0 + x, w);
                    let v = img.data[sy * w + sx];
                    let v = if v > 0.0 { v } else { sky };
                    *o = ((v - sky) / kn).asinh();
                }
            }
            for (c, v) in consts.iter().enumerate() {
                input[(c + 1) * t * t..(c + 2) * t * t].fill(*v);
            }
        };
        // Blended in as each tile finishes, so finished tiles aren't kept.
        let sums = std::sync::Mutex::new((vec![0f32; w * h], vec![0f32; w * h]));
        let done = AtomicUsize::new(0);
        let merge = |k: usize, y: &[f32]| {
            let (x0, y0) = tiles[k];
            let mut g = sums.lock().unwrap_or_else(|e| e.into_inner());
            let (acc, wsum) = &mut *g;
            for yy in 0..th {
                let wy = ramp(yy, y0, h);
                let row = (y0 + yy) * w + x0;
                for xx in 0..tw {
                    let wt = (wy * ramp(xx, x0, w)).max(1e-6);
                    acc[row + xx] += wt * (sky + kn * y[yy * t + xx].sinh());
                    wsum[row + xx] += wt;
                }
            }
            drop(g);
            progress(done.fetch_add(1, Ordering::Relaxed) + 1, tiles.len());
        };
        #[allow(unused_mut)]
        let mut engine = "CPU";
        let _ = cpu_only;
        #[cfg(feature = "gpu")]
        if let Some(prog) = (!cpu_only).then(|| self.gpu_program()).flatten() {
            let r = match crate::gpu::get() {
                Some(gpu) => prog.run(
                    gpu,
                    tiles.len(),
                    &mut |k, input| fill(k, input),
                    &mut |k, y| {
                        merge(k, y);
                        Ok(())
                    },
                    cancel,
                ),
                None => Err(anyhow::anyhow!("the GPU was switched off")),
            };
            match r {
                Ok(()) => engine = "GPU",
                Err(e) => {
                    if cancel.load(Ordering::Relaxed) {
                        anyhow::bail!("cancelled");
                    }
                    crate::gpu::fail(&format!("{}: {e:#}", self.tag));
                    let mut g = sums.lock().unwrap_or_else(|e| e.into_inner());
                    g.0.iter_mut().for_each(|v| *v = 0.0);
                    g.1.iter_mut().for_each(|v| *v = 0.0);
                    done.store(0, Ordering::Relaxed);
                }
            }
        }
        if engine == "CPU" {
            let plan = self.plan()?;
            pool().install(|| {
                tiles.par_iter().enumerate().try_for_each(|(k, _)| -> Result<()> {
                    if cancel.load(Ordering::Relaxed) {
                        anyhow::bail!("cancelled");
                    }
                    let mut input = vec![0f32; nch * t * t];
                    fill(k, &mut input);
                    let tensor: Tensor = tract_ndarray::Array4::from_shape_vec((1, nch, t, t), input)?.into();
                    let out = plan.run(tvec!(tensor.into()))?;
                    let y = out[0].to_plain_array_view::<f32>()?;
                    merge(k, y.as_slice().context("output layout")?);
                    Ok(())
                })
            })?;
        }
        laps.lap(&format!("{} tiles of {t} px ({w} × {h}) on the {engine}", tiles.len()));
        let (acc, wsum) = sums.into_inner().unwrap_or_else(|e| e.into_inner());
        let a = s.strength;
        let data: Vec<f32> = img
            .data
            .par_iter()
            .zip(acc.par_iter().zip(wsum.par_iter()))
            .map(|(&v, (&s, &ws))| if v > 0.0 { (v + a * (s / ws - v)).max(1e-7) } else { 0.0 })
            .collect();
        let mut out = Image::from_data(w, h, data);
        out.adu_scale = img.adu_scale;
        laps.lap("blend");
        Ok(out)
    }
}

/// The GPU's answer against tract's on a small synthetic tile: noise, a
/// few stars (one bright) and a smooth glow. Fails on any difference above
/// 2e-3 in the network's units (asinh of noise sigmas).
#[cfg(feature = "gpu")]
fn self_test(gpu: &crate::gpu::Gpu, onnx: &[u8], nch: usize, multiple: usize) -> Result<()> {
    let t = 64usize.div_ceil(multiple) * multiple;
    let prog = crate::gpu::nn::Program::new(gpu, onnx, nch, t)?;
    let plan = tract_plan(onnx, nch, t)?;
    let input = test_tile(nch, t);
    let tensor: Tensor = tract_ndarray::Array4::from_shape_vec((1, nch, t, t), input.clone())?.into();
    let out = plan.run(tvec!(tensor.into()))?;
    let want = out[0].to_plain_array_view::<f32>()?;
    let want = want.as_slice().context("output layout")?.to_vec();
    let mut got = Vec::new();
    prog.run(
        gpu,
        1,
        &mut |_, inp| inp.copy_from_slice(&input),
        &mut |_, y| {
            got = y.to_vec();
            Ok(())
        },
        &AtomicBool::new(false),
    )?;
    let worst = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    if crate::timing::on() {
        eprintln!("GPU self-test on {}: largest difference from tract {worst:.2e}", gpu.describe());
    }
    anyhow::ensure!(got.len() == want.len() && worst.is_finite() && worst < 2e-3, "its answers differ from the CPU's by up to {worst:.2e}");
    Ok(())
}

/// A synthetic network input: channel 0 asinh-stretched noise, stars and a
/// glow; channel 1 a FWHM of 3.2 px; controls 0.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
fn test_tile(nch: usize, t: usize) -> Vec<f32> {
    let mut input = vec![0f32; nch * t * t];
    let mut seed = 0x2545F4914F6CDD1Du64;
    let c = t as f32 / 64.0;
    for y in 0..t {
        for x in 0..t {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u = (seed >> 11) as f32 / (1u64 << 53) as f32 - 0.5;
            let (fx, fy) = (x as f32 / c, y as f32 / c);
            let mut v = 3.46 * u + 2.0 * (-((fx - 40.0).powi(2) + (fy - 20.0).powi(2)) / 400.0).exp();
            for (sx, sy, a) in [(12.0, 15.0, 400.0f32), (50.0, 48.0, 60.0), (30.0, 33.0, 15.0)] {
                v += a * (-((fx - sx).powi(2) + (fy - sy).powi(2)) / 4.5).exp();
            }
            input[y * t + x] = v.asinh();
        }
    }
    for ch in 1..nch {
        input[ch * t * t..(ch + 1) * t * t].fill(if ch == 1 { 0.8 } else { 0.0 });
    }
    input
}

/// What a network pass needs to know about `img`: background noise, sky
/// level and star FWHM (pixels; 3 when no stars are found). On images over
/// 40 MP (mosaics) the FWHM comes from a central 4,096 px square when that
/// is mostly data: finding stars over all of a 176 MP mosaic took 4 s.
pub fn measure(img: &Image) -> (f32, f32, f32) {
    let noise = crate::process::combine::noise_of(img).max(1e-6);
    let sky = {
        let mut v: Vec<f32> = img.data.iter().step_by(11).copied().filter(|v| *v > 0.0).collect();
        crate::stats::median_mut(&mut v)
    };
    let mut cfg = crate::config::Config::default();
    cfg.analysis.trails = false;
    let (w, h) = (img.width, img.height);
    let crop = (w * h > 40_000_000).then(|| {
        let (cw, ch) = (w.min(4096), h.min(4096));
        let (x0, y0) = ((w - cw) / 2, (h - ch) / 2);
        let data: Vec<f32> = (y0..y0 + ch).flat_map(|y| img.data[y * w + x0..y * w + x0 + cw].iter().copied()).collect();
        let filled = data.iter().filter(|v| **v > 0.0).count();
        (filled * 2 > data.len()).then(|| {
            let mut c = Image::from_data(cw, ch, data);
            c.adu_scale = img.adu_scale;
            c
        })
    });
    let fwhm = crate::analysis::analyze_light(crop.flatten().as_ref().unwrap_or(img), &cfg).fwhm.unwrap_or(3.0);
    (noise, sky, fwhm)
}

/// The networks' own threads. A tile takes seconds, and rayon workers
/// finish every job queued on their pool before taking work handed in from
/// outside it, so on the global pool a network pass held up anything else
/// that used rayon (the GUI's previews, which render on the UI thread,
/// waited 12 s) until it ended.
fn pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("crux-net-{i}"))
            .build()
            .expect("network thread pool")
    })
}

/// Cosine ramp over the first `ov` pixels from an edge.
fn edge(i: usize, ov: usize) -> f32 {
    if ov == 0 || i >= ov {
        return 1.0;
    }
    let x = (i as f32 + 0.5) / ov as f32;
    0.5 - 0.5 * (std::f32::consts::PI * x).cos()
}

fn reflect(i: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let p = 2 * (n - 1);
    let m = i % p;
    if m < n {
        m
    } else {
        p - m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramps_and_reflection() {
        assert_eq!(reflect(5, 4), 1);
        assert_eq!(reflect(3, 4), 3);
        assert!((edge(0, 32) - 0.0024).abs() < 0.01);
        assert_eq!(edge(40, 32), 1.0);
        for i in 0..32 {
            let s = edge(i, 32) + edge(31 - i, 32);
            assert!((s - 1.0).abs() < 1e-5, "{i}: {s}");
        }
    }

    /// Several tiles in batches of up to 4 and a part batch, a ragged
    /// edge and an empty border: the GPU must give the CPU's answer.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_and_cpu_agree_across_tiles() {
        if crate::gpu::get().is_none() {
            eprintln!("no GPU here: skipped");
            return;
        }
        for kind in [Kind::Denoise, Kind::Sharpen, Kind::Starless] {
            let n = Network::builtin(kind, 128).unwrap();
            let (w, h) = (400usize, 290usize);
            let t = test_tile(1, 512);
            let data: Vec<f32> = (0..w * h).map(|k| if k % w < 3 { 0.0 } else { 0.01 + 0.0002 * t[(k / w) * 512 + k % w].sinh() }).collect();
            let img = Image::from_data(w, h, data);
            let go = |cpu| n.run_on(&img, 0.0002, 0.01, 3.0, &[1.0], &Settings::default(), &|_, _| {}, &AtomicBool::new(false), cpu).unwrap();
            let (g, c) = (go(false), go(true));
            let worst = g.data.iter().zip(&c.data).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst < 2e-6, "{kind:?}: GPU and CPU differ by {worst:e}");
            assert!(g.data[w * 10 + 1] == 0.0);
        }
    }

    #[test]
    fn builtin_denoiser_runs_and_keeps_empty_pixels() {
        let n = Network::builtin(Kind::Denoise, 64).unwrap();
        assert_eq!(n.meta.channels(), 2);
        // A flat sky with noise and an empty border.
        let (w, h) = (100usize, 80usize);
        let mut seed = 1u64;
        let mut data = vec![0f32; w * h];
        for y in 0..h {
            for x in 4..w {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
                data[y * w + x] = 0.01 + 0.0002 * u * 3.46;
            }
        }
        let img = Image::from_data(w, h, data);
        let out = n.run(&img, 0.0002, 0.01, 3.0, &[], &Settings::default(), &|_, _| {}, &AtomicBool::new(false)).unwrap();
        assert!(out.data[0] == 0.0 && out.data[w * 10 + 2] == 0.0);
        let before = crate::stats::median_sigma_mut(&mut img.data.iter().copied().filter(|v| *v > 0.0).collect::<Vec<_>>()).1;
        let after = crate::stats::median_sigma_mut(&mut out.data.iter().copied().filter(|v| *v > 0.0).collect::<Vec<_>>()).1;
        assert!(after < before * 0.6, "noise {before} → {after}");
    }
}
