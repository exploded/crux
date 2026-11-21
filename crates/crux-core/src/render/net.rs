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
    plan: Plan,
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

impl Network {
    /// A built-in model.
    pub fn builtin(kind: Kind, tile: usize) -> Result<Network> {
        let (onnx, json, tag) = builtin(kind).ok_or_else(|| anyhow::anyhow!("this build of crux has no {} network", kind.format()))?;
        let meta: Meta = serde_json::from_str(json).context("built-in model metadata")?;
        Self::from_parts(&mut std::io::Cursor::new(onnx), meta, kind, tile, tag.to_string())
    }

    /// `onnx` with its `.json` beside it; `tile` is rounded up to the
    /// model's multiple.
    pub fn load(onnx: &Path, tile: usize) -> Result<Network> {
        let meta_path = onnx.with_extension("json");
        let meta: Meta = serde_json::from_slice(&std::fs::read(&meta_path).with_context(|| format!("read {}", meta_path.display()))?)
            .with_context(|| format!("parse {}", meta_path.display()))?;
        let kind = [Kind::Denoise, Kind::Sharpen, Kind::Starless].into_iter().find(|k| k.format() == meta.format).ok_or_else(|| anyhow::anyhow!("{}: not a crux network ({})", meta_path.display(), meta.format))?;
        let mut f = std::fs::File::open(onnx).with_context(|| format!("read {}", onnx.display()))?;
        let tag = onnx.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        Self::from_parts(&mut f, meta, kind, tile, tag)
    }

    fn from_parts(r: &mut dyn std::io::Read, meta: Meta, kind: Kind, tile: usize, tag: String) -> Result<Network> {
        anyhow::ensure!(meta.format == kind.format() && meta.version == 1, "not a {} network this version understands", kind.format());
        let m = meta.multiple.max(1);
        let tile = tile.div_ceil(m) * m;
        let plan = tract_onnx::onnx().model_for_read(r)?.with_input_fact(0, f32::fact([1, meta.channels(), tile, tile]).into())?.into_optimized()?.into_runnable()?;
        Ok(Network { plan, meta, tile, tag })
    }

    /// Run on `img`, whose background noise is `noise` and sky level `sky`
    /// (normalised units, as `combine::noise_of` and the median give), with
    /// stars of `fwhm` pixels; `controls` fill channels 2… (missing ones are
    /// zero).
    #[allow(clippy::too_many_arguments)]
    pub fn run(&self, img: &Image, noise: f32, sky: f32, fwhm: f32, controls: &[f32], s: &Settings, progress: &(dyn Fn(usize, usize) + Sync), cancel: &AtomicBool) -> Result<Image> {
        anyhow::ensure!(noise > 0.0, "noise must be positive");
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
        let done = AtomicUsize::new(0);
        let results: Vec<Result<(usize, usize, Vec<f32>)>> = pool().install(|| {
            tiles
            .par_iter()
            .map(|&(x0, y0)| {
                if cancel.load(Ordering::Relaxed) {
                    anyhow::bail!("cancelled");
                }
                let (tw, th) = (t.min(w), t.min(h));
                let mut input = vec![0f32; nch * t * t];
                for y in 0..t {
                    // Reflect into the image when it's smaller than a tile.
                    let sy = reflect(y0 + y, h);
                    for x in 0..t {
                        let sx = reflect(x0 + x, w);
                        let v = img.data[sy * w + sx];
                        let v = if v > 0.0 { v } else { sky };
                        input[y * t + x] = ((v - sky) / kn).asinh();
                    }
                }
                for (c, v) in consts.iter().enumerate() {
                    input[(c + 1) * t * t..(c + 2) * t * t].fill(*v);
                }
                let tensor: Tensor = tract_ndarray::Array4::from_shape_vec((1, nch, t, t), input)?.into();
                let out = self.plan.run(tvec!(tensor.into()))?;
                let y = out[0].to_plain_array_view::<f32>()?;
                let y = y.as_slice().context("output layout")?;
                let mut res = vec![0f32; tw * th];
                for yy in 0..th {
                    for xx in 0..tw {
                        res[yy * tw + xx] = sky + kn * y[yy * t + xx].sinh();
                    }
                }
                progress(done.fetch_add(1, Ordering::Relaxed) + 1, tiles.len());
                Ok((x0, y0, res))
            })
            .collect()
        });
        let mut acc = vec![0f32; w * h];
        let mut wsum = vec![0f32; w * h];
        for r in results {
            let (x0, y0, res) = r?;
            let (tw, th) = (t.min(w), t.min(h));
            for yy in 0..th {
                let wy = ramp(yy, y0, h);
                for xx in 0..tw {
                    let wt = (wy * ramp(xx, x0, w)).max(1e-6);
                    let k = (y0 + yy) * w + x0 + xx;
                    acc[k] += wt * res[yy * tw + xx];
                    wsum[k] += wt;
                }
            }
        }
        let a = s.strength;
        let data: Vec<f32> = img
            .data
            .par_iter()
            .zip(acc.par_iter().zip(wsum.par_iter()))
            .map(|(&v, (&s, &ws))| if v > 0.0 { (v + a * (s / ws - v)).max(1e-7) } else { 0.0 })
            .collect();
        let mut out = Image::from_data(w, h, data);
        out.adu_scale = img.adu_scale;
        Ok(out)
    }
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
