//! Native processing (milestones 11–12): masters, calibration, cosmetic
//! correction, registration and integration. Reads frames where
//! they are and writes only to the workspace (`C:\crux-work` by default).

pub mod calibrate;
pub mod combine;
pub mod integrate;
pub mod job;
pub mod mosaic;
pub mod register;
pub mod resample;

use crate::header::Header;
use crate::image::Image;
use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;

/// Read a frame and its header (any supported format).
pub fn read_with_header(path: &Path) -> Result<(Image, Header)> {
    let img = crate::read_image(path).with_context(|| format!("read {}", path.display()))?;
    let hdr = crate::meta::read_meta(path).map(|m| m.header).unwrap_or_default();
    Ok((img, hdr))
}

/// Load a master for calibration.
pub fn load_master(path: &Path, label: &str) -> Result<calibrate::Master> {
    let (img, hdr) = read_with_header(path)?;
    Ok(calibrate::Master { path: path.to_path_buf(), label: label.to_string(), image: Arc::new(img), exptime: hdr.f64("EXPTIME").or_else(|| hdr.f64("EXPOSURE")) })
}

/// Difference statistics between two images (for validation).
#[derive(Debug, Clone)]
pub struct Diff {
    pub n: usize,
    pub mean_a: f64,
    pub mean_b: f64,
    /// Median and robust sigma of a − b.
    pub median: f64,
    pub sigma: f64,
    pub rms: f64,
    pub max_abs: f64,
    /// Pixels differing by more than 1e-4 (≈ 6.5 ADU).
    pub over: usize,
    /// Best-fit b ≈ gain · a + offset.
    pub gain: f64,
    pub offset: f64,
}

pub fn diff(a: &Image, b: &Image) -> Result<Diff> {
    anyhow::ensure!(a.width == b.width && a.height == b.height, "sizes differ: {}×{} vs {}×{}", a.width, a.height, b.width, b.height);
    let n = a.data.len();
    let (mut sa, mut sb, mut saa, mut sab, mut s2, mut mx, mut over) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0usize);
    let mut d: Vec<f32> = Vec::with_capacity(n / 7 + 1);
    for (i, (&x, &y)) in a.data.iter().zip(&b.data).enumerate() {
        let (x, y) = (x as f64, y as f64);
        sa += x;
        sb += y;
        saa += x * x;
        sab += x * y;
        let e = x - y;
        s2 += e * e;
        mx = mx.max(e.abs());
        if e.abs() > 1e-4 {
            over += 1;
        }
        if i % 7 == 0 {
            d.push(e as f32);
        }
    }
    let nf = n as f64;
    let gain = (sab - sa * sb / nf) / (saa - sa * sa / nf);
    let offset = (sb - gain * sa) / nf;
    let (m, s) = crate::stats::median_sigma_mut(&mut d);
    Ok(Diff { n, mean_a: sa / nf, mean_b: sb / nf, median: m as f64, sigma: s as f64, rms: (s2 / nf).sqrt(), max_abs: mx, over, gain, offset })
}
