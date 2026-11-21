//! PixInsight-style automatic screen transfer function (STF): clip shadows
//! at median − 2.8·MAD, then a midtones transfer that puts the median at
//! 25% brightness. Used for previews, thumbnails and the GUI viewer.

use crate::image::Image;
use crate::stats::median_sigma_mut;

/// Midtones transfer function with balance `m`.
#[inline]
pub fn mtf(x: f32, m: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stf {
    /// Shadows clipping point (normalised).
    pub c0: f32,
    /// Midtones balance.
    pub m: f32,
}

impl Stf {
    /// PixInsight's AutoStretch defaults: shadows −2.8 MAD, target 0.25.
    pub fn auto(img: &Image) -> Stf {
        let mut s: Vec<f32> = img.data.iter().step_by(13).copied().collect();
        // `sigma` is the normalised MAD (MAD × 1.4826), as PixInsight uses.
        let (med, sigma) = median_sigma_mut(&mut s);
        let c0 = (med - 2.8 * sigma).clamp(0.0, 1.0);
        // Balance that maps the (clipped) median to 0.25.
        let m = mtf(((med - c0) / (1.0 - c0)).max(1e-6), 0.25);
        Stf { c0, m }
    }

    #[inline]
    pub fn apply(&self, v: f32) -> f32 {
        mtf((v - self.c0) / (1.0 - self.c0), self.m)
    }
}

/// Bump when thumbnails change; cached ones are rebuilt.
pub const THUMB_VERSION: u32 = 1;
/// Longest side of a contact-sheet thumbnail, pixels.
pub const THUMB_SIZE: usize = 288;

/// Contact-sheet thumbnail: block-averaged and auto-stretched.
pub fn thumbnail(img: &Image) -> crate::cache::Thumb {
    let (w, h, _, data) = preview(img, THUMB_SIZE);
    crate::cache::Thumb { w: w as u32, h: h as u32, data }
}

/// Block-averaged, stretched 8-bit preview no larger than `max_dim`.
pub fn preview(img: &Image, max_dim: usize) -> (usize, usize, usize, Vec<u8>) {
    let f = img.width.max(img.height).div_ceil(max_dim).max(1);
    let (w, h) = (img.width / f, img.height / f);
    let stf = Stf::auto(img);
    let mut out = vec![0u8; w * h];
    let k = 1.0 / (f * f) as f32;
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for dy in 0..f {
                let row = &img.data[(y * f + dy) * img.width + x * f..];
                s += row[..f].iter().sum::<f32>();
            }
            out[y * w + x] = (stf.apply(s * k) * 255.0).round() as u8;
        }
    }
    (w, h, f, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtf_properties() {
        assert_eq!(mtf(0.0, 0.3), 0.0);
        assert_eq!(mtf(1.0, 0.3), 1.0);
        assert!((mtf(0.3, 0.3) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn auto_puts_median_at_quarter() {
        let (img, _) = crate::synth::render(&crate::synth::Params::default());
        let stf = Stf::auto(&img);
        let med = crate::stats::median(&img.data);
        assert!((stf.apply(med) - 0.25).abs() < 0.02, "{}", stf.apply(med));
    }
}
