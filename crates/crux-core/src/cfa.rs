//! One-shot-colour (Bayer) sensors.
//!
//! A raw OSC frame is one plane in which each pixel saw one colour, in a
//! repeating 2 × 2 pattern (`BAYERPAT`, shifted by `XBAYROFF`/`YBAYROFF`).
//! The pattern applies from the first stored row, which is crux's row 0
//! (N.I.N.A. writes `ROWORDER = 'TOP-DOWN'`).
//!
//! - Calibration (bias, dark, flat) works on the pattern as it is: every
//!   step is per pixel.
//! - Measuring (stars, trails, registration) uses [`luminance`]: a 3 × 3
//!   [1 2 1] kernel that gives every pixel the same mix (¼ R, ½ G, ¼ B), so
//!   there's no checkerboard and no shift, at full resolution.
//! - Hot pixels are judged and replaced against same-colour neighbours,
//!   2 px away.
//! - [`debayer`] splits a calibrated frame into R, G and B planes before
//!   registration, so each colour is warped and stacked like a filter.

use crate::header::Header;
use crate::image::Image;
use serde::{Deserialize, Serialize};

pub const R: u8 = 0;
pub const G: u8 = 1;
pub const B: u8 = 2;

/// Channel names, the "filters" OSC lights are stacked as.
pub const CHANNELS: [&str; 3] = ["R", "G", "B"];

/// The lines a dual-band filter passes, as (red channel, green + blue):
/// H-alpha and OIII → `["H", "O"]`, SII and OIII → `["S", "O"]`. Behind
/// such a filter all the light, stars included, is in those two bands, so
/// red holds the first and green and blue both hold OIII. Recognised by
/// the lines in the name ("HaO3", "haoiii", "SIIOIII") or a known filter
/// (Optolong L-eXtreme, L-eNhance, L-Ultimate; Antlia ALP-T; IDAS NBZ;
/// "duo-band").
pub fn dual_band(filter: &str) -> Option<[&'static str; 2]> {
    let f: String = filter.to_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    let o3 = f.contains("oiii") || f.contains("o3");
    let s2 = f.replace("oiii", "").contains("sii") || f.contains("s2");
    let ha = f.contains("ha") || f.contains("halpha");
    if o3 && s2 {
        return Some(["S", "O"]);
    }
    if o3 && ha {
        return Some(["H", "O"]);
    }
    let brands = ["lextreme", "lenhance", "lultimate", "alpt", "nbz", "duoband", "dualband"];
    brands.iter().any(|b| f.contains(b)).then_some(["H", "O"])
}

/// A 2 × 2 colour pattern: colours of (0,0), (1,0), (0,1), (1,1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bayer {
    pub cells: [u8; 4],
}

impl Bayer {
    /// From a pattern name (`RGGB`, `GRBG`, …) and the X/Y offsets.
    pub fn parse(pattern: &str, xoff: i64, yoff: i64) -> Option<Bayer> {
        let p: Vec<u8> = pattern
            .trim()
            .chars()
            .map(|c| match c.to_ascii_uppercase() {
                'R' => Some(R),
                'G' => Some(G),
                'B' => Some(B),
                _ => None,
            })
            .collect::<Option<_>>()?;
        if p.len() != 4 || p.iter().filter(|&&c| c == G).count() != 2 || !p.contains(&R) || !p.contains(&B) {
            return None;
        }
        let (xo, yo) = (xoff.rem_euclid(2) as usize, yoff.rem_euclid(2) as usize);
        let mut cells = [0u8; 4];
        for y in 0..2 {
            for x in 0..2 {
                cells[y * 2 + x] = p[((y + yo) & 1) * 2 + ((x + xo) & 1)];
            }
        }
        Some(Bayer { cells })
    }

    /// The colour at (x, y).
    #[inline]
    pub fn colour(&self, x: usize, y: usize) -> u8 {
        self.cells[(y & 1) * 2 + (x & 1)]
    }

    /// The pattern's name at (0, 0), e.g. "RGGB".
    pub fn name(&self) -> String {
        self.cells.iter().map(|&c| ['R', 'G', 'B'][c as usize]).collect()
    }
}

/// The frame's Bayer pattern, if its header has one and it's a single
/// plane (a debayered or stacked colour image has NAXIS3 = 3 instead).
pub fn from_header(h: &Header) -> Option<Bayer> {
    if h.i64("NAXIS3").is_some_and(|n| n > 1) {
        return None;
    }
    let pat = h.str("BAYERPAT").or_else(|| h.str("COLORTYP")).or_else(|| h.str("CFA-PAT"))?;
    Bayer::parse(&pat, h.i64("XBAYROFF").unwrap_or(0), h.i64("YBAYROFF").unwrap_or(0))
}

/// Mirror an index into 0..n keeping its parity (−1 → 1, n → n − 2), so a
/// mirrored neighbour has the same colour as the real one would.
#[inline]
fn mirror(i: isize, n: usize) -> usize {
    let n = n as isize;
    let j = if i < 0 {
        -i
    } else if i >= n {
        2 * (n - 1) - i
    } else {
        i
    };
    j.clamp(0, n - 1) as usize
}

/// Full-resolution luminance for measuring: [1 2 1] ⊗ [1 2 1] / 16. Each
/// output pixel is ¼ R + ½ G + ¼ B whatever its position in the pattern.
pub fn luminance(img: &Image) -> Image {
    let (w, h) = (img.width, img.height);
    if w < 2 || h < 2 {
        return img.clone();
    }
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        let r = img.row(y);
        let o = &mut tmp[y * w..(y + 1) * w];
        for x in 0..w {
            let a = r[mirror(x as isize - 1, w)];
            let c = r[mirror(x as isize + 1, w)];
            o[x] = 0.25 * a + 0.5 * r[x] + 0.25 * c;
        }
    }
    let mut out = Image { width: w, height: h, data: vec![0f32; w * h], adu_scale: img.adu_scale };
    for y in 0..h {
        let (ya, yc) = (mirror(y as isize - 1, h), mirror(y as isize + 1, h));
        for x in 0..w {
            out.data[y * w + x] = 0.25 * tmp[ya * w + x] + 0.5 * tmp[y * w + x] + 0.25 * tmp[yc * w + x];
        }
    }
    out
}

/// Offsets of the 8 same-colour neighbours (2 px away).
const RING2: [(isize, isize); 8] = [(-2, -2), (0, -2), (2, -2), (-2, 0), (2, 0), (-2, 2), (0, 2), (2, 2)];

fn same_colour(img: &Image, x: usize, y: usize) -> [f32; 8] {
    let mut v = [0f32; 8];
    for (k, (dx, dy)) in RING2.iter().enumerate() {
        v[k] = img.get(mirror(x as isize + dx, img.width), mirror(y as isize + dy, img.height));
    }
    v
}

/// Median of 8 values: mean of the middle two (ImageCalibration's rule).
fn median8(mut v: [f32; 8]) -> f32 {
    v.sort_by(f32::total_cmp);
    0.5 * (v[3] + v[4])
}

/// Replace one pixel with the median of its 8 same-colour neighbours.
pub fn replace_with_neighbours(img: &mut Image, x: usize, y: usize) {
    let m = median8(same_colour(img, x, y));
    img.set(x, y, m);
}

/// Noise of a raw CFA frame: MAD of the difference between each pixel and
/// its same-colour neighbour 2 px to the right, ÷ √2, on a subsample.
fn noise(img: &Image) -> f32 {
    let (w, h) = (img.width, img.height);
    if w < 4 || h < 4 {
        return 0.0;
    }
    let step = ((w * h / 200_000).max(1) as f64).sqrt().ceil() as usize;
    let mut d = Vec::new();
    for y in (0..h).step_by(step) {
        let r = img.row(y);
        for x in (0..w - 2).step_by(step) {
            d.push(r[x + 2] - r[x]);
        }
    }
    let m = crate::stats::median_mut(&mut d);
    for v in d.iter_mut() {
        *v = (*v - m).abs();
    }
    crate::stats::median_mut(&mut d) * 1.4826 / std::f32::consts::SQRT_2
}

/// Hot pixels in a raw CFA frame, for measuring (calibration uses the
/// master dark's list instead): ≥ `k`σ over the median of the 8 same-colour
/// neighbours, while none of the 4 direct neighbours (other colours) rises
/// over its own same-colour median by more than 15% of that. A star lifts
/// its direct neighbours too, even when it's only 2 px across.
pub fn hot_pixels(img: &Image, k: f32) -> Vec<(usize, usize)> {
    let (w, h) = (img.width, img.height);
    let sigma = noise(img);
    if sigma <= 0.0 || w < 5 || h < 5 {
        return Vec::new();
    }
    let t = k * sigma;
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let v = img.get(x, y);
            // Quick reject before sorting: the pixel must beat all four
            // same-colour neighbours on its row and column by t.
            let l = img.get(mirror(x as isize - 2, w), y);
            let r = img.get(mirror(x as isize + 2, w), y);
            let u = img.get(x, mirror(y as isize - 2, h));
            let d = img.get(x, mirror(y as isize + 2, h));
            if v - l.max(r).max(u).max(d) < 0.5 * t {
                continue;
            }
            let excess = v - median8(same_colour(img, x, y));
            if excess < t {
                continue;
            }
            let lifted = [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)].iter().any(|(dx, dy)| {
                let (nx, ny) = (mirror(x as isize + dx, w), mirror(y as isize + dy, h));
                img.get(nx, ny) - median8(same_colour(img, nx, ny)) > 0.15 * excess
            });
            if !lifted {
                out.push((x, y));
            }
        }
    }
    out
}

/// What the measuring code reads for an OSC frame: hot pixels replaced
/// (same-colour), then [`luminance`].
pub fn measure_image(raw: &Image) -> Image {
    let mut img = raw.clone();
    for (x, y) in hot_pixels(raw, 8.0) {
        replace_with_neighbours(&mut img, x, y);
    }
    let mut lum = luminance(&img);
    mark_clipped(&img, &mut lum);
    lum
}

/// Raw level (normalised) treated as clipped when it feeds a luminance pixel.
const CLIP_LEVEL: f32 = 0.9;

/// Lift every luminance pixel that any clipped raw pixel feeds (its 3 × 3
/// window) to full scale. The kernel otherwise averages a clipped green
/// pixel with unclipped red and blue ones to well under the saturation
/// level, so the star isn't flagged and its clipped flux reaches the
/// photometry: sharp frames lose more than soft ones, and transparency
/// followed focus (NGC 3372, ASI294MC: the two softest frames read 1.0,
/// the rest 0.7).
fn mark_clipped(raw: &Image, lum: &mut Image) {
    let (w, h) = (raw.width, raw.height);
    for y in 0..h {
        for x in 0..w {
            if raw.data[y * w + x] < CLIP_LEVEL {
                continue;
            }
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    lum.data[yy * w + xx] = 1.0;
                }
            }
        }
    }
}

/// Bilinear demosaic: each missing colour is the mean of that colour's
/// pixels in the 3 × 3 window (4 direct neighbours for green at a red or
/// blue pixel; 2 for red/blue at green; 4 diagonals for red at blue).
/// Returns R, G, B planes.
pub fn debayer(img: &Image, b: Bayer) -> [Image; 3] {
    let (w, h) = (img.width, img.height);
    let mut planes: [Image; 3] = std::array::from_fn(|_| Image { width: w, height: h, data: vec![0f32; w * h], adu_scale: img.adu_scale });
    let [pr, pg, pb] = &mut planes;
    let outs: [&mut Vec<f32>; 3] = [&mut pr.data, &mut pg.data, &mut pb.data];
    // Rows in parallel: each output row only reads the input.
    use rayon::prelude::*;
    let rows: Vec<[Vec<f32>; 3]> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut o: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; w]);
            for x in 0..w {
                let own = b.colour(x, y);
                let v = img.get(x, y);
                let mut sum = [0f32; 3];
                let mut n = [0u32; 3];
                for dy in -1isize..=1 {
                    let yy = mirror(y as isize + dy, h);
                    for dx in -1isize..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let xx = mirror(x as isize + dx, w);
                        let c = b.colour(xx, yy) as usize;
                        sum[c] += img.get(xx, yy);
                        n[c] += 1;
                    }
                }
                for c in 0..3 {
                    o[c][x] = if c as u8 == own { v } else if n[c] > 0 { sum[c] / n[c] as f32 } else { v };
                }
            }
            o
        })
        .collect();
    for (y, r) in rows.into_iter().enumerate() {
        for c in 0..3 {
            outs[c][y * w..(y + 1) * w].copy_from_slice(&r[c]);
        }
    }
    planes
}

/// How well the two greens of each 2 × 2 cell agree, as their median
/// difference over the frame's median level (0 = identical). A pattern
/// read one pixel out puts red or blue in a "green" slot, which shows up
/// here on any sky with colour in it. For checks and the report.
pub fn green_mismatch(img: &Image, b: Bayer) -> f32 {
    let g: Vec<usize> = (0..4).filter(|&i| b.cells[i] == G).collect();
    if g.len() != 2 {
        return f32::NAN;
    }
    let (g0, g1) = ((g[0] % 2, g[0] / 2), (g[1] % 2, g[1] / 2));
    let mut d = Vec::new();
    let mut lv = Vec::new();
    for y in (0..img.height.saturating_sub(1)).step_by(2) {
        for x in (0..img.width.saturating_sub(1)).step_by(2) {
            let a = img.get(x + g0.0, y + g0.1);
            let c = img.get(x + g1.0, y + g1.1);
            d.push(a - c);
            lv.push(0.5 * (a + c));
        }
    }
    let md = crate::stats::median_mut(&mut d);
    let ml = crate::stats::median_mut(&mut lv);
    if ml > 0.0 {
        md.abs() / ml
    } else {
        f32::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_band_filters() {
        for f in ["haoiii", "HaO3", "Ha-OIII", "L-eXtreme", "ALP-T", "NBZ"] {
            assert_eq!(dual_band(f), Some(["H", "O"]), "{f}");
        }
        for f in ["SIIOIII", "SII-OIII", "S2O3"] {
            assert_eq!(dual_band(f), Some(["S", "O"]), "{f}");
        }
        for f in ["uvir", "UV/IR cut", "L-Pro", "", "Ha", "OIII", "R"] {
            assert_eq!(dual_band(f), None, "{f}");
        }
    }

    fn h(cards: &[(&str, &str)]) -> Header {
        let mut h = Header::default();
        for (k, v) in cards {
            h.push(*k, v.to_string(), "");
        }
        h
    }

    #[test]
    fn parse_patterns_and_offsets() {
        let b = Bayer::parse("RGGB", 0, 0).unwrap();
        assert_eq!(b.name(), "RGGB");
        assert_eq!(b.colour(0, 0), R);
        assert_eq!(b.colour(1, 1), B);
        assert_eq!(b.colour(2, 3), G);
        assert_eq!(Bayer::parse("RGGB", 1, 0).unwrap().name(), "GRBG");
        assert_eq!(Bayer::parse("RGGB", 0, 1).unwrap().name(), "GBRG");
        assert_eq!(Bayer::parse("RGGB", 1, 1).unwrap().name(), "BGGR");
        assert_eq!(Bayer::parse("gbrg", 0, 0).unwrap().name(), "GBRG");
        assert!(Bayer::parse("RGBG", 0, 0).is_some());
        assert!(Bayer::parse("RRGB", 0, 0).is_none());
        assert!(Bayer::parse("CMYG", 0, 0).is_none());
        assert_eq!(from_header(&h(&[("BAYERPAT", "GRBG")])).unwrap_name(), "GRBG");
        assert!(from_header(&h(&[("BAYERPAT", "RGGB"), ("NAXIS3", "3")])).is_none());
        assert!(from_header(&h(&[("FILTER", "L")])).is_none());
    }

    trait UnwrapName {
        fn unwrap_name(self) -> String;
    }
    impl UnwrapName for Option<Bayer> {
        fn unwrap_name(self) -> String {
            self.unwrap().name()
        }
    }

    /// A flat-coloured scene seen through `b`.
    fn mosaic(w: usize, hgt: usize, b: Bayer, rgb: [f32; 3]) -> Image {
        let mut img = Image::new(w, hgt);
        for y in 0..hgt {
            for x in 0..w {
                img.set(x, y, rgb[b.colour(x, y) as usize]);
            }
        }
        img
    }

    #[test]
    fn luminance_has_no_checkerboard() {
        let b = Bayer::parse("RGGB", 0, 0).unwrap();
        let img = mosaic(16, 12, b, [0.2, 0.4, 0.1]);
        let l = luminance(&img);
        let want = 0.25 * 0.2 + 0.5 * 0.4 + 0.25 * 0.1;
        for v in &l.data {
            assert!((v - want).abs() < 1e-6, "{v} vs {want}");
        }
    }

    #[test]
    fn debayer_flat_scene_exact() {
        for pat in ["RGGB", "GRBG", "GBRG", "BGGR"] {
            let b = Bayer::parse(pat, 0, 0).unwrap();
            let img = mosaic(10, 9, b, [0.3, 0.5, 0.15]);
            let [r, g, bl] = debayer(&img, b);
            for i in 0..img.len() {
                assert!((r.data[i] - 0.3).abs() < 1e-6, "{pat} R {}", r.data[i]);
                assert!((g.data[i] - 0.5).abs() < 1e-6, "{pat} G");
                assert!((bl.data[i] - 0.15).abs() < 1e-6, "{pat} B");
            }
            assert!(green_mismatch(&img, b) < 1e-6);
            // Read one pixel out: a "green" slot holds red or blue.
            let wrong = Bayer::parse(pat, 1, 0).unwrap();
            assert!(green_mismatch(&img, wrong) > 0.5, "{pat} shifted");
        }
    }

    #[test]
    fn hot_pixel_found_but_not_a_small_star() {
        let b = Bayer::parse("RGGB", 0, 0).unwrap();
        let (w, hgt) = (64, 64);
        let mut img = mosaic(w, hgt, b, [0.1, 0.12, 0.08]);
        // Deterministic noise.
        let mut s = 12345u32;
        for v in img.data.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v += ((s >> 8) as f32 / 16_777_216.0 - 0.5) * 0.004;
        }
        // A hot pixel.
        img.set(20, 20, 0.6);
        // An undersampled star (FWHM ~2 px) centred on a pixel.
        let (cx, cy) = (40.0f32, 41.0f32);
        let sig = 2.0 / 2.3548;
        for y in 35..48 {
            for x in 34..47 {
                let r2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                let v = img.get(x, y) + 0.5 * (-r2 / (2.0 * sig * sig)).exp();
                img.set(x, y, v);
            }
        }
        let hot = hot_pixels(&img, 8.0);
        assert_eq!(hot, vec![(20, 20)]);
        let m = measure_image(&img);
        assert!(m.get(20, 20) < 0.13, "hot pixel left in the luminance: {}", m.get(20, 20));
        assert!(m.get(40, 41) > 0.2, "star lost");
    }
}
