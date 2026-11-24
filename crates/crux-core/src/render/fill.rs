//! Channels that cover less of the frame than the others. Frames of one
//! filter taken at another framing (NGC 3372's HaOIII frames are turned
//! about 40° from its RGB frames) leave that channel empty over part of
//! the picture, so the mix there has no H or O. Without filling, the
//! palette turns those parts another colour with a sharp, straight edge,
//! and the trimmed crop keeps only the rectangle inside the turned frame.
//!
//! Where a channel has no data but its stand-ins do (H or S from R, O from
//! G and B, L from R, G and B, R from H or S, G and B from O), it takes a
//! linear fit of the stand-in: `a · stand-in + b`, fitted on 16 px block
//! medians where both have data, rejecting outliers. The fill is feathered
//! into the channel's own data over [`FEATHER_SHARE`] of the long side, so
//! no edge shows. Pixels where neither has data stay empty.

use crate::image::Image;
use crate::process::combine::MIN_DATA;
use rayon::prelude::*;

/// The feather, as a share of the picture's long side.
const FEATHER_SHARE: f32 = 1.0 / 40.0;
/// Fill only when at least this share of the picture can be filled.
const MIN_MISSING: f64 = 0.01;
/// Blocks for the fit, and how far from the channel's edge they must be
/// (resampling softens the last few pixels).
const BLOCK: usize = 16;
const EDGE: f32 = 8.0;

/// How to fill one channel.
pub struct Fill {
    /// The stand-in channels (indices); their mean is the stand-in.
    pub donors: Vec<usize>,
    pub a: f32,
    pub b: f32,
    /// Per pixel: 1 keeps the channel's own value, 0 takes the fill.
    weight: Vec<f32>,
}

/// The channels that can stand in for `channel`, among `names`.
fn stand_ins(channel: &str, names: &[String]) -> Vec<usize> {
    let want: &[&str] = match channel {
        "H" | "S" => &["R"],
        "O" => &["G", "B"],
        "L" => &["R", "G", "B"],
        "R" => &["H", "S"],
        "G" | "B" => &["O"],
        _ => &[],
    };
    let found: Vec<usize> = want.iter().filter_map(|w| names.iter().position(|n| n == w)).collect();
    // R takes one of H and S, H first.
    if channel == "R" {
        found.into_iter().take(1).collect()
    } else {
        found
    }
}

/// The stand-in value at pixel `i`: the donors' mean, or None when any of
/// them has no data there.
fn stand_in(donors: &[&Image], i: usize) -> Option<f32> {
    let mut s = 0.0;
    for d in donors {
        let v = d.data[i];
        if v <= 0.0 || !v.is_finite() {
            return None;
        }
        s += v;
    }
    Some(s / donors.len() as f32)
}

/// Which channels need filling, and how. `None` for a channel that covers
/// as much as its stand-ins (or has none, or the fit fails).
pub fn plan(images: &[&Image], names: &[String]) -> Vec<Option<Fill>> {
    (0..images.len())
        .map(|k| {
            let donors = stand_ins(&names[k], names);
            if donors.is_empty() || donors.contains(&k) {
                return None;
            }
            let ds: Vec<&Image> = donors.iter().map(|&d| images[d]).collect();
            if ds.iter().any(|d| (d.width, d.height) != (images[k].width, images[k].height)) {
                return None;
            }
            plan_one(images[k], &ds).map(|(a, b, weight)| Fill { donors, a, b, weight })
        })
        .collect()
}

fn plan_one(t: &Image, ds: &[&Image]) -> Option<(f32, f32, Vec<f32>)> {
    let (w, h) = (t.width, t.height);
    let n = w * h;
    let gap: Vec<bool> = (0..n).into_par_iter().map(|i| t.data[i] <= 0.0 && stand_in(ds, i).is_some()).collect();
    let missing = gap.iter().filter(|g| **g).count();
    if (missing as f64) < MIN_MISSING * n as f64 {
        return None;
    }
    let dist = distance_to(&gap, w, h);
    // Block medians where both have data, away from the channel's edge.
    let (bw, bh) = (w / BLOCK, h / BLOCK);
    let pairs: Vec<(f32, f32)> = (0..bw * bh)
        .into_par_iter()
        .filter_map(|k| {
            let (bx, by) = (k % bw, k / bw);
            let (mut xs, mut ys) = (Vec::with_capacity(BLOCK * BLOCK), Vec::with_capacity(BLOCK * BLOCK));
            for y in by * BLOCK..(by + 1) * BLOCK {
                for x in bx * BLOCK..(bx + 1) * BLOCK {
                    let i = y * w + x;
                    let v = t.data[i];
                    if v > 0.0 && dist[i] > EDGE {
                        if let Some(s) = stand_in(ds, i) {
                            xs.push(s);
                            ys.push(v);
                        }
                    }
                }
            }
            (xs.len() >= BLOCK * BLOCK / 2).then(|| (crate::stats::median_mut(&mut xs), crate::stats::median_mut(&mut ys)))
        })
        .collect();
    let (a, b) = robust_line(&pairs)?;
    let feather = (w.max(h) as f32 * FEATHER_SHARE).max(8.0);
    let weight: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            if gap[i] {
                0.0
            } else if t.data[i] > 0.0 && dist[i].is_finite() {
                smoothstep(dist[i] / feather)
            } else {
                1.0
            }
        })
        .collect();
    Some((a, b, weight))
}

/// `y = a·x + b` by least squares, refitted five times without points
/// more than 3σ (MAD) off the line. None with too few points or a slope
/// that isn't positive.
fn robust_line(pairs: &[(f32, f32)]) -> Option<(f32, f32)> {
    let mut keep: Vec<(f64, f64)> = pairs.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    let mut line = None;
    for _ in 0..6 {
        if keep.len() < 50 {
            return None;
        }
        let m = keep.len() as f64;
        let (mx, my) = (keep.iter().map(|p| p.0).sum::<f64>() / m, keep.iter().map(|p| p.1).sum::<f64>() / m);
        let sxx: f64 = keep.iter().map(|p| (p.0 - mx).powi(2)).sum();
        let sxy: f64 = keep.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        if sxx <= 0.0 {
            return None;
        }
        let a = sxy / sxx;
        let b = my - a * mx;
        line = Some((a, b));
        let mut res: Vec<f64> = pairs.iter().map(|&(x, y)| (y as f64 - a * x as f64 - b).abs()).collect();
        res.sort_by(f64::total_cmp);
        let sigma = 1.4826 * res[res.len() / 2];
        if sigma <= 0.0 {
            break;
        }
        keep = pairs.iter().map(|&(x, y)| (x as f64, y as f64)).filter(|p| (p.1 - a * p.0 - b).abs() <= 3.0 * sigma).collect();
    }
    let (a, b) = line?;
    (a > 0.0 && a.is_finite() && b.is_finite()).then_some((a as f32, b as f32))
}

fn smoothstep(x: f32) -> f32 {
    let s = x.clamp(0.0, 1.0);
    s * s * (3.0 - 2.0 * s)
}

/// The channel filled: its own data where the weight is 1, the fit where
/// 0, blended between. `donors` are the fill's stand-in images (the same
/// kind as `t`: masters, or starless masters).
pub fn apply(f: &Fill, t: &Image, donors: &[&Image]) -> Image {
    let data: Vec<f32> = t
        .data
        .par_iter()
        .enumerate()
        .map(|(i, &v)| {
            let wgt = f.weight[i];
            if wgt >= 1.0 {
                return v;
            }
            match stand_in(donors, i) {
                Some(s) => {
                    let fill = f.a * s + f.b;
                    let own = if v > 0.0 { v } else { fill };
                    (wgt * own + (1.0 - wgt) * fill).clamp(MIN_DATA, 1.0)
                }
                None => v,
            }
        })
        .collect();
    Image { width: t.width, height: t.height, data, adu_scale: t.adu_scale }
}

/// Chamfer distance (1, √2) to the nearest `source` pixel; infinite with
/// none. The picture's edge doesn't count.
fn distance_to(source: &[bool], w: usize, h: usize) -> Vec<f32> {
    const D: f32 = std::f32::consts::SQRT_2;
    let mut d: Vec<f32> = source.iter().map(|&s| if s { 0.0 } else { f32::INFINITY }).collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if d[i] == 0.0 {
                continue;
            }
            let mut m = d[i];
            if x > 0 {
                m = m.min(d[i - 1] + 1.0);
            }
            if y > 0 {
                m = m.min(d[i - w] + 1.0);
                if x > 0 {
                    m = m.min(d[i - w - 1] + D);
                }
                if x + 1 < w {
                    m = m.min(d[i - w + 1] + D);
                }
            }
            d[i] = m;
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            if d[i] == 0.0 {
                continue;
            }
            let mut m = d[i];
            if x + 1 < w {
                m = m.min(d[i + 1] + 1.0);
            }
            if y + 1 < h {
                m = m.min(d[i + w] + 1.0);
                if x + 1 < w {
                    m = m.min(d[i + w + 1] + D);
                }
                if x > 0 {
                    m = m.min(d[i + w - 1] + D);
                }
            }
            d[i] = m;
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// R everywhere; H = 2R + 0.01 on the left half only.
    fn pair() -> (Image, Image) {
        let (w, h) = (400, 240);
        let r: Vec<f32> = (0..w * h).map(|i| 0.05 + 0.1 * ((i % w) as f32 / w as f32) + 0.02 * (((i * 7919) % 101) as f32 / 101.0)).collect();
        let hh: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 2.0 * r[i] + 0.01 } else { 0.0 }).collect();
        (Image::from_data(w, h, hh), Image::from_data(w, h, r))
    }

    #[test]
    fn a_half_covered_channel_is_filled_from_its_stand_in() {
        let (hh, r) = pair();
        let fills = plan(&[&hh, &r], &names(&["H", "R"]));
        assert!(fills[1].is_none(), "R covers everything");
        let f = fills[0].as_ref().expect("H is filled");
        assert!((f.a - 2.0).abs() < 1e-3 && (f.b - 0.01).abs() < 1e-3, "a {} b {}", f.a, f.b);
        let out = apply(f, &hh, &[&r]);
        // The relation holds exactly here, so the fill and the feather
        // reproduce it everywhere.
        for i in 0..out.data.len() {
            assert!((out.data[i] - (2.0 * r.data[i] + 0.01)).abs() < 1e-4, "pixel {i}");
        }
    }

    #[test]
    fn full_coverage_and_no_stand_in_leave_channels_alone() {
        let (_, r) = pair();
        let full = Image::from_data(r.width, r.height, r.data.iter().map(|v| v * 3.0).collect());
        assert!(plan(&[&full, &r], &names(&["H", "R"]))[0].is_none());
        let (hh, _) = pair();
        assert!(plan(&[&hh, &full], &names(&["H", "O"]))[0].is_none(), "O doesn't stand in for H");
    }

    #[test]
    fn empty_where_the_stand_in_is_empty_too() {
        let (hh, mut r) = pair();
        let w = r.width;
        for (i, v) in r.data.iter_mut().enumerate() {
            if i % w >= w - 10 {
                *v = 0.0;
            }
        }
        let fills = plan(&[&hh, &r], &names(&["H", "R"]));
        let out = apply(fills[0].as_ref().unwrap(), &hh, &[&r]);
        assert!(out.data.iter().enumerate().all(|(i, v)| (i % w >= w - 10) == (*v == 0.0)));
    }
}
