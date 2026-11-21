//! Plate solving from the cached star lists, hinted by the header's RA/Dec
//! and pixel scale (FOCALLEN, XPIXSZ). Quads of each star and its three
//! nearest neighbours are matched by their shape (five distance ratios),
//! the quad centres vote for an affine model (RANSAC), and the model is
//! refined on every star. Like ASTAP, but on stars crux already measured.

use crate::catalog::{self, Catalog, CatStar};
use crate::stars::Star;
use serde::{Deserialize, Serialize};

/// Bump when solutions change, so stored ones are solved again (1: the
/// matrix rebuilt about the solved centre, 2026-11-19).
pub const SOLVE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Solution {
    /// Image centre, degrees.
    pub ra: f64,
    pub dec: f64,
    /// Arcseconds per pixel.
    pub scale: f64,
    /// Image rotation in degrees, (−180, 180], the same angle ASTAP and
    /// FITS `CROTA2` give (agrees with ASTAP to 0.01° on James's frames).
    pub rotation: f64,
    /// The image is mirrored relative to the sky (east and west swapped
    /// when north is up).
    pub mirrored: bool,
    /// Pixel (x, y) → standard coordinates (ξ, η) in degrees about (ra, dec):
    /// ξ = cd[0]·(x − cx) + cd[1]·(y − cy), η = cd[2]·(x − cx) + cd[3]·(y − cy).
    pub cd: [f64; 4],
    pub cx: f64,
    pub cy: f64,
    pub matched: usize,
    /// Residual, arcseconds.
    pub rms: f64,
    /// Distance of the solved centre from the mount's reported position
    /// (RA/DEC), arcminutes.
    pub offset_arcmin: Option<f64>,
    /// Distance from the planned target (OBJCTRA/OBJCTDEC), arcminutes.
    #[serde(default)]
    pub target_offset_arcmin: Option<f64>,
    /// `SOLVE_VERSION` when solved (0: before versions).
    #[serde(default)]
    pub version: u32,
}

impl Solution {
    pub fn sky_to_pixel(&self, ra: f64, dec: f64) -> Option<(f64, f64)> {
        let (xi, eta) = catalog::project(self.ra, self.dec, ra, dec)?;
        let c = &self.cd;
        let det = c[0] * c[3] - c[1] * c[2];
        if det == 0.0 {
            return None;
        }
        let dx = (c[3] * xi - c[1] * eta) / det;
        let dy = (-c[2] * xi + c[0] * eta) / det;
        Some((self.cx + dx, self.cy + dy))
    }

    pub fn pixel_to_sky(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = (x - self.cx, y - self.cy);
        catalog::deproject(self.ra, self.dec, self.cd[0] * dx + self.cd[1] * dy, self.cd[2] * dx + self.cd[3] * dy)
    }
}

#[derive(Debug, Clone)]
pub struct Hint {
    pub ra: f64,
    pub dec: f64,
    /// Arcseconds per pixel. A few per cent out is enough to fail a sparse
    /// field: pass hints through `Learned` when solving several frames.
    pub scale: f64,
    /// How far the true centre may be from the hint, degrees.
    pub radius: f64,
    /// The planned target (OBJCTRA/OBJCTDEC), degrees.
    pub target: Option<(f64, f64)>,
}

impl Hint {
    /// From a frame's header: RA/DEC (or OBJCTRA/OBJCTDEC) and FOCALLEN ×
    /// XPIXSZ × binning (or SCALE/PIXSCALE).
    pub fn from_meta(m: &crate::FrameMeta) -> Option<Hint> {
        let h = &m.header;
        let scale = h.f64("SCALE").or_else(|| h.f64("PIXSCALE")).or_else(|| {
            let f = m.focallen?;
            let p = m.xpixsz?;
            Some(206.264_806 * p * m.binning() as f64 / f)
        })?;
        let target = match (h.get("OBJCTRA").and_then(parse_sexagesimal), h.get("OBJCTDEC").and_then(parse_sexagesimal)) {
            (Some(r), Some(d)) => Some((r * 15.0, d)),
            _ => None,
        };
        let (ra, dec) = match (m.ra, m.dec) {
            (Some(r), Some(d)) => (r, d),
            _ => target?,
        };
        Some(Hint { ra, dec, scale, radius: 1.0, target })
    }
}

/// "18 20 47" or "-16:10:12" → degrees (or hours).
pub fn parse_sexagesimal(s: &str) -> Option<f64> {
    let t = s.trim();
    let neg = t.starts_with('-');
    let parts: Vec<f64> = t.trim_start_matches(['-', '+']).split([' ', ':']).filter(|p| !p.is_empty()).map(|p| p.parse::<f64>()).collect::<Result<_, _>>().ok()?;
    let v = parts.first()? + parts.get(1).unwrap_or(&0.0) / 60.0 + parts.get(2).unwrap_or(&0.0) / 3600.0;
    Some(if neg { -v } else { v })
}

#[derive(Clone, Copy)]
struct Quad {
    ratios: [f32; 5],
    size: f32,
    cx: f32,
    cy: f32,
    /// Direction of the longest side, radians in [0, π).
    axis: f32,
}

/// Quads of each star and its three nearest neighbours.
fn quads(pts: &[(f64, f64)]) -> Vec<Quad> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for i in 0..pts.len() {
        let mut d: Vec<(f64, usize)> = (0..pts.len()).filter(|&j| j != i).map(|j| ((pts[j].0 - pts[i].0).powi(2) + (pts[j].1 - pts[i].1).powi(2), j)).collect();
        if d.len() < 3 {
            continue;
        }
        d.select_nth_unstable_by(2, |a, b| a.0.total_cmp(&b.0));
        let mut ids = [i, d[0].1, d[1].1, d[2].1];
        ids.sort_unstable();
        if !seen.insert(ids) {
            continue;
        }
        let p: Vec<(f64, f64)> = ids.iter().map(|&k| pts[k]).collect();
        let mut ds = Vec::with_capacity(6);
        let mut longest = (0.0f64, 0.0f64);
        for a in 0..4 {
            for b in a + 1..4 {
                let d = ((p[a].0 - p[b].0).powi(2) + (p[a].1 - p[b].1).powi(2)).sqrt();
                if d > longest.0 {
                    longest = (d, (p[b].1 - p[a].1).atan2(p[b].0 - p[a].0).rem_euclid(std::f64::consts::PI));
                }
                ds.push(d);
            }
        }
        ds.sort_by(|a, b| b.total_cmp(a));
        if ds[0] <= 0.0 {
            continue;
        }
        let r = [ds[1] / ds[0], ds[2] / ds[0], ds[3] / ds[0], ds[4] / ds[0], ds[5] / ds[0]].map(|v| v as f32);
        out.push(Quad { ratios: r, size: ds[0] as f32, cx: (p.iter().map(|q| q.0).sum::<f64>() / 4.0) as f32, cy: (p.iter().map(|q| q.1).sum::<f64>() / 4.0) as f32, axis: longest.1 as f32 });
    }
    out
}

/// x' = a0 + a1·x + a2·y, y' = b0 + b1·x + b2·y.
#[derive(Debug, Clone, Copy)]
struct Aff([f64; 6]);

impl Aff {
    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let a = &self.0;
        (a[0] + a[1] * x + a[2] * y, a[3] + a[4] * x + a[5] * y)
    }
    fn det(&self) -> f64 {
        self.0[1] * self.0[5] - self.0[2] * self.0[4]
    }
    fn fit(pairs: &[((f64, f64), (f64, f64))]) -> Option<Aff> {
        if pairs.len() < 3 {
            return None;
        }
        let mut m = [[0f64; 3]; 3];
        let (mut vx, mut vy) = ([0f64; 3], [0f64; 3]);
        for &((x, y), (u, v)) in pairs {
            let r = [1.0, x, y];
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] += r[i] * r[j];
                }
                vx[i] += r[i] * u;
                vy[i] += r[i] * v;
            }
        }
        let a = solve3(m, vx)?;
        let b = solve3(m, vy)?;
        Some(Aff([a[0], a[1], a[2], b[0], b[1], b[2]]))
    }
}

fn solve3(m: [[f64; 3]; 3], v: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &[[f64; 3]; 3]| m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let d = det(&m);
    if d.abs() < 1e-12 {
        return None;
    }
    let mut out = [0f64; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let mut mc = m;
        for r in 0..3 {
            mc[r][c] = v[r];
        }
        *o = det(&mc) / d;
    }
    Some(out)
}

/// Pixel scales measured by solving, by the header's nominal scale. Header
/// focal lengths are often a few per cent out (the 2012 AT12IN frames say
/// 1200 mm and solve at about 1,157 mm: 3.7%, up to 75 px at the corners),
/// which dense fields survive and sparse ones (Blue, Green) don't. Once one
/// frame solves, the rest use its scale.
#[derive(Debug, Default, Clone)]
pub struct Learned {
    scales: std::collections::HashMap<i64, f64>,
}

impl Learned {
    fn key(h: &Hint) -> i64 {
        (h.scale * 1e4).round() as i64
    }
    /// Whether a measured scale is known for this hint's nominal scale.
    pub fn knows(&self, h: &Hint) -> bool {
        self.scales.contains_key(&Self::key(h))
    }
    /// The hint with the measured scale, if known.
    pub fn hint(&self, h: &Hint) -> Hint {
        let mut x = h.clone();
        if let Some(&s) = self.scales.get(&Self::key(h)) {
            x.scale = s;
        }
        x
    }
    /// Remember a solution's scale (the first one for each nominal scale).
    pub fn learn(&mut self, h: &Hint, s: &Solution) {
        self.scales.entry(Self::key(h)).or_insert(s.scale);
    }
    /// `solve` with the learned scale, learning from the result.
    pub fn solve(&mut self, cat: &Catalog, stars: &[Star], w: f64, h: f64, hint: &Hint) -> anyhow::Result<Option<Solution>> {
        let r = solve(cat, stars, w, h, &self.hint(hint))?;
        if let Some(s) = &r {
            self.learn(hint, s);
        }
        Ok(r)
    }
}

/// Solve one frame. `stars` brightest first; `w`, `h` the frame size.
pub fn solve(cat: &Catalog, stars: &[Star], w: f64, h: f64, hint: &Hint) -> anyhow::Result<Option<Solution>> {
    if stars.len() < 12 {
        return Ok(None);
    }
    let (cx, cy) = (w / 2.0, h / 2.0);
    let half_diag = 0.5 * (w * w + h * h).sqrt() * hint.scale / 3600.0;
    // Brightest-first lists rarely agree star for star (H-alpha vs BP
    // magnitudes, nebulae, dense fields), so widen step by step.
    let near = hint.radius.min(0.5);
    let tries = [(near, 150, 1.3, false), (near, 150, 1.3, true), (near, 300, 4.0, false), (near, 300, 4.0, true), (hint.radius, 150, 1.3, false), (hint.radius, 300, 4.0, true)];
    for (radius, n_img, density, spread_out) in tries {
        let r = half_diag + radius;
        let n_img = stars.len().min(n_img);
        // Catalogue density matched to the image stars over the search area.
        let area_ratio = (std::f64::consts::PI * r * r) / ((w * hint.scale / 3600.0) * (h * hint.scale / 3600.0));
        let n_cat = ((n_img as f64 * area_ratio * density) as usize).clamp(200, 12000);
        let cs = cat.cone(hint.ra, hint.dec, r, n_cat)?;
        if cs.len() < 20 {
            continue;
        }
        if let Some(s) = solve_with(&cs, stars, n_img, spread_out, false, cx, cy, hint)? {
            // Refit about the solved centre, with the stars around it.
            let h2 = Hint { ra: s.ra, dec: s.dec, scale: s.scale, radius: 0.05, target: None };
            let cs2 = cat.cone(s.ra, s.dec, half_diag * 1.1, 6000)?;
            let mut s2 = refine_all(&cs2, stars, cx, cy, &h2, &s).filter(|x| x.matched >= s.matched).unwrap_or(s);
            s2.offset_arcmin = Some(catalog::separation(hint.ra, hint.dec, s2.ra, s2.dec) * 60.0);
            s2.target_offset_arcmin = hint.target.map(|(r, d)| catalog::separation(r, d, s2.ra, s2.dec) * 60.0);
            return Ok(Some(s2));
        }
    }
    // Last resort: brute-force rotation voting (needs only some shared stars).
    let r = half_diag + near;
    let cs = cat.cone(hint.ra, hint.dec, r, 6000)?;
    if cs.len() >= 20 {
        if let Some(s) = solve_with(&cs, stars, 150, false, true, cx, cy, hint)? {
            let h2 = Hint { ra: s.ra, dec: s.dec, scale: s.scale, radius: 0.05, target: None };
            let cs2 = cat.cone(s.ra, s.dec, half_diag * 1.1, 6000)?;
            let mut s2 = refine_all(&cs2, stars, cx, cy, &h2, &s).filter(|x| x.matched >= s.matched).unwrap_or(s);
            s2.offset_arcmin = Some(catalog::separation(hint.ra, hint.dec, s2.ra, s2.dec) * 60.0);
            s2.target_offset_arcmin = hint.target.map(|(r, d)| catalog::separation(r, d, s2.ra, s2.dec) * 60.0);
            return Ok(Some(s2));
        }
    }
    Ok(None)
}

/// Up to `n` stars spread over a 4×3 grid (brightest first in each cell,
/// taken in turn), so a bright nebula full of knots (30 Doradus) doesn't
/// supply every quad.
fn spread(stars: &[Star], n: usize, w: f64, h: f64) -> Vec<Star> {
    let (gx, gy) = (4usize, 3usize);
    let mut cells: Vec<std::collections::VecDeque<Star>> = vec![Default::default(); gx * gy];
    for s in stars {
        let i = ((s.x as f64 / w * gx as f64) as usize).min(gx - 1);
        let j = ((s.y as f64 / h * gy as f64) as usize).min(gy - 1);
        cells[j * gx + i].push_back(*s);
    }
    let mut out = Vec::with_capacity(n);
    while out.len() < n && cells.iter().any(|c| !c.is_empty()) {
        for c in cells.iter_mut() {
            if let Some(s) = c.pop_front() {
                out.push(s);
                if out.len() == n {
                    break;
                }
            }
        }
    }
    out
}

/// Catalogue → tangent-plane pixels at the hinted scale.
fn cat_pixels(cs: &[CatStar], hint: &Hint) -> Vec<(f64, f64)> {
    let k = 3600.0 / hint.scale;
    cs.iter().filter_map(|s| catalog::project(hint.ra, hint.dec, s.ra, s.dec)).map(|(xi, eta)| (xi * k, eta * k)).collect()
}

fn solve_with(cs: &[CatStar], stars: &[Star], n_img: usize, spread_out: bool, vote: bool, cx: f64, cy: f64, hint: &Hint) -> anyhow::Result<Option<Solution>> {
    let pick: Vec<Star> = if spread_out { spread(stars, n_img, cx * 2.0, cy * 2.0) } else { stars.iter().take(n_img).copied().collect() };
    let img: Vec<(f64, f64)> = pick.iter().map(|s| (s.x as f64 - cx, s.y as f64 - cy)).collect();
    let cp = cat_pixels(cs, hint);
    let qi = quads(&img);
    let mut qc = quads(&cp);
    qc.sort_by(|a, b| a.ratios[0].total_cmp(&b.ratios[0]));
    let tol = 0.008f32;
    // Matched quads: (image quad, catalogue quad).
    let mut matches: Vec<(Quad, Quad)> = Vec::new();
    for q in &qi {
        let lo = qc.partition_point(|c| c.ratios[0] < q.ratios[0] - tol);
        for c in qc[lo..].iter().take_while(|c| c.ratios[0] <= q.ratios[0] + tol) {
            let size = q.size / c.size;
            if (size - 1.0).abs() > 0.08 {
                continue;
            }
            if (1..5).all(|k| (c.ratios[k] - q.ratios[k]).abs() <= tol) {
                matches.push((*q, *c));
            }
        }
    }
    let pairs: Vec<((f64, f64), (f64, f64))> = matches.iter().map(|(q, c)| ((q.cx as f64, q.cy as f64), (c.cx as f64, c.cy as f64))).collect();
    // Hough vote: each match fixes rotation (from the longest side, up to
    // 180°) for either parity, and then the shift.
    let mut votes: std::collections::HashMap<(i8, i32, i32, i32), Vec<usize>> = std::collections::HashMap::new();
    const ROT_BIN: f64 = 2.0;
    const T_BIN: f64 = 30.0;
    for (k, (q, c)) in matches.iter().enumerate() {
        let sc = (c.size / q.size) as f64;
        for parity in [1i8, -1] {
            let base = if parity > 0 { c.axis as f64 - q.axis as f64 } else { c.axis as f64 + q.axis as f64 };
            for flip in [0.0, std::f64::consts::PI] {
                let phi = (base + flip).rem_euclid(2.0 * std::f64::consts::PI);
                let (cs, sn) = (phi.cos(), phi.sin());
                let (x, y) = (q.cx as f64, q.cy as f64 * parity as f64);
                let (tx, ty) = (c.cx as f64 - sc * (cs * x - sn * y), c.cy as f64 - sc * (sn * x + cs * y));
                let key = (parity, (phi.to_degrees() / ROT_BIN).floor() as i32, (tx / T_BIN).floor() as i32, (ty / T_BIN).floor() as i32);
                votes.entry(key).or_default().push(k);
            }
        }
    }
    let mut best_bin: Option<(usize, Vec<usize>)> = None;
    for &(p, r, x, y) in votes.keys() {
        let mut ids: Vec<usize> = Vec::new();
        for dr in -1..=1 {
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let rr = (r + dr).rem_euclid((360.0 / ROT_BIN) as i32);
                    if let Some(v) = votes.get(&(p, rr, x + dx, y + dy)) {
                        ids.extend(v);
                    }
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        if best_bin.as_ref().is_none_or(|b| ids.len() > b.0) {
            best_bin = Some((ids.len(), ids));
        }
    }
    let debug = std::env::var_os("CRUX_SOLVE_DEBUG").is_some();
    if debug {
        eprintln!("solve: {} image stars ({} quads), {} catalogue ({} quads), {} quad matches, best bin {}", img.len(), qi.len(), cp.len(), qc.len(), matches.len(), best_bin.as_ref().map(|b| b.0).unwrap_or(0));
    }
    let mut best: Option<(usize, Aff)> = None;
    if let Some((n, ids)) = best_bin.filter(|b| b.0 >= 4) {
        let sel: Vec<_> = ids.iter().map(|&k| pairs[k]).collect();
        if let Some(m) = Aff::fit(&sel) {
            let inl = pairs.iter().filter(|(p, q)| dist(m.apply(p.0, p.1), *q) < 4.0).count();
            if inl >= 4 && (0.85..=1.15).contains(&m.det().abs()) {
                best = Some((inl.max(n.min(inl)), m));
            }
        }
    }
    // Fallback: RANSAC on triples of quad matches; the scale is known (≈ 1 here).
    if best.is_none() {
        let mut rng = crate::synth::Rng::new(1);
        let n = pairs.len();
        for _ in 0..3000.min(n * n * n) {
            let (a, b, c) = ((rng.uniform() * n as f64) as usize % n, (rng.uniform() * n as f64) as usize % n, (rng.uniform() * n as f64) as usize % n);
            if a == b || b == c || a == c {
                continue;
            }
            let Some(m) = Aff::fit(&[pairs[a], pairs[b], pairs[c]]) else { continue };
            let det = m.det().abs();
            if !(0.85..=1.15).contains(&det) {
                continue;
            }
            let inl = pairs.iter().filter(|(p, q)| dist(m.apply(p.0, p.1), *q) < 4.0).count();
            if best.as_ref().is_none_or(|b| inl > b.0) {
                best = Some((inl, m));
            }
        }
    }
    if let Some((inl, m)) = best.filter(|b| b.0 >= 4) {
        let inliers: Vec<_> = pairs.iter().copied().filter(|(p, q)| dist(m.apply(p.0, p.1), *q) < 4.0).collect();
        let m = Aff::fit(&inliers).unwrap_or(m);
        if debug {
            eprintln!("  quads: {inl} inliers");
        }
        if let Some(sol) = finish(&cp, stars, cx, cy, hint, m) {
            return Ok(Some(sol));
        }
    }
    if !vote {
        return Ok(None);
    }
    // Fallback: try every rotation and both parities, voting on the shift.
    for m in vote_models(&cp, &pick, cx, cy) {
        if let Some(sol) = finish(&cp, stars, cx, cy, hint, m) {
            if debug {
                eprintln!("  solved by rotation voting");
            }
            return Ok(Some(sol));
        }
    }
    Ok(None)
}

/// Candidate models from brute-force rotation (0.5° steps) and parity,
/// with a shift histogram of every image–catalogue pair. Best first.
fn vote_models(cp: &[(f64, f64)], pick: &[Star], cx: f64, cy: f64) -> Vec<Aff> {
    let img: Vec<(f64, f64)> = pick.iter().take(80).map(|s| (s.x as f64 - cx, s.y as f64 - cy)).collect();
    // Catalogue stars near the image area only, brightest first.
    let r = (cx * cx + cy * cy).sqrt() * 1.6;
    let cat: Vec<(f64, f64)> = cp.iter().copied().filter(|p| (p.0 * p.0 + p.1 * p.1).sqrt() < r).take(600).collect();
    if img.len() < 12 || cat.len() < 12 {
        return Vec::new();
    }
    const BIN: f64 = 20.0;
    // Shifts fall within ±(catalogue radius + image radius).
    let span = r + (cx * cx + cy * cy).sqrt();
    let nb = (2.0 * span / BIN).ceil() as usize + 2;
    let mut grid = vec![0u16; nb * nb];
    let mut touched: Vec<u32> = Vec::with_capacity(img.len() * cat.len());
    let mut peaks: Vec<(u32, i8, f64, (f64, f64))> = Vec::new();
    for parity in [1i8, -1] {
        for step in 0..720 {
            let phi = (step as f64 * 0.5).to_radians();
            let (c, s) = (phi.cos(), phi.sin());
            for &(x, y) in &img {
                let y = y * parity as f64;
                let (u, v) = (c * x - s * y, s * x + c * y);
                for &(a, b) in &cat {
                    let (i, j) = (((a - u + span) / BIN) as usize, ((b - v + span) / BIN) as usize);
                    if i + 1 < nb && j + 1 < nb {
                        let k = j * nb + i;
                        if grid[k] == 0 {
                            touched.push(k as u32);
                        }
                        grid[k] = grid[k].saturating_add(1);
                    }
                }
            }
            let mut best = (0u32, 0usize);
            for &k in &touched {
                let k = k as usize;
                let sum = grid[k] as u32 + grid[k + 1] as u32 + grid[k + nb] as u32 + grid[k + nb + 1] as u32;
                if sum > best.0 {
                    best = (sum, k);
                }
            }
            for &k in &touched {
                grid[k as usize] = 0;
            }
            touched.clear();
            let (i, j) = (best.1 % nb, best.1 / nb);
            peaks.push((best.0, parity, phi, ((i as f64 + 1.0) * BIN - span, (j as f64 + 1.0) * BIN - span)));
        }
    }
    peaks.sort_by(|a, b| b.0.cmp(&a.0));
    peaks
        .into_iter()
        .take(4)
        .filter(|p| p.0 >= 8)
        .map(|(_, parity, phi, (tx, ty))| {
            let (c, s) = (phi.cos(), phi.sin());
            let p = parity as f64;
            Aff([tx, c, -s * p, ty, s, c * p])
        })
        .collect()
}

/// Catalogue plane positions on a grid, for nearest-neighbour lookups.
struct CatGrid<'a> {
    cp: &'a [(f64, f64)],
    cell: f64,
    map: std::collections::HashMap<(i64, i64), Vec<usize>>,
}

impl<'a> CatGrid<'a> {
    fn new(cp: &'a [(f64, f64)], cell: f64) -> Self {
        let mut map: std::collections::HashMap<(i64, i64), Vec<usize>> = Default::default();
        for (i, c) in cp.iter().enumerate() {
            map.entry(((c.0 / cell).floor() as i64, (c.1 / cell).floor() as i64)).or_default().push(i);
        }
        CatGrid { cp, cell, map }
    }
    fn nearest(&self, q: (f64, f64), tol: f64) -> Option<(f64, (f64, f64))> {
        let (gx, gy) = ((q.0 / self.cell).floor() as i64, (q.1 / self.cell).floor() as i64);
        let k = (tol / self.cell).ceil() as i64;
        let mut best: Option<(f64, (f64, f64))> = None;
        for y in gy - k..=gy + k {
            for x in gx - k..=gx + k {
                for &i in self.map.get(&(x, y)).map(|v| v.as_slice()).unwrap_or(&[]) {
                    let d = dist(q, self.cp[i]);
                    if d < tol && best.is_none_or(|b| d < b.0) {
                        best = Some((d, self.cp[i]));
                    }
                }
            }
        }
        best
    }
}

/// Refine on all stars by nearest neighbours and build the solution.
fn finish(cp: &[(f64, f64)], stars: &[Star], cx: f64, cy: f64, hint: &Hint, mut m: Aff) -> Option<Solution> {
    let img: Vec<(f64, f64)> = stars.iter().map(|s| (s.x as f64 - cx, s.y as f64 - cy)).collect();
    let grid = CatGrid::new(cp, 12.0);
    let debug = std::env::var_os("CRUX_SOLVE_DEBUG").is_some();
    let mut matched = 0;
    let mut rms = f64::MAX;
    for tol in [12.0, 6.0, 3.0, 2.0, 1.5] {
        let pairs: Vec<((f64, f64), (f64, f64))> = img.iter().filter_map(|&p| grid.nearest(m.apply(p.0, p.1), tol).map(|(_, c)| (p, c))).collect();
        if pairs.len() < 10 {
            if debug {
                eprintln!("  finish: only {} pairs within {tol} px", pairs.len());
            }
            return None;
        }
        m = Aff::fit(&pairs)?;
        let r: Vec<f64> = pairs.iter().map(|(p, c)| dist(m.apply(p.0, p.1), *c)).collect();
        rms = (r.iter().map(|v| v * v).sum::<f64>() / r.len() as f64).sqrt();
        matched = pairs.len();
    }
    if debug {
        eprintln!("  finish: {matched} matched, rms {rms:.2} px");
    }
    if matched < 10 || rms > 2.0 {
        return None;
    }
    // The model maps image pixels into the tangent plane about the hint.
    // Rebuild it about the solved centre: tangent planes at different
    // points are turned by about Δα·sin δ, so using the hint plane's matrix
    // as it stands put the rotation 0.6° out on LMC frames whose mount
    // position was 20′ off (Dec −70°).
    let k = hint.scale / 3600.0;
    let sky = |x: f64, y: f64| {
        let (u, v) = m.apply(x, y);
        catalog::deproject(hint.ra, hint.dec, u * k, v * k)
    };
    let (ra, dec) = sky(0.0, 0.0);
    let h = 0.25 * cx.min(cy).max(8.0);
    let std_at = |x: f64, y: f64| {
        let (r, d) = sky(x, y);
        catalog::project(ra, dec, r, d)
    };
    let (xp, xm, yp, ym) = (std_at(h, 0.0)?, std_at(-h, 0.0)?, std_at(0.0, h)?, std_at(0.0, -h)?);
    let cd = [(xp.0 - xm.0) / (2.0 * h), (yp.0 - ym.0) / (2.0 * h), (xp.1 - xm.1) / (2.0 * h), (yp.1 - ym.1) / (2.0 * h)];
    let det = cd[0] * cd[3] - cd[1] * cd[2];
    let scale = det.abs().sqrt() * 3600.0;
    let rotation = rotation_of(&cd);
    Some(Solution { ra, dec, scale, rotation, mirrored: det > 0.0, cd, cx, cy, matched, rms: rms * hint.scale, offset_arcmin: None, target_offset_arcmin: None, version: SOLVE_VERSION })
}

fn refine_all(cs: &[CatStar], stars: &[Star], cx: f64, cy: f64, hint: &Hint, s: &Solution) -> Option<Solution> {
    let cp = cat_pixels(cs, hint);
    let k = 3600.0 / hint.scale;
    // The previous solution's linear part in plane pixels; centre at 0.
    let m = Aff([0.0, s.cd[0] * k, s.cd[1] * k, 0.0, s.cd[2] * k, s.cd[3] * k]);
    finish(&cp, stars, cx, cy, hint, m)
}

/// Image rotation (ASTAP's / CROTA2's convention) from a CD matrix.
pub fn rotation_of(cd: &[f64; 4]) -> f64 {
    // Up = towards row 0 = (0, −1) in pixels.
    let (ue, un) = (-cd[1], -cd[3]);
    let pa_up = ue.atan2(un).to_degrees();
    let mut rotation = (180.0 - pa_up).rem_euclid(360.0);
    if rotation > 180.0 {
        rotation -= 360.0;
    }
    rotation
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic field from the installed catalogue: project real stars
    /// through a known rotation and scale, then solve. Skipped without ASTAP.
    #[test]
    fn solves_a_synthetic_field() {
        let Some(cat) = Catalog::find() else { return };
        let (w, h, scale) = (3000.0, 2000.0, 1.2);
        let (ra0, dec0, rot) = (83.82, -5.39, 37.0f64);
        let cs = cat.cone(ra0, dec0, 0.7, 400).unwrap();
        let (c, s) = (rot.to_radians().cos(), rot.to_radians().sin());
        let stars: Vec<Star> = cs
            .iter()
            .filter_map(|st| {
                let (xi, eta) = catalog::project(ra0, dec0, st.ra, st.dec)?;
                let (u, v) = (xi * 3600.0 / scale, eta * 3600.0 / scale);
                let (x, y) = (w / 2.0 + c * u - s * v, h / 2.0 + s * u + c * v);
                ((10.0..w - 10.0).contains(&x) && (10.0..h - 10.0).contains(&y)).then(|| Star { x: x as f32, y: y as f32, flux: 30.0 - st.mag, peak: 1.0, fwhm: 3.0, hfr: 2.0, ecc: 0.1, pa: 0.0, snr: 50.0, area: 20, flags: 0 })
            })
            .take(300)
            .collect();
        let hint = Hint { ra: ra0 + 0.1, dec: dec0 - 0.1, scale: scale * 1.01, radius: 1.0, target: None };
        let sol = solve(&cat, &stars, w, h, &hint).unwrap().expect("solved");
        assert!(catalog::separation(sol.ra, sol.dec, ra0, dec0) * 3600.0 < 1.0, "{} {}", sol.ra, sol.dec);
        assert!((sol.scale - scale).abs() < 0.001, "{}", sol.scale);
        assert!(sol.rms < 0.1, "{}", sol.rms);
    }

    #[test]
    fn sexagesimal() {
        assert!((parse_sexagesimal("18 20 47").unwrap() - 18.346_388).abs() < 1e-5);
        assert!((parse_sexagesimal("-16:10:12").unwrap() + 16.17).abs() < 1e-6);
    }

    #[test]
    fn quads_are_shape_only() {
        let pts = vec![(0.0, 0.0), (10.0, 0.0), (0.0, 7.0), (3.0, 4.0), (50.0, 50.0), (60.0, 55.0), (52.0, 61.0), (58.0, 48.0)];
        let rot: Vec<(f64, f64)> = pts.iter().map(|&(x, y)| (-y * 1.3 + 5.0, x * 1.3 - 2.0)).collect();
        let (a, b) = (quads(&pts), quads(&rot));
        assert_eq!(a.len(), b.len());
        for (p, q) in a.iter().zip(&b) {
            for k in 0..5 {
                assert!((p.ratios[k] - q.ratios[k]).abs() < 1e-5);
            }
            assert!((q.size / p.size - 1.3).abs() < 1e-5);
        }
    }
}
