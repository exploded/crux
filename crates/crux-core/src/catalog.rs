//! Star catalogue: ASTAP's `.1476` databases (D50 and friends, Gaia DR3),
//! read in place from the ASTAP install. Nothing is copied or modified.
//!
//! Format (decoded from the files; matches ASTAP's reader): the sky is cut
//! into 1,476 tiles in 36 declination rings (polar caps 2.57° high, the
//! rest 180/35° each), tile `t` of ring `r` covering RA `(t−1)·360/n` to
//! `t·360/n`. Each file `<db>_RRTT.1476` has a 110-byte text header, then
//! 5-byte records sorted by magnitude: a record `FF FF FF d m` starts a
//! magnitude block (dec high byte `d − 128`, magnitude `(m − 16) / 10`);
//! other records are RA (24 bits, little endian, × 360/(2²⁴−1)) and the low
//! 16 bits of Dec (× 90/(2²³−1) with the block's high byte).

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const TILES: [usize; 36] = [1, 3, 9, 15, 21, 27, 33, 38, 43, 48, 52, 56, 60, 63, 65, 67, 68, 69, 69, 68, 67, 65, 63, 60, 56, 52, 48, 43, 38, 33, 27, 21, 15, 9, 3, 1];
const RING_H: f64 = 180.0 / 35.0;
const CAP: f64 = 90.0 - 87.428_571_428_571_43;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CatStar {
    /// Degrees (J2000, epoch 2025 for D50).
    pub ra: f64,
    pub dec: f64,
    /// BP magnitude.
    pub mag: f32,
}

pub struct Catalog {
    pub dir: PathBuf,
    /// "d50", "d80", "d05", "h18"…
    pub name: String,
    tiles: Mutex<HashMap<(usize, usize), Arc<Vec<CatStar>>>>,
}

impl std::fmt::Debug for Catalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Catalog({} in {})", self.name, self.dir.display())
    }
}

/// Declination band of ring `r` (1-based).
pub fn ring_band(r: usize) -> (f64, f64) {
    match r {
        1 => (-90.0, -90.0 + CAP),
        36 => (90.0 - CAP, 90.0),
        _ => {
            let lo = -90.0 + CAP + (r - 2) as f64 * RING_H;
            (lo, lo + RING_H)
        }
    }
}

/// Decode one tile file.
pub fn decode(bytes: &[u8]) -> Vec<CatStar> {
    let mut out = Vec::with_capacity(bytes.len() / 5);
    let (mut dec9, mut mag) = (0i32, 0f32);
    let mut i = 110;
    while i + 5 <= bytes.len() {
        let b = &bytes[i..i + 5];
        i += 5;
        if b[0] == 0xFF && b[1] == 0xFF && b[2] == 0xFF {
            dec9 = b[3] as i32 - 128;
            mag = (b[4] as f32 - 16.0) / 10.0;
            continue;
        }
        let ra = (b[0] as u32 | (b[1] as u32) << 8 | (b[2] as u32) << 16) as f64 * 360.0 / 16_777_215.0;
        let dec = ((b[3] as i32) | (b[4] as i32) << 8 | dec9 << 16) as f64 * 90.0 / 8_388_607.0;
        out.push(CatStar { ra, dec, mag });
    }
    out
}

/// Angular distance in degrees.
pub fn separation(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let (a1, d1, a2, d2) = (ra1.to_radians(), dec1.to_radians(), ra2.to_radians(), dec2.to_radians());
    let h = ((d2 - d1) / 2.0).sin().powi(2) + d1.cos() * d2.cos() * ((a2 - a1) / 2.0).sin().powi(2);
    (2.0 * h.sqrt().min(1.0).asin()).to_degrees()
}

impl Catalog {
    /// Open the database `name` in `dir`.
    pub fn open(dir: &Path, name: &str) -> Result<Catalog> {
        let probe = dir.join(format!("{name}_0101.1476"));
        anyhow::ensure!(probe.is_file(), "no {name} star database in {}", dir.display());
        Ok(Catalog { dir: dir.to_path_buf(), name: name.to_string(), tiles: Mutex::new(HashMap::new()) })
    }

    /// Find an installed database: `CRUX_CATALOG` (a folder), then ASTAP's
    /// usual install folders. Prefers the deepest (D80, D50, D20, D05, V50, H18).
    pub fn find() -> Option<Catalog> {
        let mut dirs: Vec<PathBuf> = Vec::new();
        if let Some(d) = std::env::var_os("CRUX_CATALOG") {
            dirs.push(PathBuf::from(d));
        }
        if cfg!(windows) {
            dirs.push(PathBuf::from(r"C:\Program Files\astap"));
            dirs.push(PathBuf::from(r"C:\Program Files (x86)\astap"));
        } else {
            dirs.push(PathBuf::from("/opt/astap"));
            dirs.push(PathBuf::from("/usr/share/astap/data"));
            dirs.push(PathBuf::from("/Applications/ASTAP.app/Contents/MacOS"));
        }
        for d in dirs {
            for name in ["d80", "d50", "d20", "d05", "v50", "h18", "h17"] {
                if let Ok(c) = Catalog::open(&d, name) {
                    return Some(c);
                }
            }
        }
        None
    }

    fn tile(&self, ring: usize, t: usize) -> Result<Arc<Vec<CatStar>>> {
        if let Some(v) = self.tiles.lock().unwrap().get(&(ring, t)) {
            return Ok(v.clone());
        }
        let p = self.dir.join(format!("{}_{:02}{:02}.1476", self.name, ring, t));
        let b = std::fs::read(&p).with_context(|| format!("read {}", p.display()))?;
        let v = Arc::new(decode(&b));
        let mut cache = self.tiles.lock().unwrap();
        if cache.len() > 64 {
            cache.clear();
        }
        cache.insert((ring, t), v.clone());
        Ok(v)
    }

    /// Stars within `radius` degrees of (ra, dec), brightest first, at most `max`.
    pub fn cone(&self, ra: f64, dec: f64, radius: f64, max: usize) -> Result<Vec<CatStar>> {
        let (lo, hi) = (dec - radius, dec + radius);
        let mut out: Vec<CatStar> = Vec::new();
        for r in 1..=36 {
            let (blo, bhi) = ring_band(r);
            if bhi < lo || blo > hi {
                continue;
            }
            let n = TILES[r - 1];
            // RA half-width of the cone within this band.
            let worst = blo.max(lo).abs().max(bhi.min(hi).abs());
            let cosd = worst.min(89.9).to_radians().cos();
            let half = if n == 1 || radius / cosd >= 180.0 || hi >= 89.0 || lo <= -89.0 { 360.0 } else { (radius / cosd).min(180.0) };
            let w = 360.0 / n as f64;
            for t in 1..=n {
                if half < 180.0 {
                    let (a0, a1) = ((t - 1) as f64 * w, t as f64 * w);
                    let c = (a0 + a1) / 2.0;
                    let d = ((ra - c + 540.0).rem_euclid(360.0) - 180.0).abs();
                    if d > half + w / 2.0 {
                        continue;
                    }
                }
                for s in self.tile(r, t)?.iter() {
                    if (s.dec - dec).abs() <= radius && separation(ra, dec, s.ra, s.dec) <= radius {
                        out.push(*s);
                    }
                }
            }
        }
        out.sort_by(|a, b| a.mag.total_cmp(&b.mag));
        out.truncate(max);
        Ok(out)
    }
}

/// Gnomonic projection about (ra0, dec0): standard coordinates in degrees
/// (ξ east, η north).
pub fn project(ra0: f64, dec0: f64, ra: f64, dec: f64) -> Option<(f64, f64)> {
    let (a0, d0, a, d) = (ra0.to_radians(), dec0.to_radians(), ra.to_radians(), dec.to_radians());
    let c = d0.sin() * d.sin() + d0.cos() * d.cos() * (a - a0).cos();
    if c <= 0.0 {
        return None;
    }
    let xi = d.cos() * (a - a0).sin() / c;
    let eta = (d0.cos() * d.sin() - d0.sin() * d.cos() * (a - a0).cos()) / c;
    Some((xi.to_degrees(), eta.to_degrees()))
}

/// Inverse of `project`.
pub fn deproject(ra0: f64, dec0: f64, xi: f64, eta: f64) -> (f64, f64) {
    let (a0, d0) = (ra0.to_radians(), dec0.to_radians());
    let (x, y) = (xi.to_radians(), eta.to_radians());
    let rho = (x * x + y * y).sqrt();
    if rho == 0.0 {
        return (ra0, dec0);
    }
    let c = rho.atan();
    let dec = (c.cos() * d0.sin() + y * c.sin() * d0.cos() / rho).asin();
    let ra = a0 + (x * c.sin()).atan2(rho * d0.cos() * c.cos() - y * d0.sin() * c.sin());
    (ra.to_degrees().rem_euclid(360.0), dec.to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rings_cover_the_sky() {
        assert_eq!(TILES.iter().sum::<usize>(), 1476);
        assert!((ring_band(1).0 + 90.0).abs() < 1e-9);
        assert!((ring_band(18).1).abs() < 1e-9);
        assert!((ring_band(19).0).abs() < 1e-9);
        assert!((ring_band(35).1 - (90.0 - CAP)).abs() < 1e-9);
    }

    #[test]
    fn decodes_records() {
        let mut b = vec![b' '; 110];
        // Magnitude block: dec high byte −127 (1 + 128 = 129 → stored 1?), mag 5.5.
        b.extend_from_slice(&[0xFF, 0xFF, 0xFF, 1, 71]);
        b.extend_from_slice(&[0x76, 0x91, 0xE1, 0xF1, 0x7B]);
        let s = decode(&b);
        assert_eq!(s.len(), 1);
        assert!((s[0].mag - 5.5).abs() < 1e-6);
        assert!((s[0].ra - 317.2).abs() < 0.1, "{}", s[0].ra);
        assert!((s[0].dec + 88.96).abs() < 0.01, "{}", s[0].dec);
    }

    #[test]
    fn projection_round_trip() {
        let (xi, eta) = project(275.2, -16.2, 275.6, -15.9).unwrap();
        let (ra, dec) = deproject(275.2, -16.2, xi, eta);
        assert!((ra - 275.6).abs() < 1e-9 && (dec + 15.9).abs() < 1e-9);
    }
}
