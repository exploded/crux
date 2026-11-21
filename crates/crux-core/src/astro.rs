//! Low-precision Sun and Moon positions (Montenbruck & Pfleger's MiniSun /
//! MiniMoon, accurate to a few arcminutes), sidereal time and horizon
//! coordinates. Plenty for "why is the sky bright".

use chrono::{NaiveDateTime, Timelike};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

const RAD: f64 = PI / 180.0;
const ARCSEC: f64 = PI / (180.0 * 3600.0);

fn frac(x: f64) -> f64 {
    x - x.floor()
}

/// Julian Day for a UTC time (ΔT ignored: ~70 s doesn't matter here).
pub fn julian_day(t: NaiveDateTime) -> f64 {
    let d = t.date();
    let unix_days = (d - chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days() as f64;
    let secs = t.time().num_seconds_from_midnight() as f64 + t.time().nanosecond() as f64 * 1e-9;
    2440587.5 + unix_days + secs / 86400.0
}

fn centuries(jd: f64) -> f64 {
    (jd - 2451545.0) / 36525.0
}

fn obliquity(t: f64) -> f64 {
    (23.43929111 - 46.8150 / 3600.0 * t) * RAD
}

/// Ecliptic (λ, β) to equatorial (RA, Dec), all radians.
fn ecl_to_eq(lambda: f64, beta: f64, eps: f64) -> (f64, f64) {
    let (sl, cl) = lambda.sin_cos();
    let (sb, cb) = beta.sin_cos();
    let (se, ce) = eps.sin_cos();
    let ra = (sl * ce * cb - sb * se).atan2(cl * cb);
    let dec = (sb * ce + cb * se * sl).asin();
    (ra.rem_euclid(2.0 * PI), dec)
}

/// Geocentric Sun (RA°, Dec°).
pub fn sun_radec(jd: f64) -> (f64, f64) {
    let t = centuries(jd);
    let m = 2.0 * PI * frac(0.993133 + 99.997361 * t);
    let l = 2.0 * PI * frac(0.7859453 + m / (2.0 * PI) + (6893.0 * m.sin() + 72.0 * (2.0 * m).sin() + 6191.2 * t) / 1296.0e3);
    let (ra, dec) = ecl_to_eq(l, 0.0, obliquity(t));
    (ra / RAD, dec / RAD)
}

/// Geocentric Moon (RA°, Dec°, distance km).
pub fn moon_radec(jd: f64) -> (f64, f64, f64) {
    let t = centuries(jd);
    let l0 = frac(0.606433 + 1336.855225 * t);
    let l = 2.0 * PI * frac(0.374897 + 1325.552410 * t);
    let ls = 2.0 * PI * frac(0.993133 + 99.997361 * t);
    let d = 2.0 * PI * frac(0.827361 + 1236.853086 * t);
    let f = 2.0 * PI * frac(0.259086 + 1342.227825 * t);
    let dl = 22640.0 * l.sin() - 4586.0 * (l - 2.0 * d).sin() + 2370.0 * (2.0 * d).sin() + 769.0 * (2.0 * l).sin()
        - 668.0 * ls.sin()
        - 412.0 * (2.0 * f).sin()
        - 212.0 * (2.0 * l - 2.0 * d).sin()
        - 206.0 * (l + ls - 2.0 * d).sin()
        + 192.0 * (l + 2.0 * d).sin()
        - 165.0 * (ls - 2.0 * d).sin()
        - 125.0 * d.sin()
        - 110.0 * (l + ls).sin()
        + 148.0 * (l - ls).sin()
        - 55.0 * (2.0 * f - 2.0 * d).sin();
    let s = f + (dl + 412.0 * (2.0 * f).sin() + 541.0 * ls.sin()) * ARCSEC;
    let h = f - 2.0 * d;
    let n = -526.0 * h.sin() + 44.0 * (l + h).sin() - 31.0 * (-l + h).sin() - 23.0 * (ls + h).sin() + 11.0 * (-ls + h).sin()
        - 25.0 * (-2.0 * l + f).sin()
        + 21.0 * (-l + f).sin();
    let lambda = 2.0 * PI * frac(l0 + dl / 1296.0e3);
    let beta = (18520.0 * s.sin() + n) * ARCSEC;
    let (ra, dec) = ecl_to_eq(lambda, beta, obliquity(t));
    // Main distance terms (Meeus ch. 47), good to a few hundred km.
    let dist = 385000.56 - 20905.355 * l.cos() - 3699.111 * (2.0 * d - l).cos() - 2955.968 * (2.0 * d).cos()
        - 569.925 * (2.0 * l).cos()
        + 48.888 * ls.cos()
        + 246.158 * (2.0 * d - 2.0 * l).cos()
        - 152.138 * (2.0 * d - ls - l).cos()
        - 170.733 * (2.0 * d + l).cos()
        - 204.586 * (2.0 * d - ls).cos();
    (ra / RAD, dec / RAD, dist)
}

/// Greenwich mean sidereal time, degrees.
pub fn gmst_deg(jd: f64) -> f64 {
    let t = centuries(jd);
    (280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t * t - t * t * t / 38710000.0).rem_euclid(360.0)
}

/// Altitude and azimuth (degrees, azimuth from north through east).
pub fn alt_az(ra: f64, dec: f64, lat: f64, lon_east: f64, jd: f64) -> (f64, f64) {
    let ha = (gmst_deg(jd) + lon_east - ra) * RAD;
    let (dec, lat) = (dec * RAD, lat * RAD);
    let alt = (lat.sin() * dec.sin() + lat.cos() * dec.cos() * ha.cos()).asin();
    let az = (-ha.sin() * dec.cos()).atan2(dec.sin() * lat.cos() - dec.cos() * lat.sin() * ha.cos());
    (alt / RAD, (az / RAD).rem_euclid(360.0))
}

/// Great-circle separation, degrees.
pub fn separation(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let (d1, d2) = (dec1 * RAD, dec2 * RAD);
    let c = d1.sin() * d2.sin() + d1.cos() * d2.cos() * ((ra1 - ra2) * RAD).cos();
    c.clamp(-1.0, 1.0).acos() / RAD
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Twilight {
    Day,
    Civil,
    Nautical,
    Astronomical,
    Night,
}

impl Twilight {
    pub fn from_sun_alt(alt: f64) -> Self {
        match alt {
            a if a > -0.833 => Self::Day,
            a if a > -6.0 => Self::Civil,
            a if a > -12.0 => Self::Nautical,
            a if a > -18.0 => Self::Astronomical,
            _ => Self::Night,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Day => "daylight",
            Self::Civil => "civil twilight",
            Self::Nautical => "nautical twilight",
            Self::Astronomical => "astronomical twilight",
            Self::Night => "night",
        }
    }
}

/// Where the Sun and Moon were during an exposure.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SkyContext {
    pub sun_alt: f64,
    pub twilight: Twilight,
    /// Topocentric altitude (parallax applied).
    pub moon_alt: f64,
    /// Illuminated fraction, 0–1.
    pub moon_illum: f64,
    /// Moon–target separation, degrees (None without target coordinates).
    pub moon_sep: Option<f64>,
    /// Target altitude, degrees.
    pub target_alt: Option<f64>,
}

impl SkyContext {
    pub fn at(utc: NaiveDateTime, lat: f64, lon_east: f64, target: Option<(f64, f64)>) -> Self {
        let jd = julian_day(utc);
        let (sra, sdec) = sun_radec(jd);
        let (mra, mdec, mdist) = moon_radec(jd);
        let (sun_alt, _) = alt_az(sra, sdec, lat, lon_east, jd);
        let (malt_geo, _) = alt_az(mra, mdec, lat, lon_east, jd);
        let parallax = (6378.14 / mdist).asin() / RAD;
        let moon_alt = malt_geo - parallax * (malt_geo * RAD).cos();
        // Phase angle ≈ 180° − elongation (Sun distance ≫ Moon distance).
        let elong = separation(sra, sdec, mra, mdec);
        let moon_illum = (1.0 - (elong * RAD).cos()) / 2.0;
        let moon_sep = target.map(|(ra, dec)| separation(ra, dec, mra, mdec));
        let target_alt = target.map(|(ra, dec)| alt_az(ra, dec, lat, lon_east, jd).0);
        SkyContext { sun_alt, twilight: Twilight::from_sun_alt(sun_alt), moon_alt, moon_illum, moon_sep, target_alt }
    }

    /// Light sources that brighten the sky, in plain English.
    pub fn describe(&self) -> Vec<String> {
        let mut v = Vec::new();
        if self.twilight != Twilight::Night {
            v.push(format!("Sun {:.0}° below the horizon ({})", -self.sun_alt, self.twilight.name()));
        }
        if self.moon_alt > 0.0 {
            let sep = self.moon_sep.map(|s| format!(", {s:.0}° away")).unwrap_or_default();
            v.push(format!("Moon {:.0}% lit{sep}, altitude {:.0}°", self.moon_illum * 100.0, self.moon_alt));
        }
        v
    }

    /// True when the Moon or twilight plausibly brightens the sky.
    pub fn bright_sources(&self) -> bool {
        self.twilight != Twilight::Night || (self.moon_alt > 0.0 && self.moon_illum > 0.05)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").unwrap()
    }

    #[test]
    fn julian_day_epoch() {
        assert!((julian_day(t("2000-01-01 12:00:00")) - 2451545.0).abs() < 1e-9);
        assert!((julian_day(t("1987-04-10 00:00:00")) - 2446895.5).abs() < 1e-9);
    }

    #[test]
    fn sidereal_time_meeus_12a() {
        // 1987 April 10, 0h UT: GMST 13h10m46.3668s.
        let g = gmst_deg(julian_day(t("1987-04-10 00:00:00")));
        assert!((g - 197.693195).abs() < 1e-4, "{g}");
    }

    #[test]
    fn sun_meeus_25a() {
        // 1992 October 13, 0h TD: RA 198.38083°, Dec −7.78507°.
        let (ra, dec) = sun_radec(julian_day(t("1992-10-13 00:00:00")));
        assert!((ra - 198.38083).abs() < 0.03, "ra {ra}");
        assert!((dec + 7.78507).abs() < 0.03, "dec {dec}");
    }

    #[test]
    fn moon_meeus_47a_48a() {
        // 1992 April 12, 0h TD: RA 134.688470°, Dec 13.768368°, 368409.7 km; k = 0.6786.
        let jd = julian_day(t("1992-04-12 00:00:00"));
        let (ra, dec, dist) = moon_radec(jd);
        assert!((ra - 134.688470).abs() < 0.15, "ra {ra}");
        assert!((dec - 13.768368).abs() < 0.15, "dec {dec}");
        assert!((dist - 368409.7).abs() < 1500.0, "dist {dist}");
        let (sra, sdec) = sun_radec(jd);
        let k = (1.0 - (separation(sra, sdec, ra, dec) * RAD).cos()) / 2.0;
        assert!((k - 0.6786).abs() < 0.01, "k {k}");
    }

    #[test]
    fn melbourne_evening() {
        // 2026-11-05 21:21 local (10:21 UTC): end of astronomical twilight-ish, Sun well down.
        let c = SkyContext::at(t("2026-11-05 10:21:13"), -37.81, 144.96, Some((87.65, -68.32)));
        assert!(c.sun_alt < -12.0 && c.sun_alt > -25.0, "{}", c.sun_alt);
        // LMC at airmass ~1.95 → altitude ~30.8° (CENTALT in the header).
        assert!((c.target_alt.unwrap() - 30.8).abs() < 0.5, "{:?}", c.target_alt);
    }
}
