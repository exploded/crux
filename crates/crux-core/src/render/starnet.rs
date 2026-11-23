//! The star remover network for natural star colours: each master's
//! starless version, made once and kept
//! (`<image dir>\starless\<channel>_<master hash>_<model>.xisf`). `prepare`
//! uses it in place of the classic push-pull fill, after the same gradient
//! correction. The star discs (for measuring each star's light) still come
//! from crux's own detection.

use super::Target;
use anyhow::Result;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::atomic::AtomicBool;

/// Can the network run here? `Ok` or why not.
pub fn availability() -> std::result::Result<(), String> {
    #[cfg(feature = "ml")]
    {
        if super::net::builtin(super::net::Kind::Starless).is_some() {
            Ok(())
        } else {
            Err("crux's star remover isn't in this build yet".into())
        }
    }
    #[cfg(not(feature = "ml"))]
    {
        Err("this build of crux has no networks (built without the ml feature)".into())
    }
}

#[cfg_attr(not(feature = "ml"), allow(dead_code))]
fn hex(s: &str) -> String {
    Sha256::digest(s.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Fill in every master's `starless` (made now or found in the cache).
/// `progress(master, masters, fraction)`.
pub fn apply(t: &Target, out: &Path, progress: &(dyn Fn(usize, usize, f32) + Sync), cancel: &AtomicBool) -> Result<Target> {
    #[cfg(feature = "ml")]
    {
        use super::net::{builtin, Kind, Network, Settings};
        use anyhow::Context;
        let (_, _, tag) = builtin(Kind::Starless).ok_or_else(|| anyhow::anyhow!("crux's star remover isn't in this build yet"))?;
        let _one = super::enhance::LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = t.image_dir(out).join("starless");
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        let mut res = t.clone();
        let n = t.masters.len();
        let mut keep = Vec::new();
        for (i, m) in t.masters.iter().enumerate() {
            let (size, mtime) = std::fs::metadata(&m.path)
                .map(|md| (md.len(), md.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos()).unwrap_or(0)))
                .unwrap_or((0, 0));
            let mh = hex(&format!("{}:{size}:{mtime}", crate::paths::key(&m.path)));
            let file = dir.join(format!("{}_{mh}_{tag}.xisf", crate::process::job::safe_name(&m.channel)));
            if file.is_file() {
                super::enhance::touch(&file);
            } else {
                let laps = crate::timing::Laps::new(format!("starless {}", m.channel));
                let img = crate::read_image(&m.path)?;
                laps.lap("read");
                let (noise, sky, fwhm) = super::net::measure(&img);
                laps.lap(&format!("noise, sky and FWHM ({fwhm:.2} px)"));
                let net = Network::shared(Kind::Starless)?;
                let sl = net.run(&img, noise, sky, fwhm, &[], &Settings::default(), &|k, nn| progress(i, n, k as f32 / nn.max(1) as f32), cancel)?;
                let tmp = super::enhance::tmp_path(&file);
                crate::xisf::write_image(&tmp, &sl, &crate::header::Header::default(), None)?;
                std::fs::rename(&tmp, &file)?;
                laps.lap("network and write");
            }
            keep.push((crate::process::job::safe_name(&m.channel), file.clone()));
            res.masters[i].starless = Some(file);
        }
        // Per channel, the most recently used few (the masters of other
        // enhance settings, so switching back finds them); everything else
        // (other channels, an older network, leftovers) goes.
        if let Ok(rd) = std::fs::read_dir(&dir) {
            let mut per: Vec<Vec<std::path::PathBuf>> = vec![Vec::new(); keep.len()];
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                // `<channel>_<16 hex>_<tag>.xisf` for this channel exactly.
                let slot = keep.iter().position(|(ch, _)| {
                    name.strip_prefix(&format!("{ch}_"))
                        .and_then(|r| r.strip_suffix(&format!("_{tag}.xisf")))
                        .is_some_and(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()))
                });
                match slot {
                    Some(i) => per[i].push(e.path()),
                    None => {
                        let _ = std::fs::remove_file(e.path());
                    }
                }
            }
            for (files, (_, current)) in per.into_iter().zip(&keep) {
                for p in super::enhance::oldest_beyond(files, current, super::enhance::KEEP_RESULTS) {
                    let _ = std::fs::remove_file(p);
                }
            }
        }
        Ok(res)
    }
    #[cfg(not(feature = "ml"))]
    {
        let _ = (t, out, progress, cancel);
        anyhow::bail!("this build of crux has no networks (built without the ml feature)")
    }
}

#[cfg(all(test, feature = "ml"))]
mod tests {
    use super::*;
    use crate::header::Header;
    use crate::image::Image;

    #[test]
    fn starless_masters_are_made_once_and_kept() {
        let dir = std::env::temp_dir().join(format!("crux-starnet-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (w, h) = (160usize, 120usize);
        let mut data = vec![0.02f32; w * h];
        // One star.
        for y in 0..h {
            for x in 0..w {
                let d2 = (x as f32 - 80.0).powi(2) + (y as f32 - 60.0).powi(2);
                let n = (((x * 7919 + y * 104729) % 1000) as f32 / 1000.0 - 0.5) * 0.001;
                data[y * w + x] += 0.3 * (-d2 / (2.0 * 1.6f32.powi(2))).exp() + n;
            }
        }
        let master = dir.join("m.xisf");
        crate::xisf::write_image(&master, &Image::from_data(w, h, data), &Header::default(), None).unwrap();
        let t = Target { object: "Test".into(), masters: vec![super::super::Master { channel: "H".into(), filter: "H".into(), path: master, frames: 3, exposure: 900.0, starless: None, noise: 0.0 }] };
        let a = apply(&t, &dir, &|_, _, _| {}, &AtomicBool::new(false)).unwrap();
        let f = a.masters[0].starless.clone().unwrap();
        let made = std::fs::metadata(&f).unwrap().modified().unwrap();
        let sl = crate::read_image(&f).unwrap();
        // The star is (mostly) gone.
        assert!(sl.get(80, 60) < 0.02 + 0.3 * 0.2, "peak left {}", sl.get(80, 60));
        let b = apply(&t, &dir, &|_, _, _| {}, &AtomicBool::new(false)).unwrap();
        assert_eq!(b.masters[0].starless.as_ref(), Some(&f));
        assert_eq!(std::fs::metadata(&f).unwrap().modified().unwrap(), made);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
