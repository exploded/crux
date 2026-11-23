//! Recipes: every Image-stage setting in one readable, versioned TOML file.
//!
//! Saved recipes live in `%APPDATA%\crux\recipes\<name>.toml`; the one a
//! target last used is `<workspace>\<folder>\image\<object>\recipe.toml`;
//! every output image carries the recipe that made it (XISF property
//! `Crux:Recipe`, keyword `CRUX-RCP`).

use super::gradient::GradientSettings;
use super::palette::{self, Finishing, Mix, Stretching};
use super::stars::StarSettings;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Bump when a field changes meaning; older recipes still load (every
/// field has a default).
pub const RECIPE_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Recipe {
    pub name: String,
    pub version: u32,
    pub gradient: GradientSettings,
    pub mix: Mix,
    pub stretch: Stretching,
    pub finish: Finishing,
    /// Star correction / sharpening and noise reduction on the linear
    /// masters, before everything else (`enhance`).
    pub enhance: super::enhance::EnhanceSettings,
    /// Natural star colours (narrowband with OIII; ignored otherwise).
    pub stars: StarSettings,
    /// The part of the picture to keep. Belongs to one target: kept in
    /// its `recipe.toml`, left out of saved recipes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crop: Option<Crop>,
    /// Quarter turns clockwise (0–3), applied with the crop; the crop is
    /// in the turned picture. Belongs to the target like the crop.
    #[serde(skip_serializing_if = "is_zero")]
    pub rotate: u8,
}

fn is_zero(v: &u8) -> bool {
    *v == 0
}

/// A crop as fractions of the full picture (0 = left or top edge, 1 =
/// right or bottom), so it fits the proxy and full size alike.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Crop {
    pub left: f64,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
}

impl Crop {
    pub const WHOLE: Crop = Crop { left: 0.0, top: 0.0, right: 1.0, bottom: 1.0 };

    /// Clamped to the picture, at least 1% across each way.
    pub fn clamped(self) -> Crop {
        let l = self.left.clamp(0.0, 0.99);
        let t = self.top.clamp(0.0, 0.99);
        Crop { left: l, top: t, right: self.right.clamp(l + 0.01, 1.0), bottom: self.bottom.clamp(t + 0.01, 1.0) }
    }

    pub fn is_whole(&self) -> bool {
        self.left <= 1e-6 && self.top <= 1e-6 && self.right >= 1.0 - 1e-6 && self.bottom >= 1.0 - 1e-6
    }

    /// The same box once the picture is turned `q` quarter turns clockwise.
    pub fn turned(self, q: u8) -> Crop {
        (0..q % 4).fold(self, |c, _| Crop { left: 1.0 - c.bottom, top: c.left, right: 1.0 - c.top, bottom: c.right })
    }

    /// Pixel rectangle in a `w` × `h` picture: x, y, width, height.
    pub fn pixels(&self, w: usize, h: usize) -> (usize, usize, usize, usize) {
        let c = self.clamped();
        let x0 = ((c.left * w as f64).round() as usize).min(w.saturating_sub(1));
        let y0 = ((c.top * h as f64).round() as usize).min(h.saturating_sub(1));
        let x1 = ((c.right * w as f64).round() as usize).clamp(x0 + 1, w.max(1));
        let y1 = ((c.bottom * h as f64).round() as usize).clamp(y0 + 1, h.max(1));
        (x0, y0, x1 - x0, y1 - y0)
    }
}

impl Default for Recipe {
    fn default() -> Self {
        Recipe { name: String::new(), version: RECIPE_VERSION, gradient: GradientSettings::default(), mix: Mix::default(), stretch: Stretching::default(), finish: Finishing::default(), enhance: Default::default(), stars: StarSettings::default(), crop: None, rotate: 0 }
    }
}

impl Recipe {
    /// crux's default for these channels: the first preset.
    pub fn for_channels(channels: &[String]) -> Recipe {
        let mut r = Recipe::default();
        let kind = palette::kind_of(channels);
        if let Some(p) = palette::presets(kind, channels).first() {
            r.apply_preset(p, channels);
        }
        r.name = r.mix.preset.clone();
        r
    }

    /// Load a preset's weights (and its stretch and finishing, if it has
    /// any), keeping the rest.
    pub fn apply_preset(&mut self, p: &palette::Preset, channels: &[String]) {
        self.mix = palette::mix_of(p, channels);
        if let Some(s) = &p.stretch {
            self.stretch = s.clone();
        }
        if let Some(f) = &p.finish {
            self.finish = f.clone();
        }
    }

    /// Make the recipe fit these channels: re-derive a named preset's
    /// weights, or fall back to the default mix when the saved weights
    /// don't name these channels.
    pub fn fit_to(&mut self, channels: &[String]) {
        let kind = palette::kind_of(channels);
        let presets = palette::presets(kind, channels);
        if let Some(p) = presets.iter().find(|p| p.name == self.mix.preset) {
            let keep = (self.stretch.clone(), self.finish.clone());
            self.mix = palette::mix_of(p, channels);
            (self.stretch, self.finish) = keep;
            return;
        }
        let covers = channels.iter().all(|c| self.mix.weights.contains_key(c)) && self.mix.weights.keys().all(|k| channels.contains(k));
        if !covers {
            if let Some(p) = presets.first() {
                self.mix = palette::mix_of(p, channels);
            }
        }
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    pub fn from_toml(s: &str) -> Result<Recipe> {
        toml::from_str(s).context("not a crux recipe")
    }

    pub fn load(path: &Path) -> Result<Recipe> {
        let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Recipe::from_toml(&s).with_context(|| path.display().to_string())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let body = format!("# crux recipe (format {}). Every field is optional.\n{}", RECIPE_VERSION, self.to_toml());
        std::fs::write(path, body).with_context(|| format!("write {}", path.display()))
    }

    /// A file-name-safe version of the name.
    pub fn slug(&self) -> String {
        slug(if self.name.is_empty() { "Custom" } else { &self.name })
    }
}

pub fn slug(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_alphanumeric() || " -_&+.".contains(c) { c } else { '-' }).collect();
    s.trim().trim_end_matches('.').to_string()
}

/// Folder of saved recipes.
pub fn dir() -> Option<PathBuf> {
    crate::paths::config_dir().map(|d| d.join("recipes"))
}

/// Saved recipes by name.
pub fn list() -> Vec<(String, PathBuf)> {
    let Some(d) = dir() else { return Vec::new() };
    let mut v: Vec<(String, PathBuf)> = std::fs::read_dir(&d)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "toml"))
                .map(|p| (Recipe::load(&p).map(|r| r.name).ok().filter(|n| !n.is_empty()).unwrap_or_else(|| p.file_stem().unwrap_or_default().to_string_lossy().to_string()), p))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// A saved recipe by name (case-insensitive), or a path to a recipe file.
pub fn find(name_or_path: &str) -> Result<Recipe> {
    let p = Path::new(name_or_path);
    if p.is_file() {
        return Recipe::load(p);
    }
    list()
        .into_iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name_or_path))
        .map(|(_, p)| Recipe::load(&p))
        .unwrap_or_else(|| anyhow::bail!("no recipe named {name_or_path:?} (saved recipes: {})", list().iter().map(|x| x.0.clone()).collect::<Vec<_>>().join(", ")))
}

/// Save under its name in the recipes folder (without the crop and
/// rotation, which belong to one target).
pub fn save_named(r: &Recipe) -> Result<PathBuf> {
    let d = dir().ok_or_else(|| anyhow::anyhow!("no settings folder"))?;
    let p = d.join(format!("{}.toml", r.slug()));
    let mut r = r.clone();
    r.crop = None;
    r.rotate = 0;
    r.save(&p)?;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_fits_other_channels() {
        let sho: Vec<String> = ["H", "O", "S"].iter().map(|s| s.to_string()).collect();
        let mut r = Recipe::for_channels(&sho);
        assert_eq!(r.name, "Copper & Blue");
        r.stretch.background = 0.12;
        let back = Recipe::from_toml(&r.to_toml()).unwrap();
        assert_eq!(back, r);
        // A partial file takes defaults for the rest.
        let part = Recipe::from_toml("name = \"x\"\n[stretch]\nbackground = 0.2\n").unwrap();
        assert_eq!(part.stretch.background, 0.2);
        assert_eq!(part.finish, Finishing::default());
        // Applied to an Ha-only target: the named preset folds into H.
        let mut m = r.clone();
        m.fit_to(&["H".to_string()]);
        assert_eq!(m.mix.weights.len(), 1);
        assert_eq!(m.stretch.background, 0.12);
        // A crop round-trips; recipes without one still load.
        r.crop = Some(Crop { left: 0.1, top: 0.05, right: 0.9, bottom: 1.0 });
        assert_eq!(Recipe::from_toml(&r.to_toml()).unwrap().crop, r.crop);
        assert_eq!(r.crop.unwrap().pixels(1000, 500), (100, 25, 800, 475));
        assert!(Crop::WHOLE.is_whole());
        // Turning: a box at the top left goes to the top right; four turns
        // bring it back.
        let c = Crop { left: 0.0, top: 0.0, right: 0.5, bottom: 0.25 };
        assert_eq!(c.turned(1), Crop { left: 0.75, top: 0.0, right: 1.0, bottom: 0.5 });
        assert_eq!(c.turned(4), c);
        assert_eq!(c.turned(1).turned(3), c);
        r.rotate = 3;
        assert_eq!(Recipe::from_toml(&r.to_toml()).unwrap().rotate, 3);
    }
}
