//! crux-core: the UI-free heart of crux.
//!
//! File readers (FITS, XISF), the shared image type, frame metadata,
//! classification, grouping and calibration matching. Later modules
//! (calibration, registration, stacking) build on the same pieces.

pub mod analysis;
pub mod autopilot;
pub mod astro;
pub mod background;
pub mod calcheck;
pub mod calib;
pub mod cache;
pub mod cachedir;
pub mod catalog;
pub mod cfa;
pub mod cr2;
pub mod collect;
pub mod classify;
pub mod config;
pub mod decisions;
pub mod export;
pub mod fingerprint;
pub mod fits;
pub mod grade;
pub mod group;
pub mod health;
pub mod header;
pub mod image;
pub mod index;
pub mod library;
pub mod meta;
pub mod moves;
pub mod paths;
pub mod pipeline;
pub mod plan;
pub mod process;
pub mod query;
pub mod rcastro;
pub mod render;
pub mod scan;
pub mod solve;
pub mod stars;
pub mod stats;
pub mod stretch;
pub mod synth;
pub mod targets;
pub mod trails;
pub mod transparency;
pub mod watch;
pub mod xisf;

pub use classify::{FrameType, Role};
pub use config::Config;
pub use image::Image;
pub use meta::FrameMeta;
pub use scan::{Frame, ScanResult};

use anyhow::Result;
use std::path::Path;

/// Load a frame for measuring stars, trails and backgrounds: as stored,
/// or for a raw one-shot-colour frame its luminance (`cfa::measure_image`).
pub fn read_for_measuring(meta: &FrameMeta) -> Result<Image> {
    let img = read_image(&meta.path)?;
    Ok(match meta.bayer() {
        Some(_) => cfa::measure_image(&img),
        None => img,
    })
}

/// Load any supported image file.
pub fn read_image(path: &Path) -> Result<Image> {
    match meta::format_of(path) {
        Some(meta::FileFormat::Fits) => fits::read_image(path),
        Some(meta::FileFormat::Xisf) => xisf::read_image(path),
        Some(meta::FileFormat::Cr2) => cr2::read_image(path),
        None => anyhow::bail!("unsupported file type: {}", path.display()),
    }
}
