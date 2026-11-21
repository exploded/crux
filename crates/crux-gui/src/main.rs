//! crux desktop app: eframe/egui on wgpu.
//!
//! `crux-gui [FOLDER] [--cache-dir DIR] [--library DIR]`. Without a folder
//! it reopens the last one. The cache and your decisions go to
//! `<folder>\.crux` (like the CLI) unless `--cache-dir` says otherwise.

#![windows_subsystem = "windows"]

mod app;
mod auto;
mod calibrate;
mod chart;
mod frames;
mod home;
mod image;
mod inbox;
mod libview;
mod mosaic;
mod organise;
mod session;
mod settings;
mod stack;
mod theme;
mod viewer;
mod widgets;
mod worker;

use std::path::PathBuf;

fn state_file() -> Option<PathBuf> {
    crux_core::paths::config_dir().map(|d| d.join("gui.json"))
}

/// Remember the last folder (in %APPDATA%\crux, never in the data).
/// Screenshot-harness runs (`CRUX_GUI_SHOT`) don't: they open scratch and
/// test folders, and would leave James's next start in one of them.
pub fn remember_folder(dir: &std::path::Path) {
    if std::env::var_os("CRUX_GUI_SHOT").is_some() {
        return;
    }
    if let Some(f) = state_file() {
        if let Some(d) = f.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let _ = std::fs::write(f, serde_json::json!({ "last_folder": dir }).to_string());
    }
}

fn last_folder() -> Option<PathBuf> {
    let b = std::fs::read(state_file()?).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&b).ok()?;
    v.get("last_folder")?.as_str().map(PathBuf::from).filter(|p| p.is_dir())
}

fn main() -> eframe::Result {
    let mut args = app::Args { dir: None, cache_dir: None, library: None };
    let mut it = std::env::args_os().skip(1);
    while let Some(a) = it.next() {
        match a.to_string_lossy().as_ref() {
            "--cache-dir" => args.cache_dir = it.next().map(PathBuf::from),
            "--library" => args.library = it.next().map(PathBuf::from),
            _ => args.dir = Some(PathBuf::from(a)),
        }
    }
    if args.library.is_none() {
        args.library = std::env::var_os("CRUX_LIBRARY").map(PathBuf::from);
    }
    if args.dir.is_none() {
        args.dir = last_folder();
    }
    let size = std::env::var("CRUX_GUI_SIZE")
        .ok()
        .and_then(|s| s.split_once('x').and_then(|(w, h)| Some([w.parse::<f32>().ok()?, h.parse::<f32>().ok()?])))
        .unwrap_or([1500.0, 950.0]);
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("crux")
            .with_icon(window_icon())
            .with_inner_size(size)
            .with_min_inner_size([900.0, 600.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native("crux", options, Box::new(move |cc| Ok(Box::new(app::App::new(cc, args)))))
}

/// The title-bar and taskbar icon (the exe's own icon comes from build.rs).
/// `assets/make_icon.py` draws it.
fn window_icon() -> eframe::egui::IconData {
    let img = ::image::load_from_memory(include_bytes!("../assets/crux.png")).expect("embedded icon").into_rgba8();
    let (width, height) = img.dimensions();
    eframe::egui::IconData { rgba: img.into_raw(), width, height }
}
