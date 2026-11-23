//! Colours, fonts and egui style.
//! Dark only.

use crux_core::grade::Grade;
use eframe::egui::{self, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Stroke, TextStyle};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct Pal {
    pub bg: Color32,
    pub panel: Color32,
    pub panel2: Color32,
    pub line: Color32,
    pub fg: Color32,
    pub muted: Color32,
    pub faint: Color32,
    pub accent: Color32,
    pub accent_ink: Color32,
    pub well: Color32,
    pub good: Color32,
    pub warn: Color32,
    pub rej: Color32,
    pub info: Color32,
    pub f_h: Color32,
    pub f_o: Color32,
    pub f_s: Color32,
    pub f_l: Color32,
    pub f_r: Color32,
    pub f_g: Color32,
    pub f_b: Color32,
}

pub const fn hex(v: u32) -> Color32 {
    Color32::from_rgb((v >> 16) as u8, (v >> 8) as u8, v as u8)
}

impl Pal {
    pub fn dark() -> Pal {
        Pal {
            bg: hex(0x0a0d14),
            panel: hex(0x111622),
            panel2: hex(0x171d2c),
            line: hex(0x232b3e),
            fg: hex(0xdfe5f0),
            muted: hex(0x8c96ac),
            faint: hex(0x5b6479),
            accent: hex(0x8fb3ff),
            accent_ink: hex(0x0a0d14),
            well: hex(0x05070b),
            good: hex(0x4fd18b),
            warn: hex(0xf2c14e),
            rej: hex(0xff5c6c),
            info: hex(0x8fb3ff),
            f_h: hex(0xff5a6e),
            f_o: hex(0x2fd0c4),
            f_s: hex(0xff9a3c),
            f_l: hex(0xe8ecf4),
            f_r: hex(0xff6b6b),
            f_g: hex(0x6bdc8a),
            f_b: hex(0x6ba8ff),
        }
    }

    /// Fixed filter colours: H red, O teal, S amber, L white, R/G/B.
    pub fn filter(&self, f: &str) -> Color32 {
        match f.trim().to_uppercase().as_str() {
            "H" | "HA" | "HALPHA" => self.f_h,
            "O" | "OIII" | "O3" => self.f_o,
            "S" | "SII" | "S2" => self.f_s,
            "L" | "LUM" | "LUMINANCE" => self.f_l,
            "R" | "RED" => self.f_r,
            "G" | "GREEN" => self.f_g,
            "B" | "BLUE" => self.f_b,
            _ => self.muted,
        }
    }

    pub fn grade(&self, g: Grade) -> Color32 {
        match g {
            Grade::Good => self.good,
            Grade::Warn => self.warn,
            Grade::Reject => self.rej,
        }
    }
}

/// `c` at `a` (0–1) over `base`, like CSS color-mix.
pub fn mix(c: Color32, a: f32, base: Color32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 * a + y as f32 * (1.0 - a)).round() as u8;
    Color32::from_rgb(l(c.r(), base.r()), l(c.g(), base.g()), l(c.b(), base.b()))
}

pub fn alpha(c: Color32, a: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), (a * 255.0) as u8)
}

pub const SEMIBOLD: &str = "semibold";

pub fn semibold(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(SEMIBOLD.into()))
}

pub fn body(size: f32) -> FontId {
    FontId::new(size, FontFamily::Proportional)
}

pub fn mono(size: f32) -> FontId {
    FontId::new(size, FontFamily::Monospace)
}

/// Segoe UI (and its semibold) for text, Cascadia Mono or Consolas for
/// numbers; egui's own fonts stay as fallbacks for symbols.
pub fn install_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    let dir = std::path::Path::new(r"C:\Windows\Fonts");
    let mut load = |name: &str, files: &[&str]| -> bool {
        for f in files {
            if let Ok(b) = std::fs::read(dir.join(f)) {
                fonts.font_data.insert(name.to_string(), Arc::new(FontData::from_owned(b)));
                return true;
            }
        }
        false
    };
    let ui = load("ui", &["segoeui.ttf"]);
    let sb = load("ui-semibold", &["seguisb.ttf", "segoeuib.ttf"]);
    let mo = load("ui-mono", &["CascadiaMono.ttf", "consola.ttf"]);
    let sym = load("ui-symbols", &["seguisym.ttf"]);
    let prop = fonts.families.entry(FontFamily::Proportional).or_default();
    if ui {
        prop.insert(0, "ui".into());
    }
    if sym {
        prop.push("ui-symbols".into());
    }
    let prop_list = prop.clone();
    let mut semi = if sb { vec!["ui-semibold".to_string()] } else { Vec::new() };
    semi.extend(prop_list);
    fonts.families.insert(FontFamily::Name(SEMIBOLD.into()), semi);
    let monol = fonts.families.entry(FontFamily::Monospace).or_default();
    if mo {
        monol.insert(0, "ui-mono".into());
    }
    if sym {
        monol.push("ui-symbols".into());
    }
    ctx.set_fonts(fonts);
}

pub fn apply(ctx: &egui::Context, p: &Pal) {
    let mut style = (*ctx.global_style()).clone();
    style.text_styles = [
        (TextStyle::Heading, semibold(22.0)),
        (TextStyle::Body, body(14.0)),
        (TextStyle::Button, body(13.5)),
        (TextStyle::Small, body(11.5)),
        (TextStyle::Monospace, mono(13.0)),
    ]
    .into();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(12.0, 6.0);
    style.spacing.interact_size.y = 26.0;
    style.spacing.slider_rail_height = 5.0;
    let v = &mut style.visuals;
    *v = egui::Visuals::dark();
    v.dark_mode = true;
    v.override_text_color = Some(p.fg);
    v.panel_fill = p.bg;
    v.window_fill = p.panel;
    v.window_stroke = Stroke::new(1.0, p.line);
    v.window_corner_radius = CornerRadius::same(12);
    v.extreme_bg_color = p.bg;
    v.faint_bg_color = p.panel2;
    v.code_bg_color = p.panel2;
    v.hyperlink_color = p.accent;
    v.selection.bg_fill = p.accent;
    v.selection.stroke = Stroke::new(1.0, p.accent_ink);
    v.slider_trailing_fill = true;
    v.handle_shape = egui::style::HandleShape::Circle;
    let r = CornerRadius::same(8);
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = p.panel;
    w.noninteractive.weak_bg_fill = p.panel;
    w.noninteractive.bg_stroke = Stroke::new(1.0, p.line);
    w.noninteractive.fg_stroke = Stroke::new(1.0, p.fg);
    w.noninteractive.corner_radius = r;
    w.inactive.bg_fill = p.panel2;
    w.inactive.weak_bg_fill = p.panel2;
    w.inactive.bg_stroke = Stroke::new(1.0, p.line);
    w.inactive.fg_stroke = Stroke::new(1.0, p.fg);
    w.inactive.corner_radius = r;
    w.hovered.bg_fill = p.panel2;
    w.hovered.weak_bg_fill = p.panel2;
    w.hovered.bg_stroke = Stroke::new(1.0, p.faint);
    w.hovered.fg_stroke = Stroke::new(1.5, p.fg);
    w.hovered.corner_radius = r;
    w.hovered.expansion = 0.0;
    w.active.bg_fill = p.panel2;
    w.active.weak_bg_fill = p.panel2;
    w.active.bg_stroke = Stroke::new(1.0, p.accent);
    w.active.fg_stroke = Stroke::new(1.5, p.fg);
    w.active.corner_radius = r;
    w.active.expansion = 0.0;
    w.open = w.active;
    ctx.set_global_style(style);
}
