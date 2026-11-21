//! Full-resolution frame viewer. Pixels go to the GPU once as an R16F
//! texture with mips; a WGSL shader applies the auto stretch, blink against
//! the field's reference (registered with the star-match transform) and a
//! difference view, so stepping and toggling cost nothing on the CPU.
//! Frames load on two background threads, nearest-needed first.

use crux_core::transparency::Affine;
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use half::f16;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

/// Frames kept on the GPU (about 70 MB each for a 26 MP sub).
const RESIDENT: usize = 6;

pub struct FrameData {
    pub li: usize,
    pub w: u32,
    pub h: u32,
    pub c0: f32,
    pub m: f32,
    pub median: f32,
    pub sigma: f32,
    pub levels: Vec<(u32, u32, Vec<f16>)>,
    /// Read, stretch statistics and mips, milliseconds.
    pub ms: f32,
}

/// Facts about a frame on the GPU.
#[derive(Clone, Copy, Debug)]
pub struct Resident {
    pub w: u32,
    pub h: u32,
    pub c0: f32,
    pub m: f32,
    pub median: f32,
    pub sigma: f32,
}

struct Queue {
    want: VecDeque<(usize, PathBuf)>,
    busy: HashSet<usize>,
}

pub struct Loader {
    q: Arc<(Mutex<Queue>, Condvar)>,
    rx: std::sync::mpsc::Receiver<Result<FrameData, (usize, String)>>,
}

impl Loader {
    pub fn new(ctx: egui::Context) -> Loader {
        let q = Arc::new((Mutex::new(Queue { want: VecDeque::new(), busy: HashSet::new() }), Condvar::new()));
        let (tx, rx) = std::sync::mpsc::channel();
        for n in 0..2 {
            let q = q.clone();
            let tx = tx.clone();
            let ctx = ctx.clone();
            std::thread::Builder::new()
                .name(format!("crux-load-{n}"))
                .spawn(move || loop {
                    let (li, path) = {
                        let (m, cv) = &*q;
                        let mut g = m.lock().unwrap();
                        loop {
                            if let Some(x) = g.want.pop_front() {
                                g.busy.insert(x.0);
                                break x;
                            }
                            g = cv.wait(g).unwrap();
                        }
                    };
                    let r = load(li, &path).map_err(|e| (li, format!("{e:#}")));
                    q.0.lock().unwrap().busy.remove(&li);
                    if tx.send(r).is_err() {
                        return;
                    }
                    ctx.request_repaint();
                })
                .expect("spawn loader");
        }
        Loader { q, rx }
    }

    /// Replace the queue: load these, in order, skipping any already here.
    pub fn want(&self, list: Vec<(usize, PathBuf)>, have: &dyn Fn(usize) -> bool) {
        let (m, cv) = &*self.q;
        let mut g = m.lock().unwrap();
        let busy = g.busy.clone();
        g.want = list.into_iter().filter(|(li, _)| !have(*li) && !busy.contains(li)).collect();
        cv.notify_all();
    }

    pub fn try_recv(&self) -> Option<Result<FrameData, (usize, String)>> {
        self.rx.try_recv().ok()
    }
}

fn load(li: usize, path: &std::path::Path) -> anyhow::Result<FrameData> {
    let t0 = std::time::Instant::now();
    let img = crux_core::read_image(path)?;
    // A raw one-shot-colour sub shows as its luminance: the Bayer pattern
    // would be a checkerboard (hot pixels stay, as in mono subs).
    let img = match crux_core::meta::read_meta(path).ok().and_then(|m| m.bayer()) {
        Some(_) => crux_core::cfa::luminance(&img),
        None => img,
    };
    let stf = crux_core::stretch::Stf::auto(&img);
    let mut sample: Vec<f32> = img.data.iter().step_by(13).copied().collect();
    let (median, sigma) = crux_core::stats::median_sigma_mut(&mut sample);
    let (mut w, mut h) = (img.width, img.height);
    let to16 = |v: &[f32]| -> Vec<f16> { v.par_iter().map(|&x| f16::from_f32(x)).collect() };
    let mut levels = vec![(w as u32, h as u32, to16(&img.data))];
    let mut cur = img.data;
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let src = &cur;
        let next: Vec<f32> = (0..nh)
            .into_par_iter()
            .flat_map_iter(|y| {
                let (y0, y1) = ((2 * y).min(h - 1), (2 * y + 1).min(h - 1));
                (0..nw).map(move |x| {
                    let (x0, x1) = ((2 * x).min(w - 1), (2 * x + 1).min(w - 1));
                    0.25 * (src[y0 * w + x0] + src[y0 * w + x1] + src[y1 * w + x0] + src[y1 * w + x1])
                })
            })
            .collect();
        levels.push((nw as u32, nh as u32, to16(&next)));
        cur = next;
        (w, h) = (nw, nh);
    }
    Ok(FrameData { li, w: img.width as u32, h: img.height as u32, c0: stf.c0, m: stf.m, median, sigma, levels, ms: t0.elapsed().as_secs_f32() * 1000.0 })
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    img_min: [f32; 2],
    img_max: [f32; 2],
    size: [f32; 2],
    ref_size: [f32; 2],
    stf: [f32; 4],
    aff_a: [f32; 4],
    aff_b: [f32; 4],
    diff: [f32; 4],
    mode: [f32; 4],
    bg: [f32; 4],
}

struct Tex {
    _tex: wgpu::Texture,
    view: wgpu::TextureView,
}

/// GPU objects, kept in egui-wgpu's callback resources.
pub struct Gpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    ubuf: wgpu::Buffer,
    textures: HashMap<usize, Tex>,
    bind: Option<wgpu::BindGroup>,
}

const SHADER: &str = r#"
struct U {
  img_min: vec2<f32>, img_max: vec2<f32>, size: vec2<f32>, ref_size: vec2<f32>,
  stf: vec4<f32>, aff_a: vec4<f32>, aff_b: vec4<f32>, diff: vec4<f32>, mode: vec4<f32>, bg: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var rtex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

struct VO { @builtin(position) pos: vec4<f32>, @location(0) img: vec2<f32> };

@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var c = array<vec2<f32>, 4>(vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0));
  let t = c[i];
  var o: VO;
  o.pos = vec4(t.x * 2.0 - 1.0, 1.0 - t.y * 2.0, 0.0, 1.0);
  o.img = mix(u.img_min, u.img_max, t);
  return o;
}

fn mtf(x: f32, m: f32) -> f32 {
  if (x <= 0.0) { return 0.0; }
  if (x >= 1.0) { return 1.0; }
  return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m);
}
fn stretch(v: f32, c0: f32, m: f32) -> f32 { return mtf((v - c0) / (1.0 - c0), m); }

@fragment fn fs(i: VO) -> @location(0) vec4<f32> {
  let p = i.img;
  let uv = p / u.size;
  let v = textureSample(tex, samp, uv).r;
  let q = vec2(u.aff_a.x + u.aff_a.y * p.x + u.aff_a.z * p.y, u.aff_b.x + u.aff_b.y * p.x + u.aff_b.z * p.y);
  let ruv = q / u.ref_size;
  let r = textureSample(rtex, samp, ruv).r;
  let inside = all(uv >= vec2(0.0)) && all(uv <= vec2(1.0));
  let rin = all(ruv >= vec2(0.0)) && all(ruv <= vec2(1.0));
  var g = stretch(v, u.stf.x, u.stf.y);
  if (u.mode.x > 0.5 && u.mode.x < 1.5) { g = select(0.0, stretch(r, u.stf.z, u.stf.w), rin); }
  if (u.mode.x > 1.5) {
    let d = (v - u.diff.x) - u.diff.z * (r - u.diff.y);
    g = select(0.5, clamp(0.5 + d / (12.0 * u.diff.w), 0.0, 1.0), rin);
  }
  if (!inside) { return u.bg; }
  return vec4(g, g, g, 1.0);
}
"#;

impl Gpu {
    pub fn textures_remove(&mut self, li: usize) {
        self.textures.remove(&li);
    }

    pub fn new(rs: &egui_wgpu::RenderState) -> Gpu {
        let device = &rs.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("crux viewer"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("crux viewer"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex_entry(1),
                tex_entry(2),
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("crux viewer"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("crux viewer"),
            layout: Some(&pl),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: rs.target_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("crux viewer"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("crux viewer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Gpu { pipeline, layout, sampler, ubuf, textures: HashMap::new(), bind: None }
    }
}

/// Upload a loaded frame (UI thread). Returns frames evicted to stay within budget.
pub fn upload(rs: &egui_wgpu::RenderState, d: &FrameData, order: &mut VecDeque<usize>, keep: &[usize]) -> Vec<usize> {
    let tex = rs.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("crux frame"),
        size: wgpu::Extent3d { width: d.w, height: d.h, depth_or_array_layers: 1 },
        mip_level_count: d.levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (lvl, (w, h, data)) in d.levels.iter().enumerate() {
        rs.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: lvl as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 2), rows_per_image: Some(*h) },
            wgpu::Extent3d { width: *w, height: *h, depth_or_array_layers: 1 },
        );
    }
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let mut r = rs.renderer.write();
    let gpu: &mut Gpu = r.callback_resources.get_mut().expect("viewer gpu");
    gpu.textures.insert(d.li, Tex { _tex: tex, view });
    order.retain(|&x| x != d.li);
    order.push_back(d.li);
    let mut evicted = Vec::new();
    while order.len() > RESIDENT {
        let Some(pos) = order.iter().position(|x| !keep.contains(x)) else { break };
        let li = order.remove(pos).unwrap();
        gpu.textures.remove(&li);
        evicted.push(li);
    }
    evicted
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Frame,
    Reference,
    Difference,
}

/// One draw of the viewer.
pub struct Draw {
    pub frame: usize,
    pub reference: Option<usize>,
    pub img_min: [f32; 2],
    pub img_max: [f32; 2],
    pub size: [f32; 2],
    pub fr: Resident,
    pub rf: Option<Resident>,
    pub to_ref: Option<Affine>,
    pub mode: Mode,
    pub bg: [f32; 4],
    /// Transparency of frame / reference, to scale the difference.
    pub k: f32,
}

impl egui_wgpu::CallbackTrait for Draw {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _sd: &egui_wgpu::ScreenDescriptor,
        _enc: &mut wgpu::CommandEncoder,
        res: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(gpu) = res.get_mut::<Gpu>() else { return Vec::new() };
        let Some(ft) = gpu.textures.get(&self.frame) else {
            gpu.bind = None;
            return Vec::new();
        };
        let rt = self.reference.and_then(|r| gpu.textures.get(&r));
        let has_ref = rt.is_some() && self.to_ref.is_some();
        let a = self.to_ref.unwrap_or(Affine::translation(0.0, 0.0));
        let rf = self.rf.unwrap_or(self.fr);
        let mode = match (self.mode, has_ref) {
            (Mode::Reference, true) => 1.0,
            (Mode::Difference, true) => 2.0,
            _ => 0.0,
        };
        let u = Uniforms {
            img_min: self.img_min,
            img_max: self.img_max,
            size: self.size,
            ref_size: [rf.w as f32, rf.h as f32],
            stf: [self.fr.c0, self.fr.m, rf.c0, rf.m],
            aff_a: [a.a[0] as f32, a.a[1] as f32, a.a[2] as f32, 0.0],
            aff_b: [a.b[0] as f32, a.b[1] as f32, a.b[2] as f32, 0.0],
            diff: [self.fr.median, rf.median, self.k, self.fr.sigma.max(1e-6)],
            mode: [mode, 0.0, if has_ref { 1.0 } else { 0.0 }, 0.0],
            bg: self.bg,
        };
        queue.write_buffer(&gpu.ubuf, 0, bytemuck::bytes_of(&u));
        let rview = rt.map(|t| &t.view).unwrap_or(&ft.view);
        gpu.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("crux viewer"),
            layout: &gpu.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gpu.ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ft.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(rview) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&gpu.sampler) },
            ],
        }));
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, rp: &mut wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        let Some(gpu) = res.get::<Gpu>() else { return };
        let Some(bind) = &gpu.bind else { return };
        rp.set_pipeline(&gpu.pipeline);
        rp.set_bind_group(0, bind, &[]);
        rp.draw(0..4, 0..1);
    }
}

/// Pan and zoom, shared by every frame so zoom stays put while stepping.
#[derive(Clone, Copy, Debug)]
pub struct View {
    /// Screen points per image pixel; None = fit.
    pub scale: Option<f32>,
    /// Image pixel at the centre of the well.
    pub centre: egui::Vec2,
}

impl Default for View {
    fn default() -> Self {
        View { scale: None, centre: egui::Vec2::ZERO }
    }
}

impl View {
    pub fn fit_scale(rect: egui::Rect, w: f32, h: f32) -> f32 {
        (rect.width() / w).min(rect.height() / h)
    }

    /// Image coordinates at the rect's corners.
    pub fn corners(&self, rect: egui::Rect, w: f32, h: f32) -> ([f32; 2], [f32; 2], f32) {
        let (s, c) = match self.scale {
            None => (Self::fit_scale(rect, w, h), egui::vec2(w / 2.0, h / 2.0)),
            Some(s) => (s, self.centre),
        };
        let half = rect.size() / (2.0 * s);
        ([c.x - half.x, c.y - half.y], [c.x + half.x, c.y + half.y], s)
    }

    /// Screen position of an image pixel coordinate.
    pub fn to_screen(rect: egui::Rect, min: [f32; 2], s: f32, x: f32, y: f32) -> egui::Pos2 {
        egui::pos2(rect.left() + (x - min[0]) * s, rect.top() + (y - min[1]) * s)
    }
}
