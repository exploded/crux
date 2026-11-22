//! Lanczos-3 warps on the GPU: the same sums as `process::resample`
//! (4,096-step kernel table, PixInsight-style clamping, the same edge
//! rules), one output pixel per thread.
//!
//! Where each output pixel comes from is worked out on the CPU in f64 on a
//! grid every few pixels (as `warp_with` always did) and sent as integer
//! pixel plus f32 fraction. The shader interpolates relative to one grid
//! node's whole pixel, so a source coordinate of 16,000 keeps about 1e-6
//! px rather than the 1e-3 px of a plain f32.

use super::Gpu;
use crate::image::Image;
use anyhow::{Context, Result};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

/// Grid nodes with no source pixel.
const NONE: i32 = i32::MIN;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    sw: u32,
    sh: u32,
    w: u32,
    h: u32,
    gw: u32,
    step: u32,
    /// 0: like `warp` (outside → 0, no neighbour check); 1: like
    /// `warp_with` (outside → NaN, the four nearest must be non-zero).
    mode: u32,
    /// Clamping threshold, or negative for none.
    clamp: f32,
}

const SRC: &str = "
struct P { sw: u32, sh: u32, w: u32, h: u32, gw: u32, step: u32, mode: u32, clamp: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> grid: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read> lut: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;

fn lw(d: f32) -> f32 {
  let a = abs(d) * 4096.0;
  let i = u32(a);
  if (i >= 12288u) { return 0.0; }
  let f = a - f32(i);
  return lut[i] + (lut[i + 1u] - lut[i]) * f;
}

fn sum1(v: array<f32, 6>, w: array<f32, 6>) -> f32 {
  var sp = 0.0; var sn = 0.0; var wp = 0.0; var wn = 0.0;
  for (var k = 0u; k < 6u; k = k + 1u) {
    let s = v[k] * w[k];
    if (w[k] >= 0.0) { sp = sp + s; wp = wp + w[k]; } else { sn = sn + s; wn = wn + w[k]; }
  }
  if (p.clamp >= 0.0 && sp > 0.0 && -sn > p.clamp * sp) { return sp / wp; }
  return (sp + sn) / (wp + wn);
}

fn at(xx: i32, yy: i32) -> f32 {
  let cx = clamp(xx, 0, i32(p.sw) - 1);
  let cy = clamp(yy, 0, i32(p.sh) - 1);
  return src[u32(cy) * p.sw + u32(cx)];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = g.x;
  let y = g.y;
  if (x >= p.w || y >= p.h) { return; }
  let o = y * p.w + x;
  var bad = 0.0;
  if (p.mode == 1u) { bad = bitcast<f32>(0x7fc00000u); }
  let gx = x / p.step;
  let gy = y / p.step;
  let tx = f32(x % p.step) / f32(p.step);
  let ty = f32(y % p.step) / f32(p.step);
  let a = grid[gy * p.gw + gx];
  let b = grid[gy * p.gw + gx + 1u];
  let c = grid[(gy + 1u) * p.gw + gx];
  let d = grid[(gy + 1u) * p.gw + gx + 1u];
  if (a.x == -2147483647 - 1 || b.x == -2147483647 - 1 || c.x == -2147483647 - 1 || d.x == -2147483647 - 1) { outp[o] = bad; return; }
  // Relative to a's whole pixel: small numbers, full f32 precision.
  let ax = f32(0.0) + bitcast<f32>(a.z);
  let bx = f32(b.x - a.x) + bitcast<f32>(b.z);
  let cx = f32(c.x - a.x) + bitcast<f32>(c.z);
  let dx = f32(d.x - a.x) + bitcast<f32>(d.z);
  let ay = f32(0.0) + bitcast<f32>(a.w);
  let by = f32(b.y - a.y) + bitcast<f32>(b.w);
  let cy = f32(c.y - a.y) + bitcast<f32>(c.w);
  let dy = f32(d.y - a.y) + bitcast<f32>(d.w);
  let rx = (ax * (1.0 - tx) + bx * tx) * (1.0 - ty) + (cx * (1.0 - tx) + dx * tx) * ty;
  let ry = (ay * (1.0 - tx) + by * tx) * (1.0 - ty) + (cy * (1.0 - tx) + dy * tx) * ty;
  let flx = floor(rx);
  let fly = floor(ry);
  let ix = a.x + i32(flx);
  let iy = a.y + i32(fly);
  let fx = rx - flx;
  let fy = ry - fly;
  let mx = i32(p.sw) - 1;
  let my = i32(p.sh) - 1;
  if (ix < 0 || iy < 0 || ix > mx || iy > my || (ix == mx && fx > 0.0) || (iy == my && fy > 0.0)) { outp[o] = bad; return; }
  if (p.mode == 1u) {
    let n1 = src[u32(min(iy, my)) * p.sw + u32(min(ix, mx))];
    let n2 = src[u32(min(iy, my)) * p.sw + u32(min(ix + 1, mx))];
    let n3 = src[u32(min(iy + 1, my)) * p.sw + u32(min(ix, mx))];
    let n4 = src[u32(min(iy + 1, my)) * p.sw + u32(min(ix + 1, mx))];
    if (n1 == 0.0 || n2 == 0.0 || n3 == 0.0 || n4 == 0.0) { outp[o] = bad; return; }
  }
  var wx: array<f32, 6>;
  var wy: array<f32, 6>;
  for (var k = 0; k < 6; k = k + 1) {
    wx[k] = lw(fx - f32(k - 2));
    wy[k] = lw(fy - f32(k - 2));
  }
  var col: array<f32, 6>;
  for (var j = 0; j < 6; j = j + 1) {
    var vals: array<f32, 6>;
    for (var k = 0; k < 6; k = k + 1) { vals[k] = at(ix + k - 2, iy + j - 2); }
    col[j] = sum1(vals, wx);
  }
  outp[o] = sum1(col, wy);
}
";

struct Kernel {
    layout: wgpu::BindGroupLayout,
    pipe: wgpu::ComputePipeline,
    lut: wgpu::Buffer,
}

fn kernel(gpu: &Gpu) -> Result<&'static Kernel> {
    static K: OnceLock<Result<Kernel, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.scoped(|| {
            let d = &gpu.device;
            let buf = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            };
            let ro = wgpu::BufferBindingType::Storage { read_only: true };
            let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("crux warp"),
                entries: &[buf(0, wgpu::BufferBindingType::Uniform), buf(1, ro), buf(2, ro), buf(3, ro), buf(4, wgpu::BufferBindingType::Storage { read_only: false })],
            });
            let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("crux warp"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            let module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("crux warp"), source: wgpu::ShaderSource::Wgsl(SRC.into()) });
            let pipe = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("crux warp"), layout: Some(&pl), module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
            let lut = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("crux lanczos"), contents: bytemuck::cast_slice(crate::process::resample::lanczos_table()), usage: wgpu::BufferUsages::STORAGE });
            Ok(Kernel { layout, pipe, lut })
        })
        .map_err(|e| format!("{e:#}"))
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!("the GPU warp didn't build: {e}"))
}

/// Grid of source positions every `step` output pixels over a `w`×`h`
/// output (one node past each edge), as (whole x, whole y, fraction x,
/// fraction y) with `NONE` where there is no source pixel.
pub fn grid(w: usize, h: usize, step: usize, map: &(dyn Fn(f64, f64) -> Option<(f64, f64)> + Sync)) -> (usize, Vec<[i32; 4]>) {
    use rayon::prelude::*;
    let (gw, gh) = (w.div_ceil(step) + 1, h.div_ceil(step) + 1);
    let nodes = (0..gw * gh)
        .into_par_iter()
        .map(|i| match map(((i % gw) * step) as f64, ((i / gw) * step) as f64) {
            Some((sx, sy)) if sx.is_finite() && sy.is_finite() && sx.abs() < 1e9 && sy.abs() < 1e9 => {
                let (fx, fy) = (sx.floor(), sy.floor());
                [fx as i32, fy as i32, ((sx - fx) as f32).to_bits() as i32, ((sy - fy) as f32).to_bits() as i32]
            }
            _ => [NONE, NONE, 0, 0],
        })
        .collect();
    (gw, nodes)
}

/// Warp `src` to `w`×`h` on the GPU through `grid` (from [`grid`] with
/// `step`). `Ok(None)` when it's too big for one binding: use the CPU.
#[allow(clippy::too_many_arguments)]
pub fn run(gpu: &Gpu, src: &Image, w: usize, h: usize, step: usize, gw: usize, nodes: &[[i32; 4]], nan_outside: bool, clamp: Option<f32>) -> Result<Option<Vec<f32>>> {
    let (src_bytes, out_bytes) = ((src.data.len() * 4) as u64, (w * h * 4) as u64);
    if src_bytes.max(out_bytes).max((nodes.len() * 16) as u64) > gpu.max_binding() || src.data.is_empty() || w * h == 0 {
        return Ok(None);
    }
    let k = kernel(gpu)?;
    let _busy = gpu.lock();
    let d = &gpu.device;
    let p = Params { sw: src.width as u32, sh: src.height as u32, w: w as u32, h: h as u32, gw: gw as u32, step: step as u32, mode: nan_outside as u32, clamp: clamp.unwrap_or(-1.0) };
    let out = gpu.scoped(|| {
        let ubuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("crux warp params"), contents: bytemuck::bytes_of(&p), usage: wgpu::BufferUsages::UNIFORM });
        let sbuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("crux warp source"), contents: bytemuck::cast_slice(&src.data), usage: wgpu::BufferUsages::STORAGE });
        let gbuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("crux warp grid"), contents: bytemuck::cast_slice(nodes), usage: wgpu::BufferUsages::STORAGE });
        let obuf = d.create_buffer(&wgpu::BufferDescriptor { label: Some("crux warp out"), size: out_bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let rbuf = d.create_buffer(&wgpu::BufferDescriptor { label: Some("crux warp readback"), size: out_bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("crux warp"),
            layout: &k.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: k.lut.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: obuf.as_entire_binding() },
            ],
        });
        let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("crux warp") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("crux warp"), timestamp_writes: None });
            pass.set_pipeline(&k.pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(16) as u32, h.div_ceil(16) as u32, 1);
        }
        enc.copy_buffer_to_buffer(&obuf, 0, &rbuf, 0, out_bytes);
        let idx = gpu.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        rbuf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.wait_for(idx)?;
        rx.recv().context("the GPU dropped a readback")?.map_err(|e| anyhow::anyhow!("read back from the GPU: {e}"))?;
        let v = {
            let view = rbuf.slice(..).get_mapped_range().map_err(|e| anyhow::anyhow!("{e:?}"))?;
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        rbuf.unmap();
        Ok(v)
    })?;
    Ok(Some(out))
}
