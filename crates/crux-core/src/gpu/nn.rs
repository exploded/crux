//! crux's networks on the GPU. The U-Nets (`tools/ml/model.py`) use six
//! operators: Conv 3×3 (pad 1) and 1×1, LeakyReLU, MaxPool 2×2/2,
//! ConvTranspose 2×2/2, Concat on channels, plus the input Slice and the
//! residual Add. The ONNX graph is read node by node (tract's protobuf
//! types) into a list of dispatches; anything else is refused, and the
//! caller uses tract on the CPU.
//!
//! Tensors are planar C×H×W f32 (a batch of tiles one after another).
//! Concat and Slice cost nothing: a tensor is a list of channel ranges of
//! buffers, and a 3×3 conv reads up to two. LeakyReLU is folded into the
//! conv before it.
//!
//! The 3×3 conv: a workgroup makes a 32 × 8 pixel tile for 32 output
//! channels, 4 input channels at a time staged in workgroup memory; each
//! of its 128 threads keeps 4 pixels × 16 channels in registers.

use super::Gpu;
use anyhow::{bail, Context, Result};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use tract_onnx::pb;
use tract_onnx::prelude::Framework;
use wgpu::util::DeviceExt;

const TW: usize = 32;
const TH: usize = 8;
const CO: usize = 32;
const CI: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    cin_a: u32,
    cin_b: u32,
    cin: u32,
    cout: u32,
    cout_pad: u32,
    w: u32,
    h: u32,
    hw: u32,
    off_a: u32,
    off_b: u32,
    sa: u32,
    sb: u32,
    so: u32,
    act: u32,
    alpha: f32,
    tiles_x: u32,
}

/// A range of channels in an activation buffer.
#[derive(Debug, Clone, Copy)]
struct View {
    buf: usize,
    c_off: usize,
    c: usize,
}

#[derive(Debug, Clone)]
struct Tensor {
    parts: Vec<View>,
    c: usize,
    h: usize,
    w: usize,
}

impl Tensor {
    fn single(&self, what: &str) -> Result<View> {
        match self.parts.as_slice() {
            [v] => Ok(*v),
            _ => bail!("{what} on a concatenated tensor"),
        }
    }
}

/// An activation buffer: channels × height × width per batch item.
#[derive(Debug, Clone, Copy)]
struct Act {
    c: usize,
    h: usize,
    w: usize,
}

impl Act {
    fn len(&self) -> usize {
        self.c * self.h * self.w
    }
}

#[derive(Debug, Clone)]
enum Op {
    /// 3×3 or 1×1 conv over one or two channel ranges.
    Conv { k: usize, a: View, b: Option<View>, cout: usize, w: usize, bias: usize, out: usize, alpha: Option<f32> },
    ConvT { input: View, cout: usize, w: usize, bias: usize, out: usize },
    Pool { input: View, out: usize },
    Add { a: View, b: View, out: usize },
}

/// A network laid out for one tile size: dispatches, activation shapes,
/// weights (already reordered for the kernels).
pub struct Model {
    ops: Vec<Op>,
    acts: Vec<Act>,
    weights: Vec<Vec<f32>>,
    output: View,
    pub channels: usize,
    pub tile: usize,
}

fn floats(t: &pb::TensorProto) -> Result<(Vec<usize>, Vec<f32>)> {
    anyhow::ensure!(t.data_type == 1, "{}: not float data", t.name);
    let dims = t.dims.iter().map(|&d| d as usize).collect();
    let v = if !t.raw_data.is_empty() {
        t.raw_data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    } else {
        t.float_data.clone()
    };
    Ok((dims, v))
}

fn ints(t: &pb::TensorProto) -> Result<Vec<i64>> {
    anyhow::ensure!(t.data_type == 7, "{}: not int64 data", t.name);
    Ok(if !t.raw_data.is_empty() {
        t.raw_data.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect()
    } else {
        t.int64_data.clone()
    })
}

fn attr<'a>(n: &'a pb::NodeProto, name: &str) -> Option<&'a pb::AttributeProto> {
    n.attribute.iter().find(|a| a.name == name)
}

fn attr_ints(n: &pb::NodeProto, name: &str) -> Vec<i64> {
    attr(n, name).map(|a| a.ints.clone()).unwrap_or_default()
}

impl Model {
    /// Read `onnx` for `channels` input channels and square tiles of `tile`.
    pub fn parse(onnx: &[u8], channels: usize, tile: usize) -> Result<Model> {
        let proto = tract_onnx::onnx().proto_model_for_read(&mut std::io::Cursor::new(onnx)).context("read the network")?;
        let g = proto.graph.as_ref().context("the network has no graph")?;
        let mut init: HashMap<&str, (Vec<usize>, Vec<f32>)> = HashMap::new();
        for t in &g.initializer {
            if t.data_type == 1 {
                init.insert(t.name.as_str(), floats(t)?);
            }
        }
        let mut consts: HashMap<String, Vec<i64>> = HashMap::new();
        for t in &g.initializer {
            if t.data_type == 7 {
                consts.insert(t.name.clone(), ints(t)?);
            }
        }
        let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, n) in g.node.iter().enumerate() {
            for inp in &n.input {
                consumers.entry(inp.as_str()).or_default().push(i);
            }
        }
        let in_name = g.input.first().map(|v| v.name.clone()).context("the network has no input")?;
        let out_name = g.output.first().map(|v| v.name.clone()).context("the network has no output")?;
        let mut acts = vec![Act { c: channels, h: tile, w: tile }];
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(in_name, Tensor { parts: vec![View { buf: 0, c_off: 0, c: channels }], c: channels, h: tile, w: tile });
        let mut weights: Vec<Vec<f32>> = Vec::new();
        let mut ops = Vec::new();
        let mut skip: HashSet<usize> = HashSet::new();
        let get = |tensors: &HashMap<String, Tensor>, name: &str| -> Result<Tensor> { tensors.get(name).cloned().with_context(|| format!("tensor {name} isn't made by any earlier node")) };
        for (i, n) in g.node.iter().enumerate() {
            if skip.contains(&i) {
                continue;
            }
            let out = n.output.first().cloned().unwrap_or_default();
            match n.op_type.as_str() {
                "Constant" => {
                    let t = attr(n, "value").and_then(|a| a.t.as_ref()).context("Constant without a value")?;
                    consts.insert(out, ints(t)?);
                }
                "Slice" => {
                    let x = get(&tensors, &n.input[0])?;
                    let c = |k: usize| -> Result<Vec<i64>> { n.input.get(k).and_then(|nm| consts.get(nm)).cloned().with_context(|| format!("Slice input {k} isn't a constant")) };
                    let (starts, ends, axes) = (c(1)?, c(2)?, c(3)?);
                    let steps = c(4).unwrap_or_else(|_| vec![1]);
                    anyhow::ensure!(axes == [1] && steps == [1], "only channel slices are supported");
                    let v = x.single("Slice")?;
                    let s = starts[0].clamp(0, x.c as i64) as usize;
                    let e = ends[0].clamp(0, x.c as i64) as usize;
                    anyhow::ensure!(e > s, "empty slice");
                    tensors.insert(out, Tensor { parts: vec![View { buf: v.buf, c_off: v.c_off + s, c: e - s }], c: e - s, ..x });
                }
                "Conv" => {
                    let x = get(&tensors, &n.input[0])?;
                    let (wd, wv) = init.get(n.input[1].as_str()).cloned().context("Conv weights aren't initialisers")?;
                    let (_, bv) = n.input.get(2).and_then(|b| init.get(b.as_str()).cloned()).unwrap_or((vec![], vec![0.0; wd[0]]));
                    anyhow::ensure!(wd.len() == 4 && wd[1] == x.c && wd[2] == wd[3], "Conv weights {wd:?} don't fit {} channels", x.c);
                    let k = wd[2];
                    anyhow::ensure!(k == 1 || k == 3, "only 1×1 and 3×3 convs are supported");
                    anyhow::ensure!(attr(n, "group").map(|a| a.i).unwrap_or(1) == 1, "grouped conv");
                    anyhow::ensure!(attr_ints(n, "strides").iter().all(|&s| s == 1), "strided conv");
                    anyhow::ensure!(attr_ints(n, "dilations").iter().all(|&s| s == 1), "dilated conv");
                    let p = (k / 2) as i64;
                    let pads = attr_ints(n, "pads");
                    anyhow::ensure!(pads.is_empty() && k == 1 || pads.iter().all(|&q| q == p), "conv padding {pads:?}");
                    anyhow::ensure!(x.parts.len() <= 2, "conv over more than two channel ranges");
                    let cout = wd[0];
                    // A LeakyReLU straight after is done by the conv.
                    let mut alpha = None;
                    let mut out_name = out.clone();
                    if let Some(cs) = consumers.get(out.as_str()) {
                        if let [j] = cs.as_slice() {
                            let nj = &g.node[*j];
                            if nj.op_type == "LeakyRelu" {
                                alpha = Some(attr(nj, "alpha").map(|a| a.f).unwrap_or(0.01));
                                out_name = nj.output[0].clone();
                                skip.insert(*j);
                            }
                        }
                    }
                    let cin = x.c;
                    let (wi, bi) = (weights.len(), weights.len() + 1);
                    if k == 3 {
                        // [co][ci][tap] → [ci][tap][co padded to 32].
                        let cp = cout.div_ceil(CO) * CO;
                        let mut w = vec![0f32; cin * 9 * cp];
                        for co in 0..cout {
                            for ci in 0..cin {
                                for tap in 0..9 {
                                    w[(ci * 9 + tap) * cp + co] = wv[(co * cin + ci) * 9 + tap];
                                }
                            }
                        }
                        let mut b = bv.clone();
                        b.resize(cp, 0.0);
                        weights.push(w);
                        weights.push(b);
                    } else {
                        weights.push(wv.clone());
                        weights.push(bv.clone());
                    }
                    acts.push(Act { c: cout, h: x.h, w: x.w });
                    let ob = acts.len() - 1;
                    ops.push(Op::Conv { k, a: x.parts[0], b: x.parts.get(1).copied(), cout, w: wi, bias: bi, out: ob, alpha });
                    tensors.insert(out_name, Tensor { parts: vec![View { buf: ob, c_off: 0, c: cout }], c: cout, h: x.h, w: x.w });
                }
                "ConvTranspose" => {
                    let x = get(&tensors, &n.input[0])?;
                    let (wd, wv) = init.get(n.input[1].as_str()).cloned().context("ConvTranspose weights aren't initialisers")?;
                    anyhow::ensure!(wd.len() == 4 && wd[0] == x.c && wd[2] == 2 && wd[3] == 2, "only 2×2 transposed convs are supported ({wd:?})");
                    anyhow::ensure!(attr_ints(n, "strides").iter().all(|&s| s == 2), "transposed conv stride");
                    anyhow::ensure!(attr_ints(n, "pads").iter().all(|&s| s == 0), "transposed conv padding");
                    anyhow::ensure!(attr(n, "group").map(|a| a.i).unwrap_or(1) == 1, "grouped transposed conv");
                    let cout = wd[1];
                    let (_, bv) = n.input.get(2).and_then(|b| init.get(b.as_str()).cloned()).unwrap_or((vec![], vec![0.0; cout]));
                    let v = x.single("ConvTranspose")?;
                    // [ci][co][2][2] is already the kernel's layout; pad
                    // so the last channel group never reads past the end.
                    let mut w = wv.clone();
                    w.resize(w.len() + 8 * 4, 0.0);
                    weights.push(w);
                    weights.push(bv.clone());
                    acts.push(Act { c: cout, h: x.h * 2, w: x.w * 2 });
                    let ob = acts.len() - 1;
                    ops.push(Op::ConvT { input: v, cout, w: weights.len() - 2, bias: weights.len() - 1, out: ob });
                    tensors.insert(out, Tensor { parts: vec![View { buf: ob, c_off: 0, c: cout }], c: cout, h: x.h * 2, w: x.w * 2 });
                }
                "MaxPool" => {
                    let x = get(&tensors, &n.input[0])?;
                    anyhow::ensure!(attr_ints(n, "kernel_shape") == [2, 2] && attr_ints(n, "strides") == [2, 2], "only 2×2 max pooling is supported");
                    anyhow::ensure!(attr_ints(n, "pads").iter().all(|&s| s == 0), "padded pooling");
                    anyhow::ensure!(x.h % 2 == 0 && x.w % 2 == 0, "pooling an odd size");
                    let v = x.single("MaxPool")?;
                    acts.push(Act { c: x.c, h: x.h / 2, w: x.w / 2 });
                    let ob = acts.len() - 1;
                    ops.push(Op::Pool { input: v, out: ob });
                    tensors.insert(out, Tensor { parts: vec![View { buf: ob, c_off: 0, c: x.c }], c: x.c, h: x.h / 2, w: x.w / 2 });
                }
                "Concat" => {
                    anyhow::ensure!(attr(n, "axis").map(|a| a.i).unwrap_or(1) == 1, "concat on another axis");
                    let ts: Vec<Tensor> = n.input.iter().map(|nm| get(&tensors, nm)).collect::<Result<_>>()?;
                    let (h, w) = (ts[0].h, ts[0].w);
                    anyhow::ensure!(ts.iter().all(|t| t.h == h && t.w == w), "concat of different sizes");
                    let parts: Vec<View> = ts.iter().flat_map(|t| t.parts.clone()).collect();
                    let c = ts.iter().map(|t| t.c).sum();
                    tensors.insert(out, Tensor { parts, c, h, w });
                }
                "Add" => {
                    let a = get(&tensors, &n.input[0])?;
                    let b = get(&tensors, &n.input[1])?;
                    anyhow::ensure!((a.c, a.h, a.w) == (b.c, b.h, b.w), "Add of different shapes");
                    acts.push(Act { c: a.c, h: a.h, w: a.w });
                    let ob = acts.len() - 1;
                    ops.push(Op::Add { a: a.single("Add")?, b: b.single("Add")?, out: ob });
                    tensors.insert(out, Tensor { parts: vec![View { buf: ob, c_off: 0, c: a.c }], ..a });
                }
                other => bail!("the GPU can't run {other} nodes"),
            }
        }
        let o = get(&tensors, &out_name)?;
        anyhow::ensure!(o.c == 1 && o.h == tile && o.w == tile, "unexpected output shape {}×{}×{}", o.c, o.h, o.w);
        Ok(Model { ops, acts, weights, output: o.single("output")?, channels, tile })
    }

    /// Activation bytes for one tile.
    fn bytes_per_tile(&self) -> u64 {
        self.acts.iter().map(|a| a.len() as u64 * 4).sum()
    }

    fn largest_act(&self) -> u64 {
        self.acts.iter().map(|a| a.len() as u64 * 4).max().unwrap_or(0)
    }
}

fn header(wtype: &str) -> String {
    format!(
        "struct P {{ cin_a: u32, cin_b: u32, cin: u32, cout: u32, cout_pad: u32, w: u32, h: u32, hw: u32,
  off_a: u32, off_b: u32, sa: u32, sb: u32, so: u32, act: u32, alpha: f32, tiles_x: u32 }}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<{wtype}>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
fn act(v: f32) -> f32 {{ if (p.act == 1u && v < 0.0) {{ return v * p.alpha; }} return v; }}
fn rd(n: u32, ci: u32, pix: u32) -> f32 {{
  if (ci < p.cin_a) {{ return A[n * p.sa + p.off_a + ci * p.hw + pix]; }}
  return B[n * p.sb + p.off_b + (ci - p.cin_a) * p.hw + pix];
}}
"
    )
}

/// The 3×3 conv, its inner loops unrolled here so every accumulator is a
/// register.
fn conv3_src() -> String {
    let xw = TW + 2;
    let xh = TH + 2;
    let xplane = xw * xh;
    let xs_len = CI * xplane;
    let ws_len = CI * 9 * CO / 4;
    let threads = (TW / 4) * TH * 2;
    let mut s = header("vec4<f32>");
    s += &format!(
        "var<workgroup> xs: array<f32, {xs_len}>;
var<workgroup> ws: array<vec4<f32>, {ws_len}>;
fn st(ob: u32, cob: u32, pix: u32, v: vec4<f32>) {{
  for (var e = 0u; e < 4u; e = e + 1u) {{
    let co = cob + e;
    if (co < p.cout) {{ O[ob + co * p.hw + pix] = act(v[e] + bias[co]); }}
  }}
}}
@compute @workgroup_size({threads})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {{
  let n = wg.z;
  let tx = wg.x % p.tiles_x;
  let ty = wg.x / p.tiles_x;
  let x0 = i32(tx * {TW}u);
  let y0 = i32(ty * {TH}u);
  let co0 = wg.y * {CO}u;
  let pg = li % {pgs}u;
  let qh = (li / {pgs}u) * 4u;
  let px = (pg % {gx}u) * 4u;
  let py = pg / {gx}u;
  let cq = p.cout_pad / 4u;
",
        pgs = threads / 2,
        gx = TW / 4,
    );
    for i in 0..4 {
        for q in 0..4 {
            s += &format!("  var a{i}{q} = vec4<f32>(0.0);\n");
        }
    }
    s += &format!(
        "  for (var ci0 = 0u; ci0 < p.cin; ci0 = ci0 + {CI}u) {{
    for (var i = li; i < {xs_len}u; i = i + {threads}u) {{
      let c = i / {xplane}u;
      let r = (i / {xw}u) % {xh}u;
      let col = i % {xw}u;
      let gy = y0 + i32(r) - 1;
      let gx = x0 + i32(col) - 1;
      var v = 0.0;
      if (ci0 + c < p.cin && gx >= 0 && gy >= 0 && gx < i32(p.w) && gy < i32(p.h)) {{ v = rd(n, ci0 + c, u32(gy) * p.w + u32(gx)); }}
      xs[i] = v;
    }}
    for (var i = li; i < {ws_len}u; i = i + {threads}u) {{
      let c = i / {wc}u;
      let rem = i % {wc}u;
      let tap = rem / {cq8}u;
      let q = rem % {cq8}u;
      var v = vec4<f32>(0.0);
      if (ci0 + c < p.cin) {{ v = W[((ci0 + c) * 9u + tap) * cq + co0 / 4u + q]; }}
      ws[i] = v;
    }}
    workgroupBarrier();
    for (var c = 0u; c < {CI}u; c = c + 1u) {{
",
        wc = 9 * CO / 4,
        cq8 = CO / 4,
    );
    for ky in 0..3 {
        s += &format!("      {{\n        let xb = c * {xplane}u + (py + {ky}u) * {xw}u + px;\n");
        for r in 0..6 {
            s += &format!("        let r{r} = xs[xb + {r}u];\n");
        }
        for kx in 0..3 {
            let tap = ky * 3 + kx;
            s += &format!("        {{\n          let wb = (c * 9u + {tap}u) * {cq8}u + qh;\n", cq8 = CO / 4);
            for q in 0..4 {
                s += &format!("          let w{q} = ws[wb + {q}u];\n");
            }
            for i in 0..4 {
                for q in 0..4 {
                    s += &format!("          a{i}{q} = a{i}{q} + w{q} * r{};\n", i + kx);
                }
            }
            s += "        }\n";
        }
        s += "      }\n";
    }
    s += "    }\n    workgroupBarrier();\n  }\n";
    s += "  let y = u32(y0) + py;\n  if (y >= p.h) { return; }\n  let ob = n * p.so;\n";
    for i in 0..4 {
        s += &format!("  {{\n    let x = u32(x0) + px + {i}u;\n    if (x < p.w) {{\n");
        for q in 0..4 {
            s += &format!("      st(ob, co0 + (qh + {q}u) * 4u, y * p.w + x, a{i}{q});\n");
        }
        s += "    }\n  }\n";
    }
    s += "}\n";
    s
}

fn conv1_src() -> String {
    header("f32")
        + "@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let pix = g.x;
  let n = g.z;
  if (pix >= p.hw) { return; }
  for (var co = 0u; co < p.cout; co = co + 1u) {
    var s = bias[co];
    for (var ci = 0u; ci < p.cin; ci = ci + 1u) { s = s + W[co * p.cin + ci] * rd(n, ci, pix); }
    O[n * p.so + co * p.hw + pix] = act(s);
  }
}
"
}

fn convt_src() -> String {
    let mut s = header("vec4<f32>");
    s += "@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let pix = g.x;
  let n = g.z;
  if (pix >= p.hw) { return; }
  let x = pix % p.w;
  let y = pix / p.w;
  let co0 = wg.y * 8u;
";
    for k in 0..8 {
        s += &format!("  var s{k} = vec4<f32>(0.0);\n");
    }
    s += "  for (var ci = 0u; ci < p.cin; ci = ci + 1u) {\n    let v = A[n * p.sa + p.off_a + ci * p.hw + pix];\n    let wb = ci * p.cout + co0;\n";
    for k in 0..8 {
        s += &format!("    s{k} = s{k} + W[wb + {k}u] * v;\n");
    }
    s += "  }\n  let ow = 2u * p.w;\n  let o0 = (2u * y) * ow + 2u * x;\n";
    for k in 0..8 {
        s += &format!(
            "  if (co0 + {k}u < p.cout) {{
    let b = bias[co0 + {k}u];
    let ob = n * p.so + (co0 + {k}u) * 4u * p.hw;
    O[ob + o0] = s{k}.x + b;
    O[ob + o0 + 1u] = s{k}.y + b;
    O[ob + o0 + ow] = s{k}.z + b;
    O[ob + o0 + ow + 1u] = s{k}.w + b;
  }}
"
        );
    }
    s += "}\n";
    s
}

fn pool_src() -> String {
    header("f32")
        + "@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  let n = g.z;
  if (i >= p.cin * p.hw) { return; }
  let c = i / p.hw;
  let pix = i % p.hw;
  let x = pix % p.w;
  let y = pix / p.w;
  let iw = 2u * p.w;
  let b = n * p.sa + p.off_a + c * 4u * p.hw + 2u * y * iw + 2u * x;
  O[n * p.so + i] = max(max(A[b], A[b + 1u]), max(A[b + iw], A[b + iw + 1u]));
}
"
}

fn add_src() -> String {
    header("f32")
        + "@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  let n = g.z;
  if (i >= p.cin * p.hw) { return; }
  O[n * p.so + i] = A[n * p.sa + p.off_a + i] + B[n * p.sb + p.off_b + i];
}
"
}

struct Kernels {
    layout: wgpu::BindGroupLayout,
    conv3: wgpu::ComputePipeline,
    conv1: wgpu::ComputePipeline,
    convt: wgpu::ComputePipeline,
    pool: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    dummy: wgpu::Buffer,
}

fn kernels(gpu: &Gpu) -> Result<&'static Kernels> {
    static K: OnceLock<Result<Kernels, String>> = OnceLock::new();
    K.get_or_init(|| gpu.scoped(|| Ok(build_kernels(gpu))).map_err(|e| format!("{e:#}"))).as_ref().map_err(|e| anyhow::anyhow!("the GPU kernels didn't build: {e}"))
}

fn build_kernels(gpu: &Gpu) -> Kernels {
    let d = &gpu.device;
    let buf = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let ro = wgpu::BufferBindingType::Storage { read_only: true };
    let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("crux nn"),
        entries: &[buf(0, wgpu::BufferBindingType::Uniform), buf(1, ro), buf(2, ro), buf(3, ro), buf(4, ro), buf(5, wgpu::BufferBindingType::Storage { read_only: false })],
    });
    let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("crux nn"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
    let pipe = |name: &str, src: String| {
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(src.into()) });
        d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(name), layout: Some(&pl), module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
    };
    let dummy = d.create_buffer(&wgpu::BufferDescriptor { label: Some("crux nn dummy"), size: 64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    Kernels { conv3: pipe("crux conv3", conv3_src()), conv1: pipe("crux conv1", conv1_src()), convt: pipe("crux convt", convt_src()), pool: pipe("crux pool", pool_src()), add: pipe("crux add", add_src()), layout, dummy }
}

/// A network on the GPU: the model and its weights uploaded.
pub struct Program {
    model: Model,
    weights: Vec<wgpu::Buffer>,
}

impl Program {
    pub fn new(gpu: &Gpu, onnx: &[u8], channels: usize, tile: usize) -> Result<Program> {
        let model = Model::parse(onnx, channels, tile)?;
        kernels(gpu)?;
        let weights = gpu.scoped(|| {
            Ok(model
                .weights
                .iter()
                .map(|w| {
                    let mut w = w.clone();
                    if w.is_empty() {
                        w.push(0.0);
                    }
                    gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("crux nn weights"), contents: bytemuck::cast_slice(&w), usage: wgpu::BufferUsages::STORAGE })
                })
                .collect())
        })?;
        Ok(Program { model, weights })
    }

    pub fn tile(&self) -> usize {
        self.model.tile
    }

    /// Run `tiles` tiles: `fill(k, input)` writes tile k's input
    /// (channels × tile × tile), `done(k, output)` takes its output
    /// (tile × tile). Tiles go through in batches, the next batch's input
    /// prepared while the card works on the current one.
    pub fn run(&self, gpu: &Gpu, tiles: usize, fill: &mut dyn FnMut(usize, &mut [f32]), done: &mut dyn FnMut(usize, &[f32]) -> Result<()>, cancel: &std::sync::atomic::AtomicBool) -> Result<()> {
        let _busy = gpu.lock();
        let k = kernels(gpu)?;
        let m = &self.model;
        // Up to 4 tiles at once, within ~1 GB and the binding limit.
        let budget: u64 = 1 << 30;
        let batch = (budget / m.bytes_per_tile().max(1)).min(gpu.max_binding() / m.largest_act().max(1)).clamp(1, 4).min(tiles.max(1) as u64) as usize;
        let t2 = m.tile * m.tile;
        let in_len = m.channels * t2;
        let d = &gpu.device;
        let (acts, binds, readback) = gpu.scoped(|| {
            let acts: Vec<wgpu::Buffer> = m
                .acts
                .iter()
                .map(|a| d.create_buffer(&wgpu::BufferDescriptor { label: Some("crux nn act"), size: (a.len() * batch * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }))
                .collect();
            let mut binds = Vec::new();
            for op in &m.ops {
                let view_params = |v: &View| -> (u32, u32) {
                    let a = m.acts[v.buf];
                    ((v.c_off * a.h * a.w) as u32, a.len() as u32)
                };
                let (pipe, p, groups, a, b, w, bias, out) = match op {
                    Op::Conv { k: ks, a, b, cout, w, bias, out, alpha } => {
                        let o = m.acts[*out];
                        let (off_a, sa) = view_params(a);
                        let (off_b, sb) = b.as_ref().map(view_params).unwrap_or((0, 0));
                        let cin_a = a.c as u32;
                        let cin_b = b.map(|v| v.c).unwrap_or(0) as u32;
                        let tiles_x = o.w.div_ceil(TW) as u32;
                        let p = Params {
                            cin_a,
                            cin_b,
                            cin: cin_a + cin_b,
                            cout: *cout as u32,
                            cout_pad: (cout.div_ceil(CO) * CO) as u32,
                            w: o.w as u32,
                            h: o.h as u32,
                            hw: (o.w * o.h) as u32,
                            off_a,
                            off_b,
                            sa,
                            sb,
                            so: o.len() as u32,
                            act: alpha.is_some() as u32,
                            alpha: alpha.unwrap_or(0.0),
                            tiles_x,
                        };
                        if *ks == 3 {
                            let groups = (tiles_x * o.h.div_ceil(TH) as u32, cout.div_ceil(CO) as u32);
                            (&k.conv3, p, groups, a.buf, b.map(|v| v.buf), *w, *bias, *out)
                        } else {
                            ((&k.conv1), p, ((o.w * o.h).div_ceil(256) as u32, 1), a.buf, b.map(|v| v.buf), *w, *bias, *out)
                        }
                    }
                    Op::ConvT { input, cout, w, bias, out } => {
                        let ia = m.acts[input.buf];
                        let (off_a, sa) = view_params(input);
                        let o = m.acts[*out];
                        let p = Params { cin_a: input.c as u32, cin: input.c as u32, cout: *cout as u32, w: ia.w as u32, h: ia.h as u32, hw: (ia.w * ia.h) as u32, off_a, sa, so: o.len() as u32, ..Default::default() };
                        (&k.convt, p, ((ia.w * ia.h).div_ceil(64) as u32, cout.div_ceil(8) as u32), input.buf, None, *w, *bias, *out)
                    }
                    Op::Pool { input, out } => {
                        let (off_a, sa) = view_params(input);
                        let o = m.acts[*out];
                        let p = Params { cin_a: input.c as u32, cin: input.c as u32, w: o.w as u32, h: o.h as u32, hw: (o.w * o.h) as u32, off_a, sa, so: o.len() as u32, ..Default::default() };
                        (&k.pool, p, ((o.len()).div_ceil(256) as u32, 1), input.buf, None, usize::MAX, usize::MAX, *out)
                    }
                    Op::Add { a, b, out } => {
                        let (off_a, sa) = view_params(a);
                        let (off_b, sb) = view_params(b);
                        let o = m.acts[*out];
                        let p = Params { cin_a: a.c as u32, cin: a.c as u32, w: o.w as u32, h: o.h as u32, hw: (o.w * o.h) as u32, off_a, off_b, sa, sb, so: o.len() as u32, ..Default::default() };
                        (&k.add, p, (o.len().div_ceil(256) as u32, 1), a.buf, Some(b.buf), usize::MAX, usize::MAX, *out)
                    }
                };
                let ubuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("crux nn params"), contents: bytemuck::bytes_of(&p), usage: wgpu::BufferUsages::UNIFORM });
                let wb = |i: usize| if i == usize::MAX { &k.dummy } else { &self.weights[i] };
                let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("crux nn"),
                    layout: &k.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: acts[a].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: b.map(|bb| &acts[bb]).unwrap_or(&acts[a]).as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: wb(w).as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: wb(bias).as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: acts[out].as_entire_binding() },
                    ],
                });
                binds.push((pipe, bg, groups, ubuf));
            }
            let readback: Vec<wgpu::Buffer> = (0..2)
                .map(|_| d.create_buffer(&wgpu::BufferDescriptor { label: Some("crux nn readback"), size: (t2 * batch * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }))
                .collect();
            Ok((acts, binds, readback))
        })?;
        let out_act = m.acts[m.output.buf];
        let out_off = (m.output.c_off * t2 * 4) as u64;
        let out_stride = (out_act.len() * 4) as u64;
        let mut input = vec![0f32; in_len * batch];
        // Submit tiles start..start + n into readback slot `slot`.
        let mut submit = |start: usize, n: usize, slot: usize| -> Result<wgpu::SubmissionIndex> {
            for j in 0..n {
                fill(start + j, &mut input[j * in_len..(j + 1) * in_len]);
            }
            gpu.scoped(|| {
                gpu.queue.write_buffer(&acts[0], 0, bytemuck::cast_slice(&input[..n * in_len]));
                let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("crux nn") });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("crux nn"), timestamp_writes: None });
                    for (pipe, bg, groups, _) in &binds {
                        pass.set_pipeline(pipe);
                        pass.set_bind_group(0, bg, &[]);
                        pass.dispatch_workgroups(groups.0, groups.1, n as u32);
                    }
                }
                for j in 0..n as u64 {
                    enc.copy_buffer_to_buffer(&acts[m.output.buf], j * out_stride + out_off, &readback[slot], j * t2 as u64 * 4, t2 as u64 * 4);
                }
                Ok(gpu.queue.submit([enc.finish()]))
            })
        };
        // The first tile alone, timed: later submissions hold as many tiles
        // as take about a quarter of a second, so a slow card (integrated
        // graphics) never works on one long enough for Windows to reset its
        // driver (2 s).
        let mut allowed = 1usize;
        let started = std::time::Instant::now();
        let mut pending = Some((submit(0, 1, 0)?, 0usize, 1usize, 0usize));
        while let Some((idx, start, n, slot)) = pending.take() {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = gpu.wait();
                bail!("cancelled");
            }
            let next = start + n;
            if next < tiles {
                let n2 = allowed.min(tiles - next);
                pending = Some((submit(next, n2, 1 - slot)?, next, n2, 1 - slot));
            }
            let bytes = (n * t2 * 4) as u64;
            let (tx, rx) = std::sync::mpsc::channel();
            readback[slot].slice(..bytes).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            gpu.wait_for(idx)?;
            rx.recv().context("the GPU dropped a readback")?.map_err(|e| anyhow::anyhow!("read back from the GPU: {e}"))?;
            {
                let view = readback[slot].slice(..bytes).get_mapped_range().map_err(|e| anyhow::anyhow!("{e:?}"))?;
                let out: &[f32] = bytemuck::cast_slice(&view);
                for j in 0..n {
                    done(start + j, &out[j * t2..(j + 1) * t2])?;
                }
            }
            readback[slot].unmap();
            if start == 0 {
                let per_tile = started.elapsed().as_secs_f64().max(1e-4);
                allowed = ((0.25 / per_tile) as usize).clamp(1, batch);
                if crate::timing::on() && per_tile > 0.5 {
                    eprintln!("GPU: {per_tile:.2} s for one tile; one tile at a time");
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_models_parse() {
        for kind in [crate::render::net::Kind::Denoise, crate::render::net::Kind::Sharpen, crate::render::net::Kind::Starless] {
            let Some((onnx, json, _)) = crate::render::net::builtin(kind) else { continue };
            let meta: crate::render::net::Meta = serde_json::from_str(json).unwrap();
            let m = Model::parse(onnx, meta.channels(), 64).unwrap();
            assert!(m.ops.len() >= 15, "{kind:?}: {} ops", m.ops.len());
        }
    }
}
