//! Canon CR2 raw files (EOS DSLRs, 2004–2018).
//!
//! A CR2 is a little-endian TIFF: IFD 0 holds the model and the EXIF
//! pointer (exposure, ISO, local date and the Canon MakerNote), IFD 3 the
//! raw data as one lossless JPEG (ITU T.81 process 14, predictor 1, 12 or
//! 14 bits) cut into vertical slices (tag 0xC640: `[n, width, last width]`).
//! The MakerNote's SensorInfo (0x00E0) gives the active area inside the
//! masked borders, ShotInfo (0x0004) the camera temperature.
//!
//! Frames come out as raw Bayer mosaics of the active area, scaled to 16
//! bits (×4 for 14-bit data, as ZWO cameras write theirs), with the black
//! level left in like any raw FITS. The header is synthesised with the
//! FITS keywords crux reads: INSTRUME, EXPTIME, GAIN (= ISO), ISOSPEED,
//! DATE-LOC (the camera's clock), CCD-TEMP, BAYERPAT, XBINNING/YBINNING.

use crate::header::Header;
use crate::image::Image;
use anyhow::{bail, Context, Result};
use std::io::Read;
use std::path::Path;

/// What the header pass finds.
#[derive(Debug, Clone)]
pub struct Layout {
    pub header: Header,
    /// Active area (output size).
    pub width: u32,
    pub height: u32,
    /// Active area inside the sensor: left, top (inclusive).
    pub left: u32,
    pub top: u32,
    /// Raw JPEG strip.
    pub strip: (usize, usize),
    /// Slices: `[n, width, last width]` (n = 0: no slicing).
    pub slices: [u32; 3],
    pub raw_width: u32,
    pub raw_height: u32,
}

struct Tiff<'a> {
    d: &'a [u8],
}

#[derive(Clone, Copy)]
struct Entry {
    typ: u16,
    count: u32,
    value: u32,
    /// Where the 4-byte value field sits (inline values live there).
    at: usize,
}

impl<'a> Tiff<'a> {
    fn u16(&self, o: usize) -> Result<u16> {
        self.d.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]])).context("CR2: offset past the end")
    }
    fn u32(&self, o: usize) -> Result<u32> {
        self.d.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).context("CR2: offset past the end")
    }
    fn ifd(&self, off: usize) -> Result<(Vec<(u16, Entry)>, usize)> {
        let n = self.u16(off)? as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let e = off + 2 + 12 * i;
            out.push((self.u16(e)?, Entry { typ: self.u16(e + 2)?, count: self.u32(e + 4)?, value: self.u32(e + 8)?, at: e + 8 }));
        }
        Ok((out, self.u32(off + 2 + 12 * n)? as usize))
    }
    fn size(typ: u16) -> usize {
        match typ {
            3 | 8 => 2,
            4 | 9 | 11 => 4,
            5 | 10 | 12 => 8,
            _ => 1,
        }
    }
    fn data(&self, e: &Entry) -> Result<&'a [u8]> {
        let n = Self::size(e.typ) * e.count as usize;
        let start = if n <= 4 { e.at } else { e.value as usize };
        self.d.get(start..start + n).context("CR2: value past the end")
    }
    fn string(&self, e: &Entry) -> Result<String> {
        let b = self.data(e)?;
        let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
        Ok(String::from_utf8_lossy(&b[..end]).trim().to_string())
    }
    fn shorts(&self, e: &Entry) -> Result<Vec<u16>> {
        Ok(self.data(e)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }
    fn first_uint(&self, e: &Entry) -> Result<u32> {
        Ok(match e.typ {
            3 => self.shorts(e)?.first().copied().unwrap_or(0) as u32,
            _ => e.value,
        })
    }
    fn rational(&self, e: &Entry) -> Result<f64> {
        let b = self.data(e)?;
        if b.len() < 8 {
            bail!("CR2: short rational");
        }
        let (n, d) = (u32::from_le_bytes([b[0], b[1], b[2], b[3]]), u32::from_le_bytes([b[4], b[5], b[6], b[7]]));
        Ok(if d == 0 { 0.0 } else { n as f64 / d as f64 })
    }
}

fn find(ifd: &[(u16, Entry)], tag: u16) -> Option<Entry> {
    ifd.iter().find(|(t, _)| *t == tag).map(|(_, e)| *e)
}

/// The Bayer pattern of the sensor's even rows and columns, by model.
/// Canon EOS sensors read R G / G B from the active area's origin (both
/// borders are even); checked on the EOS 80D by the green mismatch and
/// red H-alpha (`crux dev cfa`).
fn pattern(_model: &str) -> &'static str {
    "RGGB"
}

/// Parse the header from the start of a CR2 file.
pub fn parse_layout(d: &[u8]) -> Result<Layout> {
    if d.len() < 16 || &d[0..4] != b"II*\0" || &d[8..10] != b"CR" {
        bail!("not a Canon CR2 file");
    }
    let t = Tiff { d };
    let (ifd0, mut next) = t.ifd(t.u32(4)? as usize)?;
    let model = find(&ifd0, 0x0110).map(|e| t.string(&e)).transpose()?.unwrap_or_default();
    let mut h = Header::default();
    let mut exposure = None;
    let mut iso = None;
    let mut date = None;
    let mut temp = None;
    let mut sensor: Option<Vec<u16>> = None;
    if let Some(exif) = find(&ifd0, 0x8769) {
        let (ex, _) = t.ifd(exif.value as usize)?;
        exposure = find(&ex, 0x829A).map(|e| t.rational(&e)).transpose()?;
        iso = find(&ex, 0x8827).map(|e| t.first_uint(&e)).transpose()?;
        date = find(&ex, 0x9003).map(|e| t.string(&e)).transpose()?;
        if let Some(mn) = find(&ex, 0x927C) {
            // Canon's MakerNote is a plain IFD; offsets are from the TIFF start.
            if let Ok((m, _)) = t.ifd(mn.value as usize) {
                sensor = find(&m, 0x00E0).map(|e| t.shorts(&e)).transpose()?;
                if let Some(si) = find(&m, 0x0004).map(|e| t.shorts(&e)).transpose()? {
                    // ShotInfo[12]: camera temperature + 128 (0 = not recorded).
                    if let Some(&v) = si.get(12) {
                        if v > 0 && v < 256 {
                            temp = Some(v as i32 - 128);
                        }
                    }
                }
            }
        }
    }
    // IFD 3 holds the raw data.
    let mut raw = None;
    for _ in 0..3 {
        if next == 0 {
            break;
        }
        let (ifd, n) = t.ifd(next)?;
        raw = Some(ifd);
        next = n;
    }
    let raw = raw.context("CR2: no raw IFD")?;
    let strip = (
        find(&raw, 0x0111).map(|e| t.first_uint(&e)).transpose()?.context("CR2: no raw strip")? as usize,
        find(&raw, 0x0117).map(|e| t.first_uint(&e)).transpose()?.context("CR2: no raw strip length")? as usize,
    );
    let mut slices = [0u32; 3];
    if let Some(e) = find(&raw, 0xC640) {
        let s = t.shorts(&e)?;
        if s.len() >= 3 {
            slices = [s[0] as u32, s[1] as u32, s[2] as u32];
        }
    }
    let raw_width = find(&raw, 0x0100).map(|e| t.first_uint(&e)).transpose()?.unwrap_or(0);
    let raw_height = find(&raw, 0x0101).map(|e| t.first_uint(&e)).transpose()?.unwrap_or(0);
    // SensorInfo: [size, width, height, ?, ?, left, top, right, bottom, …].
    let (left, top, width, height) = match sensor.as_deref() {
        Some(s) if s.len() >= 9 && s[7] > s[5] && s[8] > s[6] => (s[5] as u32, s[6] as u32, (s[7] - s[5]) as u32 + 1, (s[8] - s[6]) as u32 + 1),
        _ => (0, 0, raw_width, raw_height),
    };
    if raw_width > 0 && (left + width > raw_width || top + height > raw_height) {
        bail!("CR2: active area {width}×{height}+{left}+{top} outside the {raw_width}×{raw_height} sensor");
    }

    if !model.is_empty() {
        h.push("INSTRUME", model.clone(), "Camera model");
    }
    if let Some(e) = exposure {
        h.push("EXPTIME", format!("{e}"), "Exposure time in seconds");
    }
    if let Some(i) = iso {
        h.push("ISOSPEED", i.to_string(), "ISO");
        h.push("GAIN", i.to_string(), "ISO (DSLR)");
    }
    if let Some(d) = date.as_deref().and_then(exif_date) {
        h.push("DATE-LOC", d, "Camera clock (local time)");
    }
    if let Some(c) = temp {
        h.push("CCD-TEMP", c.to_string(), "Camera temperature in C");
    }
    h.push("BAYERPAT", pattern(&model), "Bayer pattern");
    h.push("XBINNING", "1", "");
    h.push("YBINNING", "1", "");
    Ok(Layout { header: h, width, height, left, top, strip, slices, raw_width, raw_height })
}

/// "2026:01:23 21:58:01" → "2026-01-23T21:58:01".
fn exif_date(s: &str) -> Option<String> {
    let b = s.as_bytes();
    if b.len() < 19 || b[4] != b':' || b[7] != b':' {
        return None;
    }
    Some(format!("{}-{}-{}T{}", &s[0..4], &s[5..7], &s[8..10], &s[11..19]))
}

/// Header pass: the metadata is in the first few tens of KB.
pub fn read_layout(path: &Path) -> Result<Layout> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut buf = Vec::new();
    f.by_ref().take(256 * 1024).read_to_end(&mut buf)?;
    match parse_layout(&buf) {
        Ok(l) => Ok(l),
        // Unusual files keep their tags further in.
        Err(_) => parse_layout(&std::fs::read(path)?).with_context(|| format!("CR2 header of {}", path.display())),
    }
}

pub fn read_image(path: &Path) -> Result<Image> {
    let d = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    decode(&d).with_context(|| format!("CR2 {}", path.display()))
}

/// The active area, scaled to 16 bits and normalised to [0, 1].
pub fn decode(d: &[u8]) -> Result<Image> {
    let l = parse_layout(d)?;
    let jpeg = d.get(l.strip.0..l.strip.0 + l.strip.1).context("raw strip past the end")?;
    let j = ljpeg_decode(jpeg)?;
    let total = j.samples.len();
    let (sw, sh) = if l.raw_width > 0 && l.raw_height > 0 { (l.raw_width as usize, l.raw_height as usize) } else { (j.width * j.comps, j.height) };
    if total != sw * sh {
        bail!("lossless JPEG holds {total} samples, sensor is {sw}×{sh}");
    }
    let scale = (1u32 << (16 - j.precision.min(16))) as f32 / 65535.0;
    let (w, h) = (l.width as usize, l.height as usize);
    let (x0, y0) = (l.left as usize, l.top as usize);
    let mut out = vec![0f32; w * h];
    // Samples arrive slice by slice, each slice row by row (dcraw's mapping).
    let [n, sw0, swl] = l.slices.map(|v| v as usize);
    let slice_px = sw0 * sh;
    for (k, &v) in j.samples.iter().enumerate() {
        let (row, col) = if n == 0 || sw0 == 0 {
            (k / sw, k % sw)
        } else {
            let i = (k / slice_px).min(n);
            let r = k - i * slice_px;
            let width = if i < n { sw0 } else { swl };
            (r / width, r % width + i * sw0)
        };
        if row >= y0 && row < y0 + h && col >= x0 && col < x0 + w {
            out[(row - y0) * w + (col - x0)] = v as f32 * scale;
        }
    }
    Ok(Image { width: w, height: h, data: out, adu_scale: 65535.0 })
}

/// Black level of a decoded raw: median of the masked left border.
/// (For checks; calibration takes it out with the bias.)
pub fn black_level(d: &[u8]) -> Result<f64> {
    let l = parse_layout(d)?;
    let jpeg = d.get(l.strip.0..l.strip.0 + l.strip.1).context("raw strip past the end")?;
    let j = ljpeg_decode(jpeg)?;
    let sw = l.raw_width as usize;
    let [n, sw0, swl] = l.slices.map(|v| v as usize);
    let sh = l.raw_height as usize;
    let mut v = Vec::new();
    for (k, &s) in j.samples.iter().enumerate() {
        let (row, col) = if n == 0 || sw0 == 0 {
            (k / sw, k % sw)
        } else {
            let i = (k / (sw0 * sh)).min(n);
            let r = k - i * sw0 * sh;
            let width = if i < n { sw0 } else { swl };
            (r / width, r % width + i * sw0)
        };
        if col >= 8 && (col as u32) + 8 < l.left && row as u32 >= l.top && k % 7 == 0 {
            v.push(s);
        }
    }
    if v.is_empty() {
        bail!("no masked border");
    }
    v.sort_unstable();
    Ok(v[v.len() / 2] as f64 * (1u32 << (16 - j.precision.min(16))) as f64)
}

// ---------------------------------------------------------------------
// Lossless JPEG (process 14), as Canon writes it: one scan, predictor 1,
// no restart markers.

pub struct Ljpeg {
    pub width: usize,
    pub height: usize,
    pub comps: usize,
    pub precision: u32,
    /// Row-major, components interleaved.
    pub samples: Vec<u16>,
}

/// 16-bit lookup: next 16 bits → (code length, category).
struct Huff {
    lut: Vec<(u8, u8)>,
}

impl Huff {
    fn new(counts: &[u8; 16], values: &[u8]) -> Result<Huff> {
        let mut lut = vec![(0u8, 0u8); 1 << 16];
        let (mut code, mut k) = (0u32, 0usize);
        for (i, &c) in counts.iter().enumerate() {
            let len = i as u32 + 1;
            for _ in 0..c {
                let v = *values.get(k).context("Huffman table: too few values")?;
                k += 1;
                let shift = 16 - len;
                let start = (code << shift) as usize;
                let end = start + (1usize << shift);
                if end > lut.len() {
                    bail!("Huffman table: bad code lengths");
                }
                for e in &mut lut[start..end] {
                    *e = (len as u8, v);
                }
                code += 1;
            }
            code <<= 1;
        }
        Ok(Huff { lut })
    }
}

struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
}

impl<'a> Bits<'a> {
    #[inline]
    fn fill(&mut self) {
        while self.n <= 56 {
            let mut b = 0u8;
            if self.pos < self.d.len() {
                b = self.d[self.pos];
                if b == 0xFF {
                    match self.d.get(self.pos + 1) {
                        Some(0x00) => self.pos += 2,
                        // A marker: the scan is over; feed zeros.
                        _ => b = 0,
                    }
                } else {
                    self.pos += 1;
                }
            }
            self.acc |= (b as u64) << (56 - self.n);
            self.n += 8;
        }
    }
    #[inline]
    fn peek16(&mut self) -> u32 {
        self.fill();
        (self.acc >> 48) as u32
    }
    #[inline]
    fn skip(&mut self, k: u32) {
        self.acc <<= k;
        self.n -= k;
    }
    #[inline]
    fn get(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        self.fill();
        let v = (self.acc >> (64 - k)) as u32;
        self.skip(k);
        v
    }
    #[inline]
    fn diff(&mut self, h: &Huff) -> Result<i32> {
        let (len, s) = h.lut[self.peek16() as usize];
        if len == 0 {
            bail!("lossless JPEG: bad Huffman code");
        }
        self.skip(len as u32);
        Ok(match s {
            0 => 0,
            16 => 32768,
            s => {
                let s = s as u32;
                let v = self.get(s) as i32;
                if v < 1 << (s - 1) {
                    v - (1 << s) + 1
                } else {
                    v
                }
            }
        })
    }
}

pub fn ljpeg_decode(d: &[u8]) -> Result<Ljpeg> {
    if d.len() < 4 || d[0] != 0xFF || d[1] != 0xD8 {
        bail!("lossless JPEG: no SOI");
    }
    let mut tables: [Option<Huff>; 4] = [None, None, None, None];
    let (mut width, mut height, mut comps, mut precision) = (0usize, 0usize, 0usize, 0u32);
    let mut p = 2;
    loop {
        if p + 4 > d.len() {
            bail!("lossless JPEG: ran out before the scan");
        }
        if d[p] != 0xFF {
            p += 1;
            continue;
        }
        let marker = d[p + 1];
        let len = u16::from_be_bytes([d[p + 2], d[p + 3]]) as usize;
        let seg = d.get(p + 4..p + 2 + len).context("lossless JPEG: segment past the end")?;
        match marker {
            0xC4 => {
                let mut q = 0;
                while q + 17 <= seg.len() {
                    let id = (seg[q] & 0x0F) as usize;
                    let mut counts = [0u8; 16];
                    counts.copy_from_slice(&seg[q + 1..q + 17]);
                    let n: usize = counts.iter().map(|&c| c as usize).sum();
                    let values = seg.get(q + 17..q + 17 + n).context("lossless JPEG: short DHT")?;
                    if id < 4 {
                        tables[id] = Some(Huff::new(&counts, values)?);
                    }
                    q += 17 + n;
                }
            }
            0xC3 => {
                precision = seg[0] as u32;
                height = u16::from_be_bytes([seg[1], seg[2]]) as usize;
                width = u16::from_be_bytes([seg[3], seg[4]]) as usize;
                comps = seg[5] as usize;
            }
            0xC0..=0xC2 | 0xC5..=0xCF if marker != 0xC8 && marker != 0xCC => bail!("not a lossless JPEG (SOF{:X})", marker - 0xC0),
            0xDA => {
                let ns = seg[0] as usize;
                if comps == 0 || ns != comps {
                    bail!("lossless JPEG: scan has {ns} components, frame {comps}");
                }
                let mut sel = Vec::with_capacity(ns);
                for c in 0..ns {
                    let id = (seg[2 + 2 * c] >> 4) as usize;
                    sel.push(tables.get(id).and_then(|t| t.as_ref()).context("lossless JPEG: missing Huffman table")?);
                }
                let predictor = seg[1 + 2 * ns];
                let pt = (seg[3 + 2 * ns] & 0x0F) as u32;
                if predictor != 1 {
                    bail!("lossless JPEG: predictor {predictor} not supported");
                }
                let row = width * comps;
                let mut out = vec![0u16; row * height];
                let mut bits = Bits { d, pos: p + 2 + len, acc: 0, n: 0 };
                let init = 1i32 << (precision - pt - 1);
                for y in 0..height {
                    let base = y * row;
                    for x in 0..width {
                        for (c, h) in sel.iter().enumerate() {
                            let i = base + x * comps + c;
                            let pred = if x > 0 {
                                out[i - comps] as i32
                            } else if y > 0 {
                                out[i - row] as i32
                            } else {
                                init
                            };
                            out[i] = (pred + bits.diff(h)?) as u16;
                        }
                    }
                }
                if pt > 0 {
                    for v in &mut out {
                        *v <<= pt;
                    }
                }
                return Ok(Ljpeg { width, height, comps, precision, samples: out });
            }
            _ => {}
        }
        p += 2 + len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny lossless-JPEG encoder: every category gets a 5-bit code.
    fn encode(samples: &[u16], width: usize, height: usize, comps: usize, precision: u8) -> Vec<u8> {
        let mut o = vec![0xFF, 0xD8];
        // DHT: 17 categories, all length 5.
        let mut counts = [0u8; 16];
        counts[4] = 17;
        let mut dht = vec![0x00];
        dht.extend_from_slice(&counts);
        dht.extend(0u8..17);
        o.extend_from_slice(&[0xFF, 0xC4]);
        o.extend_from_slice(&((dht.len() + 2) as u16).to_be_bytes());
        o.extend_from_slice(&dht);
        let mut sof = vec![precision];
        sof.extend_from_slice(&(height as u16).to_be_bytes());
        sof.extend_from_slice(&(width as u16).to_be_bytes());
        sof.push(comps as u8);
        for c in 0..comps {
            sof.extend_from_slice(&[c as u8 + 1, 0x11, 0]);
        }
        o.extend_from_slice(&[0xFF, 0xC3]);
        o.extend_from_slice(&((sof.len() + 2) as u16).to_be_bytes());
        o.extend_from_slice(&sof);
        let mut sos = vec![comps as u8];
        for c in 0..comps {
            sos.extend_from_slice(&[c as u8 + 1, 0x00]);
        }
        sos.extend_from_slice(&[1, 0, 0]);
        o.extend_from_slice(&[0xFF, 0xDA]);
        o.extend_from_slice(&((sos.len() + 2) as u16).to_be_bytes());
        o.extend_from_slice(&sos);
        // Entropy-coded data.
        let (mut acc, mut n, mut bytes) = (0u64, 0u32, Vec::new());
        let put = |v: u32, k: u32, acc: &mut u64, n: &mut u32, bytes: &mut Vec<u8>| {
            *acc = (*acc << k) | v as u64;
            *n += k;
            while *n >= 8 {
                let b = (*acc >> (*n - 8)) as u8;
                bytes.push(b);
                if b == 0xFF {
                    bytes.push(0);
                }
                *n -= 8;
            }
        };
        let row = width * comps;
        for y in 0..height {
            for x in 0..width {
                for c in 0..comps {
                    let i = y * row + x * comps + c;
                    let pred = if x > 0 { samples[i - comps] as i32 } else if y > 0 { samples[i - row] as i32 } else { 1 << (precision - 1) };
                    let diff = samples[i] as i32 - pred;
                    let s = if diff == 0 { 0 } else { 32 - diff.unsigned_abs().leading_zeros() };
                    put(s, 5, &mut acc, &mut n, &mut bytes);
                    if s > 0 {
                        let v = if diff < 0 { (diff - 1) as u32 & ((1 << s) - 1) } else { diff as u32 };
                        put(v, s, &mut acc, &mut n, &mut bytes);
                    }
                }
            }
        }
        if n > 0 {
            let pad = 8 - n;
            put((1 << pad) - 1, pad, &mut acc, &mut n, &mut bytes);
        }
        o.extend_from_slice(&bytes);
        o.extend_from_slice(&[0xFF, 0xD9]);
        o
    }

    #[test]
    fn lossless_jpeg_round_trip() {
        let (w, h, c) = (7, 5, 2);
        let s: Vec<u16> = (0..w * h * c).map(|i| ((i * 7919) % 16384) as u16).collect();
        let j = ljpeg_decode(&encode(&s, w, h, c, 14)).unwrap();
        assert_eq!((j.width, j.height, j.comps, j.precision), (w, h, c, 14));
        assert_eq!(j.samples, s);
    }

    /// A minimal CR2: IFD 0 (model, EXIF with exposure, ISO, date and a
    /// MakerNote with SensorInfo and ShotInfo), IFDs 1–2 empty, IFD 3 the
    /// sliced raw.
    fn build_cr2(sensor: &[u16], sw: usize, sh: usize, slices: [u16; 3], area: [u16; 4]) -> Vec<u8> {
        // Slice the sensor the way Canon does: slice by slice, row by row.
        let (n, w0, wl) = (slices[0] as usize, slices[1] as usize, slices[2] as usize);
        let mut stream = Vec::with_capacity(sw * sh);
        for i in 0..=n {
            let width = if i < n { w0 } else { wl };
            for y in 0..sh {
                for x in 0..width {
                    stream.push(sensor[y * sw + i * w0 + x]);
                }
            }
        }
        // JPEG frame: 2 components, width × 2 = sensor width.
        let jpeg = encode(&stream, sw / 2, sh, 2, 14);

        let mut f = vec![0u8; 16];
        f[0..4].copy_from_slice(b"II*\0");
        f[4..8].copy_from_slice(&16u32.to_le_bytes());
        f[8..10].copy_from_slice(b"CR");
        f[10] = 2;
        // Lay out blobs after a fixed 4 KB of IFDs.
        let mut blobs: Vec<u8> = Vec::new();
        let base = 4096usize;
        let mut blob = |b: &[u8]| -> u32 {
            let off = base + blobs.len();
            blobs.extend_from_slice(b);
            while blobs.len() % 2 == 1 {
                blobs.push(0);
            }
            off as u32
        };
        let model = blob(b"Canon EOS Test\0");
        let expo = blob(&[&120u32.to_le_bytes()[..], &1u32.to_le_bytes()[..]].concat());
        let date = blob(b"2026:01:23 21:58:01\0");
        let si: Vec<u8> = [34, sw as u16, sh as u16, 1, 1, area[0], area[1], area[2], area[3]].iter().flat_map(|v: &u16| v.to_le_bytes()).collect();
        let si = blob(&si);
        let mut shot = vec![0u16; 34];
        shot[12] = 128 + 21;
        let shot: Vec<u8> = shot.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shot = blob(&shot);
        let slc: Vec<u8> = slices.iter().flat_map(|v| v.to_le_bytes()).collect();
        let slc = blob(&slc);
        let strip = blob(&jpeg);

        let mut ifds = vec![0u8; base - 16];
        let write_ifd = |at: usize, entries: &[(u16, u16, u32, u32)], next: u32, ifds: &mut Vec<u8>| {
            let o = at - 16;
            ifds[o..o + 2].copy_from_slice(&(entries.len() as u16).to_le_bytes());
            for (k, &(tag, typ, count, val)) in entries.iter().enumerate() {
                let e = o + 2 + 12 * k;
                ifds[e..e + 2].copy_from_slice(&tag.to_le_bytes());
                ifds[e + 2..e + 4].copy_from_slice(&typ.to_le_bytes());
                ifds[e + 4..e + 8].copy_from_slice(&count.to_le_bytes());
                ifds[e + 8..e + 12].copy_from_slice(&val.to_le_bytes());
            }
            let e = o + 2 + 12 * entries.len();
            ifds[e..e + 4].copy_from_slice(&next.to_le_bytes());
        };
        let (i0, ex, mn, i1, i2, i3) = (16, 512, 1024, 1536, 1600, 1664);
        write_ifd(i0, &[(0x0110, 2, 15, model), (0x8769, 4, 1, ex as u32)], i1 as u32, &mut ifds);
        write_ifd(ex, &[(0x829A, 5, 1, expo), (0x8827, 3, 1, 800), (0x9003, 2, 20, date), (0x927C, 7, 100, mn as u32)], 0, &mut ifds);
        write_ifd(mn, &[(0x0004, 3, 34, shot), (0x00E0, 3, 9, si)], 0, &mut ifds);
        write_ifd(i1, &[], i2 as u32, &mut ifds);
        write_ifd(i2, &[], i3 as u32, &mut ifds);
        write_ifd(
            i3,
            &[(0x0100, 3, 1, sw as u32), (0x0101, 3, 1, sh as u32), (0x0111, 4, 1, strip), (0x0117, 4, 1, jpeg.len() as u32), (0xC640, 3, 3, slc)],
            0,
            &mut ifds,
        );
        f.extend_from_slice(&ifds);
        f.extend_from_slice(&blobs);
        f
    }

    #[test]
    fn cr2_slices_crop_and_header() {
        let (sw, sh) = (24, 10);
        let sensor: Vec<u16> = (0..sw * sh).map(|i| (1000 + (i * 37) % 9000) as u16).collect();
        // Two slices of 10 and a last one of 4; active area x 4..=19, y 2..=8.
        let d = build_cr2(&sensor, sw, sh, [2, 10, 4], [4, 2, 19, 8]);
        let l = parse_layout(&d).unwrap();
        assert_eq!((l.width, l.height, l.left, l.top), (16, 7, 4, 2));
        let h = &l.header;
        assert_eq!(h.get("INSTRUME"), Some("Canon EOS Test"));
        assert_eq!(h.f64("EXPTIME"), Some(120.0));
        assert_eq!(h.f64("GAIN"), Some(800.0));
        assert_eq!(h.get("DATE-LOC"), Some("2026-01-23T21:58:01"));
        assert_eq!(h.f64("CCD-TEMP"), Some(21.0));
        assert_eq!(h.get("BAYERPAT"), Some("RGGB"));
        let img = decode(&d).unwrap();
        assert_eq!((img.width, img.height), (16, 7));
        for y in 0..7 {
            for x in 0..16 {
                let want = sensor[(y + 2) * sw + x + 4] as f32 * 4.0 / 65535.0;
                assert!((img.data[y * 16 + x] - want).abs() < 1e-6, "({x},{y})");
            }
        }
    }
}
