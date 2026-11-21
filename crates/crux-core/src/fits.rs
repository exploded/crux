//! Minimal pure-Rust FITS reader/writer for the primary HDU.
//!
//! Supports BITPIX 8/16/32/-32/-64 with BZERO/BSCALE. Header-only reads
//! stop at the END card, so a 50 MB sub costs one or two 2880-byte blocks.

use crate::header::{clean_value, Card, Header};
use crate::image::Image;
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const BLOCK: usize = 2880;
const CARD: usize = 80;

/// Header plus where and how the pixel data is stored.
#[derive(Debug, Clone)]
pub struct FitsLayout {
    pub header: Header,
    pub data_offset: u64,
    pub bitpix: i32,
    pub naxis: Vec<usize>,
    pub bzero: f64,
    pub bscale: f64,
}

impl FitsLayout {
    pub fn width(&self) -> usize {
        self.naxis.first().copied().unwrap_or(0)
    }
    pub fn height(&self) -> usize {
        self.naxis.get(1).copied().unwrap_or(0)
    }
    pub fn planes(&self) -> usize {
        self.naxis.iter().skip(2).product::<usize>().max(1)
    }
}

pub fn read_layout(path: &Path) -> Result<FitsLayout> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    read_layout_from(&mut f).with_context(|| format!("FITS header of {}", path.display()))
}

pub fn read_layout_from<R: Read>(r: &mut R) -> Result<FitsLayout> {
    let mut header = Header::default();
    let mut block = [0u8; BLOCK];
    let mut blocks = 0u64;
    let mut ended = false;
    while !ended {
        r.read_exact(&mut block).context("truncated FITS header")?;
        if blocks == 0 && &block[..9] != b"SIMPLE  =" {
            bail!("not a FITS file (no SIMPLE card)");
        }
        blocks += 1;
        if blocks > 2000 {
            bail!("FITS header has no END card");
        }
        for card in block.chunks_exact(CARD) {
            if &card[..8] == b"END     " {
                ended = true;
                break;
            }
            if let Some(c) = parse_card(card) {
                // CONTINUE long-string convention.
                if c.key == "CONTINUE" {
                    if let Some(last) = header.cards.last_mut() {
                        if last.value.ends_with('&') {
                            last.value.pop();
                            last.value.push_str(&c.value);
                            continue;
                        }
                    }
                }
                header.cards.push(c);
            }
        }
    }
    let bitpix = header.i64("BITPIX").context("missing BITPIX")? as i32;
    let n = header.i64("NAXIS").unwrap_or(0) as usize;
    let naxis = (1..=n)
        .map(|i| header.i64(&format!("NAXIS{i}")).map(|v| v as usize).context("missing NAXISn"))
        .collect::<Result<Vec<_>>>()?;
    Ok(FitsLayout {
        bzero: header.f64("BZERO").unwrap_or(0.0),
        bscale: header.f64("BSCALE").unwrap_or(1.0),
        header,
        data_offset: blocks * BLOCK as u64,
        bitpix,
        naxis,
    })
}

/// Parse one 80-byte card. Returns `None` for blank cards.
pub fn parse_card(card: &[u8]) -> Option<Card> {
    let text: String = card.iter().map(|&b| if (32..127).contains(&b) { b as char } else { ' ' }).collect();
    let key = text[..8].trim().to_string();
    if key.is_empty() && text.trim().is_empty() {
        return None;
    }
    if key == "HIERARCH" {
        let rest = &text[8..];
        let eq = rest.find('=')?;
        let (val, comment) = split_value_comment(&rest[eq + 1..]);
        return Some(Card { key: rest[..eq].trim().to_string(), value: val, comment });
    }
    if key == "CONTINUE" {
        let (val, comment) = split_value_comment(&text[8..]);
        return Some(Card { key, value: val, comment });
    }
    if &text[8..10] != "= " {
        // COMMENT, HISTORY and other commentary cards.
        return Some(Card { key, value: text[8..].trim_end().to_string(), comment: String::new() });
    }
    let (value, comment) = split_value_comment(&text[10..]);
    Some(Card { key, value, comment })
}

fn split_value_comment(field: &str) -> (String, String) {
    let t = field.trim_start();
    if let Some(body) = t.strip_prefix('\'') {
        // Find the closing quote; '' is an escaped quote.
        let bytes = body.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                break;
            }
            i += 1;
        }
        let value = clean_value(&format!("'{}'", &body[..i.min(body.len())]));
        let rest = body.get(i + 1..).unwrap_or("");
        let comment = rest.split_once('/').map(|(_, c)| c.trim().to_string()).unwrap_or_default();
        (value, comment)
    } else {
        match t.split_once('/') {
            Some((v, c)) => (v.trim().to_string(), c.trim().to_string()),
            None => (t.trim().to_string(), String::new()),
        }
    }
}

/// Read the first image plane, normalised to [0, 1] for integer data.
pub fn read_image(path: &Path) -> Result<Image> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let layout = read_layout_from(&mut f).with_context(|| format!("FITS header of {}", path.display()))?;
    read_image_with(&mut f, &layout).with_context(|| format!("FITS data of {}", path.display()))
}

pub fn read_image_with<R: Read + Seek>(r: &mut R, l: &FitsLayout) -> Result<Image> {
    let (w, h) = (l.width(), l.height());
    if l.naxis.len() < 2 || w == 0 || h == 0 {
        bail!("primary HDU has no 2-D image (NAXIS={})", l.naxis.len());
    }
    let bpp = (l.bitpix.unsigned_abs() / 8) as usize;
    if ![1, 2, 4, 8].contains(&bpp) {
        bail!("unsupported BITPIX {}", l.bitpix);
    }
    r.seek(SeekFrom::Start(l.data_offset))?;
    let n = w * h;
    let mut out = vec![0f32; n];
    let (bzero, bscale) = (l.bzero, l.bscale);
    // Normalisation: integer data to [0,1] of a 16-bit (or 8-bit) range.
    let unsigned16 = l.bitpix == 16 && bzero == 32768.0 && bscale == 1.0;
    let norm: f64 = match l.bitpix {
        8 => 255.0,
        16 | 32 => 65535.0,
        _ => 1.0,
    };
    let rows_per_chunk = (4 << 20) / (w * bpp).max(1);
    let mut buf = vec![0u8; rows_per_chunk.max(1) * w * bpp];
    let mut y = 0;
    while y < h {
        let rows = rows_per_chunk.max(1).min(h - y);
        let bytes = &mut buf[..rows * w * bpp];
        r.read_exact(bytes).context("truncated FITS data")?;
        let dst = &mut out[y * w..(y + rows) * w];
        match l.bitpix {
            8 => {
                for (d, &b) in dst.iter_mut().zip(bytes.iter()) {
                    *d = ((bzero + bscale * b as f64) / norm) as f32;
                }
            }
            16 if unsigned16 => {
                for (d, c) in dst.iter_mut().zip(bytes.chunks_exact(2)) {
                    let v = (i16::from_be_bytes([c[0], c[1]]) as i32 + 32768) as f32;
                    *d = v * (1.0 / 65535.0);
                }
            }
            16 => {
                for (d, c) in dst.iter_mut().zip(bytes.chunks_exact(2)) {
                    let v = i16::from_be_bytes([c[0], c[1]]) as f64;
                    *d = ((bzero + bscale * v) / norm) as f32;
                }
            }
            32 => {
                for (d, c) in dst.iter_mut().zip(bytes.chunks_exact(4)) {
                    let v = i32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f64;
                    *d = ((bzero + bscale * v) / norm) as f32;
                }
            }
            -32 => {
                for (d, c) in dst.iter_mut().zip(bytes.chunks_exact(4)) {
                    let v = f32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f64;
                    *d = (bzero + bscale * v) as f32;
                }
            }
            -64 => {
                for (d, c) in dst.iter_mut().zip(bytes.chunks_exact(8)) {
                    let v = f64::from_be_bytes(c.try_into().unwrap());
                    *d = (bzero + bscale * v) as f32;
                }
            }
            b => bail!("unsupported BITPIX {b}"),
        }
        y += rows;
    }
    let mut img = Image::from_data(w, h, out);
    img.adu_scale = if l.bitpix == 8 { 255.0 } else { 65535.0 };
    // Float data in ADU, or 32-bit ints spanning the full u32 range.
    if l.bitpix < 0 || l.bitpix == 32 {
        let max = img.data.iter().copied().filter(|v| v.is_finite()).fold(0f32, f32::max);
        if l.bitpix < 0 && max > 1.5 {
            img.data.iter_mut().for_each(|v| *v /= 65535.0);
        } else if l.bitpix == 32 && max > 1.5 {
            img.data.iter_mut().for_each(|v| *v *= (65535.0 / 4294967295.0) as f32);
        }
    }
    for v in img.data.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    Ok(img)
}

/// Write a 2-D primary HDU. `bitpix` is 16 (unsigned via BZERO) or -32.
pub fn write_image(path: &Path, img: &Image, bitpix: i32, extra: &Header) -> Result<()> {
    write_with(path, img.width, img.height, &[&img.data], bitpix, extra)
}

/// Write a Float32 (BITPIX −32) image of one plane (NAXIS 2) or three
/// (NAXIS3 = 3, planar R, G, B), values in [0, 1], first stored row first.
pub fn write_planes(path: &Path, width: usize, height: usize, planes: &[&[f32]], extra: &Header) -> Result<()> {
    anyhow::ensure!(planes.len() == 1 || planes.len() == 3, "FITS writer: 1 or 3 planes, not {}", planes.len());
    write_with(path, width, height, planes, -32, extra)
}

/// Write a Float32 cube: any number of same-size planes (NAXIS3 = count),
/// e.g. a stack of training patches.
pub fn write_cube(path: &Path, width: usize, height: usize, planes: &[&[f32]], extra: &Header) -> Result<()> {
    anyhow::ensure!(!planes.is_empty(), "FITS writer: no planes");
    write_with(path, width, height, planes, -32, extra)
}

fn write_with(path: &Path, width: usize, height: usize, planes: &[&[f32]], bitpix: i32, extra: &Header) -> Result<()> {
    if bitpix != 16 && bitpix != -32 {
        bail!("the FITS writer supports BITPIX 16 or -32");
    }
    for p in planes {
        anyhow::ensure!(p.len() == width * height, "FITS writer: plane size");
    }
    let mut cards: Vec<String> = vec![
        fmt_card("SIMPLE", "T", ""),
        fmt_card("BITPIX", &bitpix.to_string(), ""),
        fmt_card("NAXIS", if planes.len() > 1 { "3" } else { "2" }, ""),
        fmt_card("NAXIS1", &width.to_string(), ""),
        fmt_card("NAXIS2", &height.to_string(), ""),
    ];
    if planes.len() > 1 {
        cards.push(fmt_card("NAXIS3", &planes.len().to_string(), ""));
    }
    if bitpix == 16 {
        cards.push(fmt_card("BZERO", "32768", ""));
        cards.push(fmt_card("BSCALE", "1", ""));
    }
    let reserved = ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "BZERO", "BSCALE", "EXTEND", "END"];
    for c in &extra.cards {
        let key = c.key.trim().to_ascii_uppercase();
        if reserved.contains(&key.as_str()) {
            continue;
        }
        if key.is_empty() || key == "COMMENT" || key == "HISTORY" {
            // Commentary cards: no "= ", text from column 9.
            let text = if c.value.is_empty() { &c.comment } else { &c.value };
            let mut s = format!("{key:<8}{text}");
            s.truncate(CARD);
            cards.push(format!("{s:<80}"));
            continue;
        }
        // Standard keywords are at most 8 characters; anything longer
        // (XISF allows it) goes as a HIERARCH card.
        let looks_numeric = crate::header::parse_number(&c.value).is_some() || c.value == "T" || c.value == "F";
        // The standard's exponent letter is upper case (1.86E-4).
        let v = if looks_numeric { c.value.trim().replace('e', "E") } else { format!("'{}'", c.value.replace('\'', "''")) };
        if key.len() > 8 || !key.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'-' || b == b'_') {
            let mut s = format!("HIERARCH {key} = {v}");
            if !c.comment.is_empty() {
                s.push_str(" / ");
                s.push_str(&c.comment);
            }
            s.truncate(CARD);
            cards.push(format!("{s:<80}"));
        } else {
            cards.push(fmt_card(&key, &v, &c.comment));
        }
    }
    cards.push(format!("{:<80}", "END"));
    let mut w = BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    let mut written = 0usize;
    for c in &cards {
        w.write_all(c.as_bytes())?;
        written += CARD;
    }
    pad(&mut w, &mut written, b' ')?;
    for p in planes {
        for &v in p.iter() {
            match bitpix {
                16 => {
                    let u = (v.clamp(0.0, 1.0) * 65535.0).round() as i32 - 32768;
                    w.write_all(&(u as i16).to_be_bytes())?;
                    written += 2;
                }
                _ => {
                    w.write_all(&v.to_be_bytes())?;
                    written += 4;
                }
            }
        }
    }
    pad(&mut w, &mut written, 0)?;
    w.flush()?;
    Ok(())
}

/// The FITS copy's path beside an XISF output: same name, `.fits`.
pub fn copy_path(xisf: &Path) -> std::path::PathBuf {
    xisf.with_extension("fits")
}

/// Write `<name>.fits` beside a mono XISF output (`[process] fits`)
/// unless one at least as new is already there. Keywords are carried
/// over; XISF properties aren't (FITS has nowhere to put them). Written
/// to a temporary name and renamed into place.
pub fn ensure_copy(xisf: &Path) -> Result<std::path::PathBuf> {
    let out = copy_path(xisf);
    let mtime = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    if let (Some(src), Some(dst)) = (mtime(xisf), mtime(&out)) {
        if dst >= src {
            return Ok(out);
        }
    }
    let l = crate::xisf::read_layout(xisf).with_context(|| format!("XISF header of {}", xisf.display()))?;
    anyhow::ensure!(l.channels <= 1, "{}: FITS copies of colour XISF files aren't supported", xisf.display());
    let img = crate::xisf::read_image(xisf)?;
    // Keywords only: the properties the XISF reader maps onto FITS names
    // were never keywords in crux's own files.
    let mut h = Header::default();
    for c in l.header.cards.iter().filter(|c| c.comment != "from XISF property" && c.comment != "from XISF imageType") {
        h.push(&c.key, &c.value, &c.comment);
    }
    write_file(&out, |p| write_planes(p, img.width, img.height, &[&img.data], &h))?;
    Ok(out)
}

/// Write through `<path>.tmp`, then rename into place.
pub fn write_file(path: &Path, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    write(&tmp)?;
    std::fs::rename(&tmp, path).with_context(|| format!("replace {}", path.display()))
}

fn pad<W: Write>(w: &mut W, written: &mut usize, byte: u8) -> Result<()> {
    let rem = *written % BLOCK;
    if rem != 0 {
        let n = BLOCK - rem;
        w.write_all(&vec![byte; n])?;
        *written += n;
    }
    Ok(())
}

fn fmt_card(key: &str, value: &str, comment: &str) -> String {
    let mut s = if value.starts_with('\'') {
        format!("{:<8}= {:<20}", key, value)
    } else {
        format!("{:<8}= {:>20}", key, value)
    };
    if !comment.is_empty() {
        s.push_str(" / ");
        s.push_str(comment);
    }
    s.truncate(CARD);
    format!("{s:<80}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cards() {
        let c = parse_card(format!("{:<80}", "IMAGETYP= 'LIGHT'              / Type of exposure").as_bytes()).unwrap();
        assert_eq!((c.key.as_str(), c.value.as_str(), c.comment.as_str()), ("IMAGETYP", "LIGHT", "Type of exposure"));
        let c = parse_card(format!("{:<80}", "EXPTIME =                300.0 / [s] Exposure duration").as_bytes()).unwrap();
        assert_eq!(c.value, "300.0");
        let c = parse_card(format!("{:<80}", "OBJECT  = 'it''s / here' / c").as_bytes()).unwrap();
        assert_eq!(c.value, "it's / here");
        let c = parse_card(format!("{:<80}", "HIERARCH ESO DET GAIN = 1.5 / x").as_bytes()).unwrap();
        assert_eq!((c.key.as_str(), c.value.as_str()), ("ESO DET GAIN", "1.5"));
    }

    #[test]
    fn roundtrip_u16_and_f32() {
        let dir = std::env::temp_dir().join(format!("crux-fits-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut img = Image::new(37, 11);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = (i as f32 * 997.0 % 65535.0) / 65535.0;
        }
        let mut extra = Header::default();
        extra.push("IMAGETYP", "LIGHT", "Type");
        extra.push("EXPTIME", "300.0", "");
        for bitpix in [16, -32] {
            let p = dir.join(format!("t{bitpix}.fits"));
            write_image(&p, &img, bitpix, &extra).unwrap();
            let l = read_layout(&p).unwrap();
            assert_eq!(l.header.get("IMAGETYP"), Some("LIGHT"));
            assert_eq!(l.header.f64("EXPTIME"), Some(300.0));
            assert_eq!((l.width(), l.height()), (37, 11));
            let back = read_image(&p).unwrap();
            for (a, b) in img.data.iter().zip(&back.data) {
                assert!((a - b).abs() < 1e-5, "{a} vs {b}");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rgb_planes_and_copy_from_xisf() {
        let dir = std::env::temp_dir().join(format!("crux-fits-copy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (w, h) = (29, 13);
        let plane = |k: f32| (0..w * h).map(|i| ((i as f32 * k) % 1000.0) / 1000.0).collect::<Vec<f32>>();
        let (r, g, b) = (plane(3.0), plane(7.0), plane(11.0));
        let mut extra = Header::default();
        extra.push("OBJECT", "O'Brien's nebula", "");
        extra.push("CTYPE1", "RA---TAN", "");
        extra.push("CRVAL1", "83.8", "");
        extra.push("CD1_1", "-1.864477162990e-4", "");
        extra.push("COMMENT", "made by a test", "");
        extra.push("LONGKEYWORD", "x", "");

        // Three planes: NAXIS3 = 3, planes in order.
        let p = dir.join("rgb.fits");
        write_planes(&p, w, h, &[&r, &g, &b], &extra).unwrap();
        let l = read_layout(&p).unwrap();
        assert_eq!(l.naxis, vec![w, h, 3]);
        assert_eq!(l.header.get("OBJECT"), Some("O'Brien's nebula"));
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(bytes.len() % BLOCK, 0);
        let at = |plane: usize, i: usize| {
            let o = l.data_offset as usize + (plane * w * h + i) * 4;
            f32::from_be_bytes(bytes[o..o + 4].try_into().unwrap())
        };
        assert_eq!((at(0, 5), at(1, 5), at(2, 5)), (r[5], g[5], b[5]));

        // A mono XISF output and its FITS copy: same pixels and keywords.
        let x = dir.join("master.xisf");
        crate::xisf::write_image(&x, &Image::from_data(w, h, r.clone()), &extra, None).unwrap();
        let f = ensure_copy(&x).unwrap();
        assert_eq!(f, dir.join("master.fits"));
        let l = read_layout(&f).unwrap();
        assert_eq!(l.naxis, vec![w, h]);
        assert_eq!(l.header.get("OBJECT"), Some("O'Brien's nebula"));
        assert_eq!(l.header.get("CTYPE1"), Some("RA---TAN"));
        assert_eq!(l.header.f64("CRVAL1"), Some(83.8));
        assert_eq!(l.header.get("CD1_1"), Some("-1.864477162990E-4"));
        let back = read_image(&f).unwrap();
        assert_eq!(back.data, r);
        // Up to date: left alone.
        let before = std::fs::metadata(&f).unwrap().modified().unwrap();
        ensure_copy(&x).unwrap();
        assert_eq!(std::fs::metadata(&f).unwrap().modified().unwrap(), before);
        std::fs::remove_dir_all(&dir).ok();
    }
}
