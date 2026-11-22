//! Pure-Rust XISF 1.0 reader (monolithic files, attached data blocks).
//!
//! Supports UInt8/16/32 and Float32/64 samples, uncompressed, zlib, lz4 and
//! lz4hc, with optional byte shuffling (`+sh`). FITSKeyword elements become
//! header cards; common XISF properties are mapped onto their FITS keyword
//! names when the keyword itself is absent (WBPP masters carry few keywords).

use crate::header::{clean_value, Header};
use crate::image::Image;
use anyhow::{anyhow, bail, Context, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const SIGNATURE: &[u8; 8] = b"XISF0100";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    UInt8,
    UInt16,
    UInt32,
    Float32,
    Float64,
}

impl SampleFormat {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "UInt8" => Self::UInt8,
            "UInt16" => Self::UInt16,
            "UInt32" => Self::UInt32,
            "Float32" => Self::Float32,
            "Float64" => Self::Float64,
            o => bail!("unsupported XISF sampleFormat {o}"),
        })
    }
    pub fn bytes(self) -> usize {
        match self {
            Self::UInt8 => 1,
            Self::UInt16 => 2,
            Self::UInt32 | Self::Float32 => 4,
            Self::Float64 => 8,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::UInt8 => "UInt8",
            Self::UInt16 => "UInt16",
            Self::UInt32 => "UInt32",
            Self::Float32 => "Float32",
            Self::Float64 => "Float64",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compression {
    pub codec: String,
    pub shuffled: bool,
    pub uncompressed_size: usize,
    pub item_size: usize,
}

#[derive(Debug, Clone)]
pub struct XisfLayout {
    pub header: Header,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub sample_format: SampleFormat,
    pub data_offset: u64,
    pub data_size: u64,
    pub compression: Option<Compression>,
    pub big_endian: bool,
    /// XISF `imageType` attribute, e.g. "MasterDark".
    pub image_type: Option<String>,
    pub bounds: Option<(f64, f64)>,
}

pub fn read_layout(path: &Path) -> Result<XisfLayout> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    read_layout_from(&mut f).with_context(|| format!("XISF header of {}", path.display()))
}

pub fn read_layout_from<R: Read>(r: &mut R) -> Result<XisfLayout> {
    let mut pre = [0u8; 16];
    r.read_exact(&mut pre).context("truncated XISF file")?;
    if &pre[..8] != SIGNATURE {
        bail!("not an XISF 1.0 file");
    }
    let len = u32::from_le_bytes(pre[8..12].try_into().unwrap()) as usize;
    if len > 256 << 20 {
        bail!("XISF header length {len} is implausible");
    }
    let mut xml = vec![0u8; len];
    r.read_exact(&mut xml).context("truncated XISF header")?;
    while xml.last() == Some(&0) {
        xml.pop();
    }
    let text = String::from_utf8_lossy(&xml);
    parse_xml(&text)
}

fn parse_xml(text: &str) -> Result<XisfLayout> {
    let doc = roxmltree::Document::parse_with_options(
        text,
        roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() },
    )
    .context("XISF XML header")?;
    let image = doc
        .descendants()
        .find(|n| n.has_tag_name("Image") && n.parent().is_some_and(|p| p.has_tag_name("xisf")))
        .or_else(|| doc.descendants().find(|n| n.has_tag_name("Image")))
        .ok_or_else(|| anyhow!("XISF file has no Image element"))?;

    let geometry = image.attribute("geometry").context("Image has no geometry")?;
    let dims: Vec<usize> = geometry.split(':').map(|s| s.parse().map_err(|_| anyhow!("bad geometry {geometry}"))).collect::<Result<_>>()?;
    if dims.len() < 3 {
        bail!("only 2-D XISF images are supported (geometry {geometry})");
    }
    let sample_format = SampleFormat::parse(image.attribute("sampleFormat").unwrap_or("Float32"))?;
    let location = image.attribute("location").context("Image has no location")?;
    let parts: Vec<&str> = location.split(':').collect();
    if parts.first() != Some(&"attachment") || parts.len() != 3 {
        bail!("unsupported XISF data location {location}");
    }
    let data_offset: u64 = parts[1].parse()?;
    let data_size: u64 = parts[2].parse()?;
    let compression = image.attribute("compression").map(parse_compression).transpose()?;
    let bounds = image.attribute("bounds").and_then(|b| {
        let (lo, hi) = b.split_once(':')?;
        Some((lo.parse().ok()?, hi.parse().ok()?))
    });

    let mut header = Header::default();
    for kw in image.children().filter(|n| n.has_tag_name("FITSKeyword")) {
        header.push(
            kw.attribute("name").unwrap_or(""),
            clean_value(kw.attribute("value").unwrap_or("")),
            kw.attribute("comment").unwrap_or(""),
        );
    }
    // Map XISF properties onto FITS names when the keyword is missing.
    const MAP: &[(&str, &str)] = &[
        ("Instrument:ExposureTime", "EXPTIME"),
        ("Instrument:Filter:Name", "FILTER"),
        ("Instrument:Camera:XBinning", "XBINNING"),
        ("Instrument:Camera:YBinning", "YBINNING"),
        ("Instrument:Camera:Gain", "EGAIN"),
        ("Instrument:Camera:Name", "INSTRUME"),
        ("Instrument:Sensor:Temperature", "CCD-TEMP"),
        ("Instrument:Sensor:TargetTemperature", "SET-TEMP"),
        ("Instrument:Sensor:XPixelSize", "XPIXSZ"),
        ("Instrument:Telescope:Name", "TELESCOP"),
        ("Instrument:Telescope:FocalLength", "FOCALLEN"),
        ("Observation:Object:Name", "OBJECT"),
        ("Observation:Object:RA", "RA"),
        ("Observation:Object:Dec", "DEC"),
        ("Observation:Time:Start", "DATE-OBS"),
        ("Observation:Time:End", "DATE-END"),
        ("Observation:Location:Latitude", "SITELAT"),
        ("Observation:Location:Longitude", "SITELONG"),
    ];
    let props: Vec<(String, String)> = image
        .children()
        .chain(doc.root_element().children())
        .filter(|n| n.has_tag_name("Property"))
        .filter_map(|p| {
            let id = p.attribute("id")?.to_string();
            let v = p.attribute("value").map(str::to_string).or_else(|| p.text().map(|t| t.trim().to_string()))?;
            Some((id, v))
        })
        .collect();
    for (id, key) in MAP {
        if header.contains(key) {
            continue;
        }
        if let Some((_, v)) = props.iter().find(|(pid, _)| pid == id) {
            let v = match *key {
                // XISF stores focal length in metres and pixel size in µm.
                "FOCALLEN" => crate::header::parse_number(v).map(|m| format!("{}", m * 1000.0)).unwrap_or_else(|| v.clone()),
                // Times look like 2026-11-05T10:21:13.837Z.
                "DATE-OBS" | "DATE-END" => v.trim_end_matches('Z').to_string(),
                _ => v.clone(),
            };
            header.push(*key, v, "from XISF property");
        }
    }
    let image_type = image.attribute("imageType").map(str::to_string);
    if !header.contains("IMAGETYP") {
        if let Some(t) = &image_type {
            header.push("IMAGETYP", t.clone(), "from XISF imageType");
        }
    }
    Ok(XisfLayout {
        header,
        width: dims[0],
        height: dims[1],
        channels: dims[dims.len() - 1],
        sample_format,
        data_offset,
        data_size,
        compression,
        big_endian: image.attribute("byteOrder") == Some("big"),
        image_type,
        bounds,
    })
}

fn parse_compression(s: &str) -> Result<Compression> {
    let parts: Vec<&str> = s.split(':').collect();
    let (codec, shuffled) = match parts[0].strip_suffix("+sh") {
        Some(c) => (c.to_string(), true),
        None => (parts[0].to_string(), false),
    };
    let uncompressed_size = parts.get(1).context("compression without size")?.parse()?;
    let item_size = parts.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1);
    Ok(Compression { codec, shuffled, uncompressed_size, item_size })
}

pub fn read_image(path: &Path) -> Result<Image> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let l = read_layout_from(&mut f).with_context(|| format!("XISF header of {}", path.display()))?;
    read_image_with(&mut f, &l).with_context(|| format!("XISF data of {}", path.display()))
}

pub fn read_image_with<R: Read + Seek>(r: &mut R, l: &XisfLayout) -> Result<Image> {
    r.seek(SeekFrom::Start(l.data_offset))?;
    let mut raw = vec![0u8; l.data_size as usize];
    r.read_exact(&mut raw).context("truncated XISF data block")?;
    if let Some(c) = &l.compression {
        raw = decompress(&raw, c)?;
    }
    let bpp = l.sample_format.bytes();
    let plane = l.width * l.height;
    if raw.len() < plane * bpp {
        bail!("XISF data block too small: {} < {}", raw.len(), plane * bpp);
    }
    let channels = l.channels.max(1).min(raw.len() / (plane * bpp));
    let mut out = vec![0f32; plane];
    // Planar storage; average channels for colour images.
    for ch in 0..channels {
        let bytes = &raw[ch * plane * bpp..(ch + 1) * plane * bpp];
        decode_into(&mut out, bytes, l.sample_format, l.big_endian, l.bounds);
    }
    if channels > 1 {
        let k = 1.0 / channels as f32;
        out.iter_mut().for_each(|v| *v *= k);
    }
    for v in out.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    let mut img = Image::from_data(l.width, l.height, out);
    img.adu_scale = if l.sample_format == SampleFormat::UInt8 { 255.0 } else { 65535.0 };
    Ok(img)
}

/// Rows `y0..y1` of an uncompressed mono image, decoded like `read_image`
/// (into `out`, which must hold `(y1 - y0) * width` values). Fails on
/// compressed or multi-channel data: read those whole.
pub fn read_rows(path: &Path, l: &XisfLayout, y0: usize, y1: usize, out: &mut [f32]) -> Result<()> {
    if l.compression.is_some() || l.channels > 1 {
        bail!("{}: compressed or colour data can't be read by rows", path.display());
    }
    let bpp = l.sample_format.bytes();
    let n = (y1 - y0) * l.width;
    anyhow::ensure!(out.len() >= n && y1 <= l.height, "row range out of bounds");
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    f.seek(SeekFrom::Start(l.data_offset + (y0 * l.width * bpp) as u64))?;
    let mut raw = vec![0u8; n * bpp];
    f.read_exact(&mut raw).with_context(|| format!("truncated XISF data in {}", path.display()))?;
    let out = &mut out[..n];
    out.iter_mut().for_each(|v| *v = 0.0);
    decode_into(out, &raw, l.sample_format, l.big_endian, l.bounds);
    for v in out.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    Ok(())
}

fn decode_into(out: &mut [f32], bytes: &[u8], fmt: SampleFormat, be: bool, bounds: Option<(f64, f64)>) {
    macro_rules! dec {
        ($t:ty, $n:expr, $conv:expr) => {
            for (d, c) in out.iter_mut().zip(bytes.chunks_exact($n)) {
                let a: [u8; $n] = c.try_into().unwrap();
                let v = if be { <$t>::from_be_bytes(a) } else { <$t>::from_le_bytes(a) };
                *d += $conv(v);
            }
        };
    }
    match fmt {
        SampleFormat::UInt8 => out.iter_mut().zip(bytes).for_each(|(d, &b)| *d += b as f32 / 255.0),
        SampleFormat::UInt16 => dec!(u16, 2, |v: u16| v as f32 / 65535.0),
        SampleFormat::UInt32 => dec!(u32, 4, |v: u32| (v as f64 / 4294967295.0) as f32),
        SampleFormat::Float32 | SampleFormat::Float64 => {
            // Floats are rescaled from their declared bounds to [0, 1].
            let (lo, hi) = bounds.unwrap_or((0.0, 1.0));
            let (lo, k) = if hi > lo { (lo, 1.0 / (hi - lo)) } else { (0.0, 1.0) };
            if fmt == SampleFormat::Float32 {
                dec!(f32, 4, |v: f32| ((v as f64 - lo) * k) as f32);
            } else {
                dec!(f64, 8, |v: f64| ((v - lo) * k) as f32);
            }
        }
    }
}

fn decompress(data: &[u8], c: &Compression) -> Result<Vec<u8>> {
    let mut out = match c.codec.as_str() {
        "zlib" => {
            let mut v = Vec::with_capacity(c.uncompressed_size);
            flate2::read::ZlibDecoder::new(data).read_to_end(&mut v).context("zlib")?;
            v
        }
        "lz4" | "lz4hc" => lz4_flex::block::decompress(data, c.uncompressed_size).map_err(|e| anyhow!("lz4: {e}"))?,
        other => bail!("XISF compression codec '{other}' is not supported (re-save uncompressed, zlib or lz4)"),
    };
    if out.len() != c.uncompressed_size {
        bail!("decompressed {} bytes, expected {}", out.len(), c.uncompressed_size);
    }
    if c.shuffled && c.item_size > 1 {
        out = unshuffle(&out, c.item_size);
    }
    Ok(out)
}

fn unshuffle(data: &[u8], item: usize) -> Vec<u8> {
    let n = data.len() / item;
    let mut out = vec![0u8; data.len()];
    for b in 0..item {
        let src = &data[b * n..(b + 1) * n];
        for (i, &v) in src.iter().enumerate() {
            out[i * item + b] = v;
        }
    }
    let tail = n * item;
    out[tail..].copy_from_slice(&data[tail..]);
    out
}

fn shuffle(data: &[u8], item: usize) -> Vec<u8> {
    let n = data.len() / item;
    let mut out = vec![0u8; data.len()];
    for i in 0..n {
        for b in 0..item {
            out[b * n + i] = data[i * item + b];
        }
    }
    let tail = n * item;
    out[tail..].copy_from_slice(&data[tail..]);
    out
}

/// Write a mono Float32 XISF file. `codec` is None, "zlib" or "lz4" (always shuffled).
/// Used by tests and later modules for masters.
pub fn write_image(path: &Path, img: &Image, extra: &Header, codec: Option<&str>) -> Result<()> {
    write_with(path, img.width, img.height, &[&img.data], extra, &[], codec)
}

/// Write a Float32 image of one (Gray) or three (RGB, planar) channels,
/// uncompressed, with String properties (id, value).
pub fn write_planes(path: &Path, width: usize, height: usize, planes: &[&[f32]], extra: &Header, props: &[(&str, &str)]) -> Result<()> {
    write_with(path, width, height, planes, extra, props, None)
}

fn write_with(path: &Path, width: usize, height: usize, planes: &[&[f32]], extra: &Header, props: &[(&str, &str)], codec: Option<&str>) -> Result<()> {
    let nch = planes.len();
    anyhow::ensure!(nch == 1 || nch == 3, "XISF writer: 1 or 3 channels, not {nch}");
    for p in planes {
        anyhow::ensure!(p.len() == width * height, "XISF writer: plane size");
    }
    // Uncompressed planes go to the file straight from memory.
    let raw_len = width * height * nch * 4;
    let (block, comp_attr) = match codec {
        None => (None, String::new()),
        Some(c) => {
            let raw: Vec<u8> = planes.iter().flat_map(|p| le_bytes(p).into_owned()).collect();
            let sh = shuffle(&raw, 4);
            let comp = match c {
                "zlib" => {
                    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
                    e.write_all(&sh)?;
                    e.finish()?
                }
                "lz4" => lz4_flex::block::compress(&sh),
                o => bail!("unsupported codec {o}"),
            };
            (Some(comp), format!(" compression=\"{c}+sh:{}:4\"", raw.len()))
        }
    };
    let block_len = block.as_ref().map(|b| b.len()).unwrap_or(raw_len);
    let esc = |s: &str| s.replace('&', "&amp;").replace('"', "&quot;").replace('<', "&lt;").replace('>', "&gt;");
    let mut kws = String::new();
    for c in &extra.cards {
        let looks_numeric = crate::header::parse_number(&c.value).is_some();
        let v = if looks_numeric { c.value.clone() } else { format!("'{}'", c.value.replace('\'', "''")) };
        kws.push_str(&format!(
            "<FITSKeyword name=\"{}\" value=\"{}\" comment=\"{}\"/>",
            esc(&c.key),
            esc(&v),
            esc(&c.comment)
        ));
    }
    for (id, v) in props {
        kws.push_str(&format!("<Property id=\"{}\" type=\"String\">{}</Property>", esc(id), esc(v)));
    }
    let space = if nch == 3 { "RGB" } else { "Gray" };
    // The data offset depends on the header length, so build twice.
    let build = |offset: usize| {
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><xisf version=\"1.0\" xmlns=\"http://www.pixinsight.com/xisf\"><Image geometry=\"{}:{}:{}\" sampleFormat=\"Float32\" bounds=\"0:1\" colorSpace=\"{}\" location=\"attachment:{}:{}\"{}>{}</Image></xisf>",
            width, height, nch, space, offset, block_len, comp_attr, kws
        )
    };
    let probe = build(0).len() + 64;
    let offset = (16 + probe).div_ceil(4096) * 4096;
    let mut xml = build(offset).into_bytes();
    xml.resize(offset - 16, 0);
    let mut f = std::io::BufWriter::new(File::create(path)?);
    f.write_all(SIGNATURE)?;
    f.write_all(&(xml.len() as u32).to_le_bytes())?;
    f.write_all(&[0u8; 4])?;
    f.write_all(&xml)?;
    match &block {
        Some(b) => f.write_all(b)?,
        None => {
            for p in planes {
                f.write_all(&le_bytes(p))?;
            }
        }
    }
    f.flush()?;
    Ok(())
}

/// Little-endian bytes of `v`: borrowed as they are on little-endian
/// machines (every platform crux builds for), copied elsewhere.
fn le_bytes(v: &[f32]) -> std::borrow::Cow<'_, [u8]> {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: f32 has no padding and any byte pattern is a valid u8;
        // the length covers exactly the slice's memory.
        std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) })
    }
    #[cfg(not(target_endian = "little"))]
    {
        std::borrow::Cow::Owned(v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_codecs() {
        let dir = std::env::temp_dir().join(format!("crux-xisf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut img = Image::new(53, 17);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = ((i * 7919) % 1000) as f32 / 1000.0;
        }
        let mut extra = Header::default();
        extra.push("IMAGETYP", "Master Flat", "");
        extra.push("FILTER", "H", "");
        for codec in [None, Some("zlib"), Some("lz4")] {
            let p = dir.join(format!("t-{}.xisf", codec.unwrap_or("raw")));
            write_image(&p, &img, &extra, codec).unwrap();
            let l = read_layout(&p).unwrap();
            assert_eq!(l.header.get("IMAGETYP"), Some("Master Flat"));
            assert_eq!(l.header.get("FILTER"), Some("H"));
            assert_eq!((l.width, l.height), (53, 17));
            let back = read_image(&p).unwrap();
            assert_eq!(back.data, img.data, "codec {codec:?}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shuffle_inverse() {
        let d: Vec<u8> = (0..103).map(|i| i as u8).collect();
        assert_eq!(unshuffle(&shuffle(&d, 4), 4), d);
    }

    #[test]
    fn properties_map_to_keywords() {
        let xml = r#"<xisf version="1.0" xmlns="http://www.pixinsight.com/xisf"><Image geometry="4:2:1" sampleFormat="UInt16" location="attachment:4096:16" imageType="MasterDark"><Property id="Instrument:ExposureTime" type="Float32" value="300"/><Property id="Instrument:Filter:Name" type="String">Ha</Property></Image></xisf>"#;
        let l = parse_xml(xml).unwrap();
        assert_eq!(l.header.f64("EXPTIME"), Some(300.0));
        assert_eq!(l.header.get("FILTER"), Some("Ha"));
        assert_eq!(l.header.get("IMAGETYP"), Some("MasterDark"));
    }
}
