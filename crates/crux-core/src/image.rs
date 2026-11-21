//! In-memory image type shared by every module.
//!
//! Pixels are mono `f32`, row-major, row 0 = first row stored in the file
//! (PixInsight's default "upper-left origin" for FITS, and XISF's native
//! order). Integer data is normalised to [0, 1] like PixInsight;
//! `adu_scale` converts back to ADU (65535 for 16-bit cameras).

#[derive(Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
    /// Multiply a normalised value by this to get ADU.
    pub adu_scale: f32,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Image({}x{}, adu_scale={})", self.width, self.height, self.adu_scale)
    }
}

impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Image { width, height, data: vec![0.0; width * height], adu_scale: 65535.0 }
    }

    pub fn from_data(width: usize, height: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), width * height, "image data length mismatch");
        Image { width, height, data, adu_scale: 65535.0 }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        self.data[y * self.width + x] = v;
    }

    pub fn row(&self, y: usize) -> &[f32] {
        &self.data[y * self.width..(y + 1) * self.width]
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
