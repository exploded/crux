// Built-in networks beyond the denoiser, included by `render/net.rs`.
// Each returns (ONNX bytes, JSON metadata, version tag for caches).

fn builtin_sharpen() -> Option<(&'static [u8], &'static str, &'static str)> {
    Some((include_bytes!("sharpen-2.onnx"), include_str!("sharpen-2.json"), "sharpen-2"))
}

fn builtin_starless() -> Option<(&'static [u8], &'static str, &'static str)> {
    Some((include_bytes!("starless-1.onnx"), include_str!("starless-1.json"), "starless-1"))
}
