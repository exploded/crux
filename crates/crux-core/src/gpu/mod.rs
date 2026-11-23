//! Compute on the graphics card, through wgpu: DX12 or Vulkan on Windows,
//! Vulkan on Linux, Metal on macOS, for NVIDIA, AMD, Intel and Apple GPUs
//! alike. No vendor libraries.
//!
//! Everything here is optional: callers ask [`get`] for the device and run
//! their CPU code when it says no (no `gpu` feature, no suitable adapter,
//! switched off, or an earlier failure). A failure part-way through a job
//! turns the GPU off for the rest of the session ([`fail`]) and the caller
//! redoes the job on the CPU. Software adapters (WARP, llvmpipe) are
//! refused: they're slower than crux's own CPU code.
//!
//! Switch it off with `[gpu] use = "off"` in `crux.toml` ([`set_enabled`])
//! or `CRUX_GPU=off` in the environment.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[cfg(feature = "gpu")]
mod device;
#[cfg(feature = "gpu")]
pub use device::{get, Gpu};
#[cfg(all(feature = "gpu", feature = "ml"))]
pub mod nn;
#[cfg(feature = "gpu")]
pub mod warp;

static ENABLED: AtomicBool = AtomicBool::new(true);
static FAILED: Mutex<Option<String>> = Mutex::new(None);

/// Turn the GPU on or off for this process (the `[gpu] use` setting).
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// May crux use the GPU? The setting, `CRUX_GPU`, and no failure so far.
pub fn wanted() -> bool {
    if !cfg!(feature = "gpu") || !ENABLED.load(Ordering::Relaxed) {
        return false;
    }
    if let Some(v) = std::env::var_os("CRUX_GPU") {
        let v = v.to_string_lossy().to_ascii_lowercase();
        if matches!(v.as_str(), "off" | "0" | "no" | "false" | "cpu") {
            return false;
        }
    }
    FAILED.lock().map(|f| f.is_none()).unwrap_or(false)
}

/// Something went wrong on the GPU: say so once, and use the CPU for the
/// rest of the session.
pub fn fail(why: &str) {
    let mut f = FAILED.lock().unwrap_or_else(|e| e.into_inner());
    if f.is_none() {
        eprintln!("crux: the graphics card failed ({why}); using the CPU from now on");
        *f = Some(why.to_string());
    }
}

/// Where crux computes: the graphics card's name ("NVIDIA GeForce RTX 3070
/// (Vulkan)"), or why it uses the CPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Compute {
    Gpu(String),
    Cpu(String),
}

/// Where crux computes, opening the device if it isn't open yet.
pub fn compute() -> Compute {
    compute_with(true).expect("waited for the device")
}

/// Where crux computes, without waiting: `None` while the device hasn't
/// been opened yet ([`open`] opens it).
pub fn compute_now() -> Option<Compute> {
    compute_with(false)
}

fn compute_with(wait: bool) -> Option<Compute> {
    if !cfg!(feature = "gpu") {
        return Some(Compute::Cpu("this build has no GPU support".into()));
    }
    if let Some(why) = FAILED.lock().ok().and_then(|f| f.clone()) {
        return Some(Compute::Cpu(format!("the graphics card failed: {why}")));
    }
    if !wanted() {
        return Some(Compute::Cpu("the graphics card is switched off".into()));
    }
    #[cfg(feature = "gpu")]
    {
        let st = if wait { Some(device::status()) } else { device::status_now() };
        st.map(|s| match s {
            Ok(g) => Compute::Gpu(g.describe()),
            Err(why) => Compute::Cpu(why),
        })
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = wait;
        unreachable!()
    }
}

/// Open the device now when crux may use it (blocks; the GUI calls it on a
/// thread at start so [`compute_now`] has an answer).
pub fn open() {
    #[cfg(feature = "gpu")]
    if wanted() {
        let _ = device::status();
    }
}

/// What crux will use, for Settings and `crux config`: "NVIDIA GeForce RTX
/// 3070 (Vulkan)", or "CPU (why)".
pub fn describe() -> String {
    match compute() {
        Compute::Gpu(name) => name,
        Compute::Cpu(why) => format!("CPU ({why})"),
    }
}

/// Without the `gpu` feature there is never a device.
#[cfg(not(feature = "gpu"))]
pub fn get() -> Option<&'static ()> {
    None
}
