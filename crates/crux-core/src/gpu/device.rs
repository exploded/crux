//! The compute device: opened once per process, on first use.

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    /// One job at a time: jobs size their buffers as if they had the card
    /// to themselves.
    busy: Mutex<()>,
}

impl Gpu {
    pub fn describe(&self) -> String {
        let backend = match self.info.backend {
            wgpu::Backend::Dx12 => "DirectX 12".to_string(),
            b => format!("{b:?}"),
        };
        format!("{} ({backend})", self.info.name)
    }

    /// Hold while running a job.
    pub fn lock(&self) -> MutexGuard<'_, ()> {
        self.busy.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Run `f`, turning wgpu's validation and out-of-memory errors (which
    /// wgpu reports out of band) into an `Err`.
    pub fn scoped<T>(&self, f: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let val = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let r = f();
        let e1 = pollster::block_on(val.pop());
        let e2 = pollster::block_on(oom.pop());
        if let Some(e) = e1.or(e2) {
            anyhow::bail!("{e}");
        }
        r
    }

    /// Wait for everything submitted so far.
    pub fn wait(&self) -> anyhow::Result<()> {
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }

    /// Wait for one submission.
    pub fn wait_for(&self, idx: wgpu::SubmissionIndex) -> anyhow::Result<()> {
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }

    /// The largest storage buffer one binding may cover, bytes.
    pub fn max_binding(&self) -> u64 {
        (self.limits.max_storage_buffer_binding_size as u64).min(self.limits.max_buffer_size)
    }
}

static DEV: OnceLock<Result<Gpu, String>> = OnceLock::new();

/// The device, or why there is none (opened on the first call).
pub(super) fn status() -> Result<&'static Gpu, String> {
    DEV.get_or_init(open).as_ref().map_err(|e| e.clone())
}

/// The device when crux may use it, else `None` (use the CPU).
pub fn get() -> Option<&'static Gpu> {
    if !super::wanted() {
        return None;
    }
    status().ok()
}

fn open() -> Result<Gpu, String> {
    let backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends, ..wgpu::InstanceDescriptor::new_without_display_handle() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .map_err(|e| format!("no graphics card found: {e}"))?;
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        return Err(format!("only a software adapter ({})", info.name));
    }
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("crux compute"),
        required_features: wgpu::Features::empty(),
        required_limits: limits.clone(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .map_err(|e| format!("couldn't open {}: {e}", info.name))?;
    device.on_uncaptured_error(Arc::new(|e: wgpu::Error| super::fail(&e.to_string())));
    device.set_device_lost_callback(|_, m| super::fail(&format!("device lost: {m}")));
    Ok(Gpu { device, queue, info, limits, busy: Mutex::new(()) })
}
