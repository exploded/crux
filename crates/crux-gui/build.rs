//! Embeds the app icon in crux-gui.exe (Explorer, shortcuts, taskbar).
//! On the GNU toolchain this needs MinGW `windres` on PATH (MSYS2's
//! binutils; `build.bat` adds them). A no-op for other targets.

fn main() {
    println!("cargo:rerun-if-changed=assets/crux-gui.rc");
    println!("cargo:rerun-if-changed=assets/crux.ico");
    // Build scripts run on the host, so check the target, not cfg!(windows).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    embed_resource::compile("assets/crux-gui.rc", embed_resource::NONE)
        .manifest_optional()
        .expect("couldn't compile the Windows icon resource (is windres on PATH?)");
}
