# Building crux

How to build crux from source, and how the repo is laid out. To download the
app instead, see the [Releases](https://github.com/exploded/crux/releases)
page.

crux is written in Rust. All the features below are on by default.

## Windows

crux is developed with the GNU toolchain (`stable-x86_64-pc-windows-gnu`) and
MSYS2 for the C toolchain. The MSVC toolchain also works.

1. Install Rust with [rustup](https://rustup.rs) and add the GNU toolchain:

   ```powershell
   rustup toolchain install stable-x86_64-pc-windows-gnu
   rustup default stable-x86_64-pc-windows-gnu
   ```

2. Install [MSYS2](https://www.msys2.org) in `C:\msys64`. Then, in an MSYS2
   UCRT64 shell, install binutils and gcc:

   ```sh
   pacman -S mingw-w64-ucrt-x86_64-binutils mingw-w64-ucrt-x86_64-gcc
   ```

3. Build with `build.bat`, which puts MSYS2 on the `PATH` for cargo:

   ```powershell
   cmd /c .\build.bat            # release build of crux and crux-gui
   cmd /c .\build.bat test       # run the tests
   cmd /c .\build.bat platforms  # type-check for Linux and macOS
   ```

The binaries are in `target\release`.

## Linux and macOS

Install Rust with [rustup](https://rustup.rs). On Linux, also install a C
compiler and the windowing libraries (Debian and Ubuntu package names):

```sh
sudo apt-get install build-essential pkg-config libwayland-dev libxkbcommon-dev libx11-dev
```

On macOS, install the Xcode command-line tools (`xcode-select --install`).
Then build:

```sh
cargo build --workspace --release
```

The binaries are in `target/release`.

## Features

- `ml`: crux's neural networks (denoising, star correction and star
  removal), compiled into the binaries from `crates/crux-core/models/`. They
  run with [tract](https://github.com/sonos/tract) on the CPU, or on the
  graphics card with `gpu`.
- `gpu`: the networks and the registration warps on the graphics card,
  through [wgpu](https://wgpu.rs): DirectX 12 or Vulkan on Windows, Vulkan
  on Linux, Metal on macOS, for NVIDIA, AMD, Intel and Apple GPUs. No vendor
  libraries are needed. When there's no suitable card, or it fails, crux uses
  the CPU and gets the same result. On an RTX 3070 the networks run about
  15× faster than on an 8-core CPU.
- `sqlite`: the library index and flat library, in SQLite (bundled, so it
  needs a C compiler). Without it, the index falls back to JSON and there's
  no flat library.
- `update`: the app's check for a newer release on GitHub (ureq with rustls,
  which needs a C compiler for `ring`).

## Command line

The build also makes `crux`, a command-line tool that does the same work
from a terminal or a script. It's used for testing, validation and training
crux's networks, and isn't included in releases. Run `crux --help` for its
commands. For example:

```sh
crux report <folder> --html report.html   # grade a folder
crux process <folder>                     # calibrate, register, integrate
crux image <folder>                       # render the finished image
crux collect <folder>                     # make training bundles
```

## Releases

GitHub Actions runs the tests on Windows, Linux and macOS on every push.
Pushing a version tag such as `v0.1.0` builds crux-gui for all three and
attaches it to a draft release, which is published by hand. The Build
workflow can also be run from the Actions tab; its builds are then attached
to that run.

## Layout

| Folder | Contents |
|---|---|
| `crates/crux-core` | The library: file readers, measurement, grading, calibration, processing and rendering. |
| `crates/crux-cli` | The `crux` command-line tool, for testing and training. |
| `crates/crux-gui` | The `crux-gui` desktop app (egui on wgpu). |
| `tools/ml` | Training scripts for crux's networks (PyTorch). See its [README](tools/ml/README.md). |
