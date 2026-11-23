# crux

crux is an astrophotography image toolkit. It takes a night's raw subs from
capture software such as N.I.N.A. or an ASIAIR, and goes all the way to a
finished image. It grades every frame, matches calibration frames, then
calibrates, registers, integrates, builds mosaics and renders colour images.
**Status:** in development, used daily on Windows. The tests run on
Windows, Linux and macOS.

## Download

Builds for Windows (x86-64), Linux (x86-64) and macOS (Apple Silicon) are on
the [Releases](https://github.com/exploded/crux/releases) page. There's
nothing to install: unpack the archive and run `crux-gui`.

The builds aren't signed, so the first run needs one extra step:

- **Windows:** if SmartScreen says "Windows protected your PC", click
  **More info**, then **Run anyway**.
- **macOS:** remove the download quarantine, in Terminal, in the unpacked
  folder:

  ```sh
  xattr -d com.apple.quarantine crux-gui
  ```

- **Linux:** needs glibc 2.35 or later (Ubuntu 22.04, Debian 12, Fedora 36
  and newer), and a Vulkan driver to use the graphics card (`mesa-vulkan-drivers`
  on Debian and Ubuntu, or your GPU vendor's driver).

## What it does

- **Triage:** reads FITS, XISF and Canon CR2 files, and classifies each one
  as a light, dark, flat or bias frame. Grades lights for focus, tracking,
  sky brightness, gradients, transparency (by star photometry), patchy cloud
  and satellite or aircraft trails. Each grade comes with a plain-language
  reason.
- **Calibration checks:** finds bad bias, dark and flat frames, and checks
  flats against the lights and against each other. Builds a library of flat
  sets across all your folders and matches each night to the right set by
  its dust pattern. Matches darks and bias by their hot-pixel fingerprint.
- **Filing:** moves rejects and reorganises folders through a plan you
  preview first. Every move is journalled and can be undone. crux never
  deletes your data.
- **Processing:** calibration, hot-pixel correction, plate solving (with
  ASTAP's star catalogue), registration, integration with normalisation,
  weighting and pixel rejection, and mosaics with lens-distortion correction
  and seam matching. Works with mono and one-shot-colour cameras, including
  dual-band filters.
- **Image:** sharpening and noise reduction with crux's own neural networks,
  star removal, gradient removal, narrowband and broadband palettes,
  stretching, natural star colours with stars stretched separately,
  cropping, turning and saved recipes.
- **Automation:** a folder watcher that grades new subs as they arrive, and
  an autopilot that processes a whole library of dated folders.

## The desktop app

Open a night's folder, a target's folder, or a whole library folder that
holds many targets. crux reads the headers in a second or two, then measures
the frames in the background; later opens come from its cache.

The rail on the left has these pages:

- **Home:** the last night, the target's nights and its picture, or, for a
  library folder, a card for every target with what's been stacked and made.
  Also starts the folder watcher and the autopilot.
- **Inbox:** things worth a decision, such as frames close to a reject
  limit, frames where crux disagrees with how you filed them, flats that
  need checking, or master flats built from bad flats.
- **Frames:** a contact sheet and night timeline of every sub, a viewer that
  blinks each frame against its field's reference, and sliders for the
  reject limits. Keep or reject any frame by hand with the keyboard.
- **Calibrate:** which flats, darks and bias each night uses and why, heat
  maps of flat errors, and the flat library across all your folders.
- **Image:** the processing steps, each with a before and after preview:
  calibrate, cosmetic, register, integrate and mosaic, then sharpen and
  denoise, gradient, palette and stretch, stars, finish, and crop and save.
- **Organise:** browse the whole library, set a folder layout, preview and
  apply the moves, check disk space and duplicates, and undo any earlier
  move.
- **Settings:** processing, the calibration library, the cache, your site
  and camera, and **Start fresh**.

The top bar shows whether crux is computing on the graphics card (GPU) or
the processor (CPU).

## Build from source

crux is written in Rust. All three features below are on by default.

### Windows

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

### Linux and macOS

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

### Features

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

### Command line

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

### Releases

GitHub Actions runs the tests on Windows, Linux and macOS on every push.
Pushing a version tag such as `v0.1.0` builds crux-gui for all three and
attaches it to a draft release, which is published by hand. The Build
workflow can also be run from the Actions tab; its builds are then attached
to that run.

## Optional tools

- **Plate solving** needs one of [ASTAP](https://www.hnsky.org/astap.htm)'s
  star databases, such as D50. crux looks in ASTAP's install folder
  (`C:\Program Files\astap` on Windows). If yours is elsewhere, choose its
  folder in Settings, under Add-ons. crux reads the catalogue directly and
  doesn't run ASTAP itself.
- **RC Astro tools:** if you own licences for RC Astro's command-line tools,
  the Image stage can run them on your images. crux finds `rc-astro` in its
  install folder or on the PATH, or you can choose it in Settings, under
  Add-ons. crux isn't
  affiliated with RC Astro and doesn't include its software. RC Astro's
  software and its outputs aren't used to develop or evaluate crux's
  networks.

## Settings

Most settings are on the app's Settings page. To change one by hand, put it in
`crux.toml` in the data folder (for that folder only), or in your own
`crux.toml` (for all folders):

| System | Your `crux.toml` |
|---|---|
| Windows | `%APPDATA%\crux\crux.toml` |
| Linux | `~/.config/crux/crux.toml` |
| macOS | `~/Library/Application Support/crux/crux.toml` |

For example, set your observing site:

```toml
[site]
latitude = -37.81
longitude = 144.96
timezone = "Australia/Melbourne"
```

To always compute on the CPU, set `use = "off"` under `[gpu]` (or the
environment variable `CRUX_GPU=off`).

crux reads your data folders in place and changes them only through moves
you approve. Its caches and library index are in `%LOCALAPPDATA%\crux`
(Linux: `~/.local/share/crux`; macOS: `~/Library/Application Support/crux`).
Processing outputs go to a workspace folder, `C:\crux-work` by default on
Windows and `crux-work` in that same folder elsewhere; change it under
`[process]` or in Settings.

## Starting again

To have crux work everything out again from a fresh scan, use **Start
fresh** in Settings (or on the library's Home page). It
empties every folder's measurements, removes the caches of folders that no
longer exist, moves the library index aside as a backup, and clears
processing outputs in the workspace: only the intermediate files (registered
frames and caches), or everything crux made there, finished images included.
Your data stays, and so do your decisions unless you ask.

## Layout

| Folder | Contents |
|---|---|
| `crates/crux-core` | The library: file readers, measurement, grading, calibration, processing and rendering. |
| `crates/crux-cli` | The `crux` command-line tool, for testing and training. |
| `crates/crux-gui` | The `crux-gui` desktop app (egui on wgpu). |
| `tools/ml` | Training scripts for crux's networks (PyTorch). See its [README](tools/ml/README.md). |

## Acknowledgements

Members of the Astronomical Society of Victoria contributed raw data for
training and testing crux's networks on cameras other than the author's own.

## Licence

MIT. See [LICENSE](LICENSE).
