# crux

crux is an astrophotography image toolkit. It takes a night's raw subs from
capture software such as N.I.N.A. or an ASIAIR, and goes all the way to a
finished image. It grades every frame, matches calibration frames, then
calibrates, registers, integrates, builds mosaics and renders colour images.
It has a command-line tool (`crux`) and a desktop app (`crux-gui`).

**Status:** in development, with no releases yet. It is built and used on
Windows. The workspace also type-checks for Linux and macOS, but it hasn't
been built or run there.

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
  gradient removal, narrowband and broadband palettes, stretching, natural
  star colours, cropping and saved recipes.
- **Automation:** a folder watcher that grades new subs as they arrive, and
  an autopilot that processes a whole library of dated folders.

Run `crux --help` for the full command list.

## Build

crux is written in Rust. It's developed with the GNU toolchain on Windows
(`stable-x86_64-pc-windows-gnu`) and MSYS2 for the C toolchain.

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

GitHub Actions runs the tests on every push. It builds crux and crux-gui for
Windows, Linux and macOS (Apple Silicon) when a version tag such as `v0.1.0`
is pushed, and attaches them to a draft release. You can also run the Build
workflow by hand from the Actions tab and download the builds from that run.
The Linux and macOS builds haven't been tried on a real machine yet.

### Features

All three features are on by default:

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

## Optional tools

- **Plate solving** needs one of [ASTAP](https://www.hnsky.org/astap.htm)'s
  star databases, such as D50. crux looks in `C:\Program Files\astap`, or
  in the folder named by `CRUX_CATALOG`. It reads the catalogue directly and
  doesn't run ASTAP itself.
- **RC Astro tools:** if you own licences for RC Astro's command-line tools,
  `crux rc-astro` and the Image stage can run them on your images. crux isn't
  affiliated with RC Astro and doesn't include its software. RC Astro's
  software and its outputs aren't used to develop or evaluate crux's
  networks.

## Layout

| Folder | Contents |
|---|---|
| `crates/crux-core` | The library: file readers, measurement, grading, calibration, processing and rendering. |
| `crates/crux-cli` | The `crux` command-line tool. |
| `crates/crux-gui` | The `crux-gui` desktop app (egui on wgpu). |
| `tools/ml` | Training scripts for crux's networks (PyTorch). See its [README](tools/ml/README.md). |
| `docs` | UI design notes and the HTML mockup. |

## Settings

`crux config` prints every setting and its default value. To change a
setting, put it in `crux.toml` in the data folder (for that folder only), or
in `%APPDATA%\crux\crux.toml` (for all folders). For example, set your
observing site:

```toml
[site]
latitude = -37.81
longitude = 144.96
timezone = "Australia/Melbourne"
```

To always compute on the CPU, set `use = "off"` under `[gpu]` (or the
environment variable `CRUX_GPU=off`).

## Starting again

To have crux work everything out again from a fresh scan, use **Start
fresh** in Settings (or on the library's Home page), or run `crux fresh`. It
empties every folder's measurements, removes the caches of folders that no
longer exist, moves the library index aside as a backup, and clears
processing outputs in the workspace: only the intermediate files (registered
frames and caches), or everything crux made there, finished images included.
Your data stays, and so do your decisions unless you ask. `crux fresh
--dry-run` shows what it would clear.

## Acknowledgements

Members of the Astronomical Society of Victoria contributed raw data for
training and testing crux's networks on cameras other than the author's own.

## Licence

MIT. See [LICENSE](LICENSE).
