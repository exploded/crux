# crux

An astrophotography processor, from raw subs to a finished image.

crux grades subs, finds the right calibration frames anywhere in your
library, stacks, builds mosaics, and finishes the image, without moving or
changing your raw data.

**[⬇ Download crux for Windows, macOS or Linux](https://github.com/exploded/crux/releases/latest)** · [See it in action](https://deepspaceplace.com/crux)

<p align="center">
  <a href="https://deepspaceplace.com/static/crux/v2/m17.jpg"><img src="https://deepspaceplace.com/static/crux/v2/m17.jpg" width="800" alt="M 17, the Omega Nebula, in a gold and blue narrowband palette."></a><br>
  <em>M 17 from 4.7 hours of H-alpha, OIII and SII subs, processed entirely by crux.</em>
</p>

**Status:** in development. Used daily on Windows, and tested on Linux
(Debian) and macOS (an iMac).

## What it does

### Sorts good subs from bad

crux measures every sub for focus, tracking, cloud, sky brightness and
satellite trails, compares it with the rest of its night and its panel, and
gives each frame a grade with a reason, such as "transparency 0.03: stars at
3% of their brightness in the field's best frame". Drag a reject limit and the
whole night regrades as you drag, or override individual frames by hand.

<p align="center">
  <a href="https://deepspaceplace.com/static/crux/v2/02-frames.webp"><img src="https://deepspaceplace.com/static/crux/v2/02-frames.webp" width="800" alt="crux's Frames page: a night's transparency chart with rejected frames in red, reject limit sliders and the reasons for the selected frame."></a><br>
  <em>One night of a mosaic. Cloud rolls in after 01:00; each red frame says why it was rejected.</em>
</p>

### Never touches your data

crux reads your files where they are. Its index and cache live in its own
folder, and everything it makes (calibrated and registered frames, masters,
mosaics and images) goes to a working folder that you can delete whenever you
like.

If you want your folders tidied, crux can do that too: combine a target shot
over several folders, or file subs by night, panel or filter. You see every
move before it happens, and any change can be undone. Rejects are moved to a
`BAD` folder, never deleted.

### Finds the best flats, darks and bias in your whole library

Point crux at your top-level folders and it builds a calibration library:
every flat set, dark and bias, checked and graded. When you process a target,
crux picks the best calibration from anywhere in that library, so there's no
need to copy flats, darks and bias into every target folder.

Flats are matched by night, filter and the dust pattern they share with your
lights, and checked against the lights. Darks are matched by their hot
pixels.

<p align="center">
  <a href="https://deepspaceplace.com/static/crux/v2/04-calibrate.webp"><img src="https://deepspaceplace.com/static/crux/v2/04-calibrate.webp" width="800" alt="crux's Calibration page: each night's lights linked to a flat set from the library and to a master dark and bias, with flat error maps below."></a><br>
  <em>Each night takes its flats from the library. A set shot through cloud is left out.</em>
</p>

### Recipes make the next night easy

Everything you set in the image steps (gradient removal, palette, stretch,
finishing, star colours, sharpening and noise reduction, crop and turn) is
kept as the target's recipe. Add another night's subs and crux processes only
the new frames, stacks them in, and rebuilds the image with the same recipe.
Save a recipe by name to start other targets from it.

### Mosaics are automatic

Shoot the panels and crux does the rest. It plate-solves each panel's master,
fits the lens distortion, places the panels from where they are on the sky
(not from their names), matches their brightness, and blends the seams. This
36-panel mosaic came together in 12 seconds: 176 megapixels, with stars in the
overlaps lining up to 0.11 pixels.

<p align="center">
  <a href="https://deepspaceplace.com/static/crux/v2/lmc.jpg"><img src="https://deepspaceplace.com/static/crux/v2/lmc.jpg" width="800" alt="The Large Magellanic Cloud in H-alpha, a 36-panel mosaic."></a><br>
  <em>The Large Magellanic Cloud, 36 panels, 19.6 hours of H-alpha.</em>
</p>

### Sharpens, denoises and removes stars

crux has three neural networks of its own, trained on real amateur data from
four cameras, including subs from members of the Astronomical Society of
Victoria: a noise reducer, a star corrector that rounds out elongated or
comatic stars and can tighten them, and a star remover. New recipes correct
star shapes and reduce noise by default.

With natural star colours, crux stretches the starless image and the stars
separately, so you can make stars smaller without touching the nebula.

If you have an RC Astro licence, crux can also run BlurXTerminator and
NoiseXTerminator on your linear masters.

<p align="center">
  <a href="https://deepspaceplace.com/static/crux/v2/enhance-split.webp"><img src="https://deepspaceplace.com/static/crux/v2/enhance-split.webp" width="700" alt="A before and after split of a star field from M 17: noisy, slightly elongated stars on the left; round stars and a smooth background on the right."></a><br>
  <em>M 17's top-left corner at full size: as stacked, and with crux's star correction and 80% noise reduction.</em>
</p>

### Works with mono and one-shot colour

crux works with mono cameras and filters (narrowband SHO and HOO with nine
palette presets, LRGB, LRGB with H-alpha) and with one-shot-colour cameras,
including dual-band filters. It reads FITS, XISF and Canon CR2 files from
N.I.N.A., the ASIAIR, PixInsight's WBPP and MaxIm DL. One-shot-colour support
is the newest part of crux.

### Fast, on your graphics card

crux is native code and uses all your CPU cores: calibrating and registering a
26-megapixel sub takes about a second. The networks, and the warps that line
up frames and mosaic panels, run on your graphics card (NVIDIA, AMD, Intel or
Apple) with no vendor libraries to install. Without a suitable card, crux uses
the CPU and gets the same result.

### Gallery

<table>
  <tr>
    <td><a href="https://deepspaceplace.com/static/crux/v2/ngc2244.jpg"><img src="https://deepspaceplace.com/static/crux/v2/ngc2244.jpg" width="400" alt="NGC 2244, the Rosette Nebula, in an orange and blue narrowband palette."></a><br><em>NGC 2244, the Rosette Nebula: 4.1 hours of H-alpha, OIII and SII.</em></td>
    <td><a href="https://deepspaceplace.com/static/crux/v2/ic434.jpg"><img src="https://deepspaceplace.com/static/crux/v2/ic434.jpg" width="400" alt="The Horsehead and Flame nebulae, a three-panel narrowband mosaic."></a><br><em>IC 434, the Horsehead and Flame nebulae: three panels, 17.8 hours.</em></td>
  </tr>
  <tr>
    <td><a href="https://deepspaceplace.com/static/crux/v2/ngc7293.jpg"><img src="https://deepspaceplace.com/static/crux/v2/ngc7293.jpg" width="400" alt="NGC 7293, the Helix Nebula, in a red and blue narrowband palette."></a><br><em>NGC 7293, the Helix Nebula: 28.7 hours of H-alpha, OIII and SII.</em></td>
    <td><a href="https://deepspaceplace.com/static/crux/v2/ngc2070.jpg"><img src="https://deepspaceplace.com/static/crux/v2/ngc2070.jpg" width="400" alt="NGC 2070, the Tarantula Nebula, in LRGB."></a><br><em>NGC 2070, the Tarantula Nebula: 3.9 hours of LRGB.</em></td>
  </tr>
</table>

More screenshots and examples are on the
[crux page at deepspaceplace.com](https://deepspaceplace.com/crux).

## Download and first run

Builds for Windows (x86-64), Linux (x86-64) and macOS (Apple Silicon and
Intel, in one file) are on the
[Releases](https://github.com/exploded/crux/releases/latest) page. There's
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
  and newer), and a Vulkan driver to use the graphics card
  (`mesa-vulkan-drivers` on Debian and Ubuntu, or your GPU vendor's driver).

When a newer release is out, the app shows an **Update** button in its top
bar that opens the release's download page. It asks GitHub at most once a day
and sends nothing but crux's version; turn it off in Settings, under Updates.

## Using the app

Open a night's folder, a target's folder, or a whole library folder that holds
many targets. crux reads the headers in a second or two, then measures the
frames in the background; later opens come from its cache.

The rail on the left has these pages:

- **Home:** the last night, the target's nights and its picture, or, for a
  library folder, a card for every target with what's been stacked and made.
  Also starts the folder watcher, which grades new subs as they arrive, and
  the autopilot, which processes a whole library of dated folders.
- **Inbox:** things worth a decision, such as frames close to a reject limit,
  frames where crux disagrees with how you filed them, or flats that need
  checking.
- **Frames:** a contact sheet and night timeline of every sub, a viewer that
  blinks each frame against its field's reference, and sliders for the reject
  limits. Keep or reject any frame by hand with the keyboard.
- **Calibrate:** which flats, darks and bias each night uses and why, heat
  maps of flat errors, and the flat library across all your folders.
- **Image:** the processing steps, each with a before and after preview:
  calibrate, cosmetic, register, integrate and mosaic, then sharpen and
  denoise, gradient, palette and stretch, stars, finish, and crop and save.
- **Organise:** browse the whole library, set a folder layout, preview and
  apply the moves, check disk space and duplicates, and undo any earlier move.
- **Settings:** processing, the calibration library, the cache, your site and
  camera, and **Start fresh**.

The top bar shows whether crux is computing on the graphics card (GPU) or the
processor (CPU).

## Optional tools

- **Plate solving** needs one of [ASTAP](https://www.hnsky.org/astap.htm)'s
  star databases, such as D50. crux looks in ASTAP's install folder
  (`C:\Program Files\astap` on Windows). If yours is elsewhere, choose its
  folder in Settings, under Add-ons. crux reads the catalogue directly and
  doesn't run ASTAP itself.
- **RC Astro tools:** if you own licences for RC Astro's command-line tools,
  the Image steps can run them on your images. crux finds `rc-astro` in its
  install folder or on the PATH, or you can choose it in Settings, under
  Add-ons. crux isn't affiliated with RC Astro and doesn't include its
  software. RC Astro's software and its outputs aren't used to develop or
  evaluate crux's networks.

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
environment variable `CRUX_GPU=off`). To stop the app checking for updates,
set `check = false` under `[updates]` (or `CRUX_UPDATE_CHECK=off`).

crux reads your data folders in place and changes them only through moves you
approve. Its caches and library index are in `%LOCALAPPDATA%\crux` (Linux:
`~/.local/share/crux`; macOS: `~/Library/Application Support/crux`).
Processing outputs go to a workspace folder, `C:\crux-work` by default on
Windows and `crux-work` in that same folder elsewhere; change it under
`[process]` or in Settings.

## Starting again

To have crux work everything out again from a fresh scan, use **Start fresh**
in Settings (or on the library's Home page). It empties every folder's
measurements, removes the caches of folders that no longer exist, moves the
library index aside as a backup, and clears processing outputs in the
workspace: only the intermediate files (registered frames and caches), or
everything crux made there, finished images included. Your data stays, and so
do your decisions unless you ask.

## Build from source

See [BUILDING.md](BUILDING.md) for how to build crux, its optional features,
the `crux` command-line tool and the repo layout.

## Acknowledgements

Members of the Astronomical Society of Victoria contributed raw data for
training and testing crux's networks on cameras other than the author's own.

## Licence

MIT. See [LICENSE](LICENSE).
