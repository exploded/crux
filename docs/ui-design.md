# crux UI design

The plan for the desktop app (milestones 8 onward). Written 2026-11-18,
before any GUI code, so it can shape the remaining core milestones.
Decisions: **egui + wgpu**, **fully native processing** (crux calibrates,
registers, stacks and stretches itself), and a clickable HTML mockup to
react to before the Rust build.

## Mockup

Clickable mockup: https://claude.ai/artifact/PvV1W8cHpb4FpSRNhQVNub
(private to James). Source and build scripts are in [mockup/](mockup/).

It runs on the real LMC results (`crux report --json`, 2026-11-18). At
default limits its in-page regrading matches crux's grades on all 302 lights,
so the live threshold sliders are trustworthy. Use it as the reference for
layout, colours, copy and behaviour in the egui build. Not real: the flat
heat maps are interpolated from the 8-point edge profile, the Autopilot
estimate is an example, star/trail marks exist for 8 featured frames only,
and the Image view's "before" is an approximation.

## The idea in one line

**You judge, crux does the work.** crux runs every stage with good defaults;
your time goes on looking at results and approving or overriding them, not
on moving files, blinking subs or wiring up processes.

## What's wrong with today's tools

- **PixInsight:** the steepest learning curve, a scripts-versus-processes
  split, redundant tools, Blink for manual rejection, hand-sorting files into
  folders.
- **Wizard tools (step 1, 2, 3):** easy to start, unsatisfying to use. Rigid,
  opaque, no way to see or fix what a step decided.
- **SetiAstro Suite Pro:** no obvious place to start.
- **What worked:** the Narrowband Palette Mixer (`C:\Projects\pixinsight`):
  presets rendered from your own data, sliders, a live preview, the same code
  for preview and output.

## Who it's for

1. **The time-poor enthusiast (primary: James).** Mono narrowband and LRGB
   through N.I.N.A., multi-night mosaics, thousands of subs. Knows a good sub
   when he sees one, hates the clerical work, fussy about the final image.
2. **The newcomer.** A folder of subs from a first clear night and no idea
   what WBPP is. Needs one obvious first action and a decent result.
3. **The expert tweaker.** Wants every threshold and parameter, and proof the
   algorithms are sound: numbers, distributions, maps.

One UI serves all three through progressive disclosure: every stage panel
shows **Auto → Adjust → Expert** in place, not as separate modes.

## Principles

1. **crux proposes, you approve.** Like a pull request for your data.
2. **Show the whole journey, never trap the user in it.** A pipeline strip
   (Frames → Calibration → Stack → Image) gives a wizard's orientation, but
   every stage is clickable at any time, runs in the background and shows its
   own status.
3. **Reactive, like a build system.** Change a setting and only the affected
   downstream work reruns (cache keyed by inputs, as `cache.rs` does now). No
   Apply buttons for previews.
4. **Every verdict explains itself.** A plain sentence first ("Thin cloud:
   53% of starlight lost"), then the evidence on click: the measurement, the
   cohort distribution with this frame marked, the threshold, and the setting
   that controls it.
5. **Look only where needed.** Sort by suspicion; summarise confident goods.
   The Inbox holds only what needs human eyes.
6. **Nothing is destructive.** Grades and overrides live in a sidecar. File
   moves are a previewed, journaled, undoable plan. Real data is never written
   in place; processed output goes to a workspace folder.
7. **The image is the hero.** Dark, quiet chrome; big pixels; instant
   pan/zoom; live GPU stretch.

## Mental model

| Thing | What it is |
|---|---|
| Library | Every folder crux watches (`C:\PixInsight`, `\\nas\astronomy`). |
| Target | An object gathered across folders and nights. Mosaic panels are its children. |
| Pipeline | Per target: Frames → Calibration → Stack → Image. Each stage shows not started, running, needs review, done or stale. |
| Inbox | Cross-target queue of decisions crux wasn't sure about. |
| Recipe | Saved settings for a pipeline: one readable, versioned thing (unlike PixInsight's icons). Output images record the recipe that made them. |

Left rail: **Home · Inbox · Targets · Organise · Calibration library ·
Activity · Settings**.

## Screens

### Home

- **First run:** a drop zone and Add folder. The header scan takes ~0.2 s
  for 2,251 files, so crux answers almost at once: "LMC Mosaic: 302 lights,
  H/O/S, 7 nights, 35 panels. 189 flats, 5 masters. No raw darks, using
  masters." Pixel analysis then streams in.
- **Returning:** the Last night card, Inbox count, target gallery (latest
  stack or best sub, integration per filter as coloured bars, nights,
  status), and a calendar heat map of imaging nights.

### Frames (replaces Blink)

- **Contact sheet** of auto-stretched thumbnails, grouped night → panel →
  filter, worst first, with a grade badge on each.
- **Night timeline** (the signature view): FWHM, eccentricity, transparency
  and sky plotted through the night over target altitude, Moon altitude and
  twilight shading (all from `astro.rs`). Click a point to open the frame;
  drag a time range to select it ("everything after 03:40") and reject in
  one go.
- **Live thresholds:** drag a reject slider and frames change colour in the
  grid and chart as you drag, with the cost: "rejecting 17 frames, 1 h 25 m
  of 25 h 10 m". The palette mixer's feel, applied to triage.
- **Viewer:** full-resolution pan and zoom. Overlays by key: star ellipses
  (elongation and direction), trails, 4×4 transparency grid, background
  model, hot pixels. A 3×3 aberration inspector. **Blink against the
  reference** (the cohort's best frame, registered) rather than through every
  frame, and a **difference view** (frame − reference) where cloud and trails
  stand out. Zoom stays synced while stepping through frames.
- **Keyboard-first:** J/K next and previous, A accept, R reject, U undo,
  Space blink, 1–5 overlays. Overrides are pinned and survive re-runs.
- **Mosaic map:** panels laid out on the sky from RA/Dec, coloured by
  integration or quality. Click a tile to filter to that panel.

### Organise (the headline feature)

Moving files is the biggest waste of time today, so organisation is a
first-class screen, not an export step. Three layers:

**Virtual first.** crux knows every file's target, panel, night, filter,
role and grade wherever it sits. Organise browses the whole library by any of
those (target → panel → night → filter, night → target, grade → reason) as
**smart views**, with nothing moved. A query bar saves views: "LMC panel 12,
H, November, FWHM < 4 px". crux's own processing works from these views, so
moving files is never required, only wanted for tidiness or for other tools.

**Rules you design once.** A template builder with draggable token chips:
`{target}\{night}\{panel}\{filter}`, rejects to `{target}\BAD\{reason}`,
calibration to the flat and dark libraries. A live example path from your
real files updates as you edit. Presets: by night, by panel, by filter,
WBPP-friendly, Siril-friendly. Rules apply to a watched **intake folder**
(N.I.N.A. output): new subs are filed each morning after you approve, or
automatically for confident goods if you turn that on.

**See the plan, then do it.**
- **Before/after tree diff**, live as the rules change: counts and sizes per
  folder, moved files highlighted, collisions listed first.
- **Drag to correct:** drag frames onto a panel, target or reject bucket
  ("these 12 are really panel 13"). Stored as a correction in the decisions
  sidecar; FITS headers are never modified.
- **Apply** runs in the background with a progress tree. Cross-volume moves
  (NAS ↔ PC) copy, verify by hash, then remove the source, all journaled.
- **History:** every apply is a point on a timeline; undo any of them in a
  click. Never deletes.

**Library health:**
- **Space map:** treemap by target and kind (raw subs, masters, WBPP
  intermediates such as `calibrated` and `registered`, processed). Reclaim
  moves regenerable intermediates to a `Reclaimable` folder that you empty
  yourself.
- **Duplicates:** the same sub on the NAS and the PC (size + hash), masters
  copied into several folders.
- **Gaps and orphans:** lights with no usable flats, unused flat sets, a
  target missing a filter, panels well short of their neighbours'
  integration, files whose names and headers disagree.
- **Self-describing folders:** each organised target folder gets a small
  `crux.json` manifest (grades, decisions, recipe).
- **Zero-copy exports:** a selection exported for WBPP or Siril as NTFS hard
  links (same volume) or a file list.

### Calibration

- A matching graph per target: light groups ← flat sets ← dark flats or bias,
  plus darks and masters, with assumptions flagged ("master has no GAIN,
  matched on exposure and size").
- Flat checks as heat maps (flat vs lights error, set vs set ratio, frame
  drift) beside the FlatChecker numbers. FLAT.bad shows red: "12.6%
  left–right error, would leave a gradient in 237 lights".
- Flat library browser: sets by filter and night, dust fingerprints, which
  lights use which set; drag to swap the matched set.

### Stack and Image (native processing)

- The palette mixer's layout, evolved: stage list on the left, large preview
  on the right, preset tiles rendered from your data underneath.
- Stages: Calibrate → Cosmetic → Register → Normalise → Integrate (weights,
  rejection) → Mosaic → Gradient → Colour/palette → Stretch → Finish. Each
  stage card has Auto (one line on what it did), Adjust (3–5 key sliders)
  and Expert (every parameter), and shows where each value comes from:
  default, `crux.toml`, recipe or overridden here.
- Before/after split slider at every stage; a history strip to step back
  through intermediate results.
- Live preview from a screen-sized proxy (GPU shaders for stretch, palette,
  curves, saturation); the full-resolution render follows in the background.
- Integration report: per-frame weights, rejection maps, SNR per hour, and
  what-ifs ("adding the 9 warn frames: +4% SNR, +0.2 px FWHM").

### Activity

Every background job (scan, measure, stack, apply) with live progress, ETA
and cancel. Results stream in: thumbnails appear as frames finish.

## Two headline workflows

**The morning after (~2 minutes).** crux watches the N.I.N.A. folder and
analyses subs as they land. In the morning: "84 subs: 71 good, 9 warn, 4
reject. Cloud from 03:40. Flats OK (1.1%)." Glance at the 13 flagged frames
on the timeline, press Apply: subs are filed, rejects go to `BAD\`, the
target's stack updates.

**Autopilot ("process all of `\\nas\astronomy` again").** Add the folder,
press Process all. crux estimates first ("6,400 subs, ~330 GB to read over
SMB, ~4 h; output to `D:\crux-work`; nothing on the NAS changes"). Overnight
it groups targets, grades, matches calibration, stacks and renders each
target with the default recipe. Morning: a gallery of finished images and an
Inbox of only the low-confidence decisions. Accepting an Inbox item reruns
only what it affects.

## Feedback language

- Good has no badge; **warn** is amber; **reject** is red; **your override**
  is outlined with a pin; stale (inputs changed) is hatched.
- Reasons in one sentence, numbers on hover, evidence on click.
- Progress shows partial results, never a frozen window. Long jobs show an ETA.
- Toasts for finished jobs. The only modal is confirming a move.

## Visual design

- Dark, blue-black surfaces, one accent. (The red night mode was removed on
  2026-11-20.)
- Fixed filter colours everywhere: H red, O teal, S amber, L white, R/G/B.
- Inter (or similar) with tabular numerals for metrics; generous spacing;
  restrained motion (results fade in, smooth zoom).
- Sky details: altitude curves, Moon phase per night, mosaic sky map.

## Architecture

- `crux-gui`: eframe on wgpu, a custom `egui::Style`, `egui_plot` or custom
  painting for charts.
- **Image view:** pixels uploaded once as an R32F/RGBA16F texture with mips;
  a WGSL shader applies stretch, palette, overlays, blink and difference, so
  slider drags run at 60 fps on a 26 MP frame.
- **Job runner:** core work on rayon threads; events over a channel. The
  `(Stage, done, total)` progress callback in `pipeline::run` grows per-frame
  result events and a cancellation token.
- **Thumbnails and proxies** cached beside the binned maps (`maps/<hash>`).
- **Decisions sidecar** (overrides, pins, corrections, threshold changes)
  shared by CLI and GUI, so `crux plan` respects GUI choices.
- **Watcher:** file-system notifications, polling for SMB paths.

### Core changes the GUI needs

- Per-frame result events and cancellation from `pipeline::run` and
  `analyse_lights`.
- Regrading from cached metrics without pixel access, so threshold sliders
  respond in milliseconds.
- The decisions sidecar, read by milestone 7's plan.
- Structured reasons (value, threshold, cohort stats, config key) alongside
  the existing strings.
- Stretched thumbnails (reuse `stretch.rs` and `crux preview`).
- For Organise: a template rule engine, content hashes (duplicates, verified
  moves), per-folder manifests.

## Library index and plate solving

Organise's "virtual first" layer needs one place that knows every file, and
plate solving makes target identity independent of names and folders.

**Library index.** One SQLite database for the whole library
(`%APPDATA%\crux\library.db`) replaces the per-folder JSON sidecars as the
source of truth: headers, measurements, grades, decisions and sky position
per file. Files are keyed by path + size + mtime as now, plus a content hash
so a moved or renamed file is relinked instead of re-measured. Smart views,
calibration matching and stacking query it, so files never need moving for
crux's own work.

**Plate solving** (lights and processed images):
- **Target from the sky:** "M42", "M 42 Panel 3" and an unnamed old sub
  group by position; mislabelled subs show up.
- **Mosaic layout** from the solutions, not the `OBJECT` name.
- **Camera rotation per frame:** flats are valid only for the same rotation
  and dust. Rotation plus milestone 6's dust fingerprint decides which flat
  sets suit which lights.
- **Registration seed:** the solution is a near-exact starting alignment
  for milestone 11 and mosaic assembly.
- **Cloud signal:** a sub that fails to solve is usually clouded or trailed.

Solves are hinted: N.I.N.A. writes RA/Dec, and FOCALLEN + XPIXSZ give the
scale (0.67″/px on the AT12IN), so each takes a fraction of a second. Stars
are already detected and cached; `transparency.rs` already matches star
patterns. The new pieces are a catalogue (Gaia subset, or ASTAP's D50,
~900 MB) and quad matching against it; calling the ASTAP CLI is the
fallback. Solutions live in the index, never written into raw files (WCS
keywords only on exported copies).

**Darks, bias and dark flats can't be solved** (no stars). Match them by
fingerprint instead: headers (camera, exposure, gain, offset, binning,
temperature, date) plus measurements (level, noise, hot-pixel pattern; hot
pixels identify a sensor and drift slowly). crux can then rank "the best
dark flats for these flats" across the whole library, with reasons.

**Cost:** star detection needs pixels, so the first pass over
`\\nas\astronomy` reads every sub once (an overnight job); afterwards only
new or changed files.

## Milestones (proposed)

6. Flat library (in progress).
7. Move plan, apply and undo, plus the decisions sidecar.
8. **GUI shell and review:** Home, targets, Frames (contact sheet, timeline,
   viewer with GPU stretch and overlays, live thresholds, keyboard triage),
   Activity. Replaces Blink.
9. **GUI Organise:** the library index (SQLite), smart views and query bar,
   rule builder, intake folder, tree diff, drag-to-correct, apply with
   history and undo, library health, hard-link exports.
10. **GUI calibration:** matching graph, flat heat maps, library browser,
    dark and bias fingerprint matching across the library.
11. **Plate solving, native calibrate and register** (registration builds on
    the star matching and affine refit in `transparency.rs`, seeded by the
    solutions).
12. **Native integrate:** normalisation, weights, rejection, integration report.
13. **Image stage:** gradient (from `background.rs`), palette (port the
    mixer), stretch, finishing, presets, before/after, recipes.
14. **Mosaic** assembly; watcher and autopilot for whole libraries.

Each GUI milestone stops for James to try it on the LMC data.
