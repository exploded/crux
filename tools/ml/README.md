# crux's trained networks

Training for crux's own networks: the denoiser (below), the star corrector
(`deconv.py`) and the star remover (`starless.py`), both near the end.

The denoiser's approach is Noise2Noise
(Lehtinen et al. 2018): two stacks of the same field, made from different
frames, are two independent noisy views of one scene. A network trained to
turn one into the other learns to remove noise without ever seeing a clean
image.

**Never use RC Astro software (BlurXTerminator, NoiseXTerminator,
StarXTerminator) or anything it has processed here: not for training, not
for scoring, not for comparison.** Its licence (clause 10) forbids using the
software or its outputs to train, test, benchmark, validate or improve a
model that does what it does. Everything here is built from crux's own
stacks of raw subs.

## 1. Make training bundles

PowerShell, in the repository root:

```powershell
crux collect "<workspace>\2026-10-05_M 17" --yes
```

`<dir>` is a data folder crux has processed, its workspace folder, or a
folder of registered frames from WBPP (`*_r.xisf`) or Siril (`r_*.fit`).
Bundles go to `<workspace>\train\<folder>` (500 MB at most; `--max-mb`).
Each bundle's `bundle.html` shows what's in it. Bundles hold image patches,
star cutouts and camera details only: no file names, paths, site, observer
or day.

## 2. Set up Python (once)

PowerShell, in the repository root:

```powershell
python -m venv .local\ml\venv
.local\ml\venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.local\ml\venv\Scripts\python -m pip install numpy onnx onnxscript
```

## 3. Train

PowerShell, in `tools\ml`:

```powershell
..\..\.local\ml\venv\Scripts\python train.py --bundles <workspace>\train --test-camera STL QHY9 --out ..\..\.local\ml\runs\r1
```

Sets whose camera name contains a `--test-camera` string are held out: their
score shows how the network does on an optical train it has never seen.
The run writes `model.pt`, `log.csv`, `denoise.onnx` and `denoise.json`
(the input transform and the scores).

## 4. Score and check

```powershell
..\..\.local\ml\venv\Scripts\python evaluate.py --model ..\..\.local\ml\runs\r1\model.pt --bundles "<workspace>\train\2010-02-25 QHY9 NGC 3324"
..\..\.local\ml\venv\Scripts\python check_rust.py --run ..\..\.local\ml\runs\r1 --bundle "<workspace>\train\2010-02-25 QHY9 NGC 3324" --crux <crux.exe built with --features ml>
```

**Gain** is how many times lower the error is than the stack's own noise
error: ×2 is worth twice the frames. With A and B independent,
E|A−B|² = V_A + V_B, so a denoised D's true error is E|D−B|² − V_B, measured
without a clean image. "faint" scores only pixels near their patch's median,
where noise shows most. `evaluate.py` compares the network with a
multiscale wavelet denoise and a Gaussian blur.

## 5. Run it in crux

The networks are built into crux (the `ml` feature, on by default). To try a new model before building it in:

```powershell
crux dev denoise <master.xisf> --model ..\..\.local\ml\runs\r1\denoise.onnx --out denoised.xisf --png compare.png
```

The network sees `asinh((v − sky) / (3 · noise))` and a channel holding
FWHM ÷ 4, in 512 px tiles blended over 32 px.

## Star corrector (`deconv.py`)

Truths are the centres (70%) of crux's own masters, exported with `crux dev
export-fits` and denoised with the denoiser. Each input is a truth patch
blurred by a random aberration kernel (elongation, a coma tail, both, or
none), plus real stack noise from the bundles (half-stack differences). A
"sharpen" control channel picks between correcting shapes only and also
undoing extra symmetric blur. Truths are shrunk up to 2.6× so small stars
are covered, and a flux term keeps every star's light.

```powershell
..\..\.local\ml\venv\Scripts\python deconv.py --masters <workspace>\train\masters --bundles <workspace>\train --denoiser ..\..\.local\ml\runs\r2-holdout-stl-qhy9\model.pt --out ..\..\.local\ml\runs\d2 --max-scale 2.6 --p-scale 0.7 --flux-weight 0.2
bash eval_sharpen.sh ..\..\.local\ml\runs\d2 <crux.exe> <scratch>
```

`eval_sharpen.sh` prints crux's 3 × 3 FWHM/eccentricity and the flux kept
on M 17 H and the held-out NGC 6188 (STL-11000M) and NGC 3324 (QHY9).

## Star remover (`starless.py`)

Backgrounds are crux's classic starless (`crux dev starless`) of the same
masters, denoised. Real star cutouts from the bundles' PSF samples are
painted on at random places and brightnesses (with `--bright`, halos and
diffraction spikes on bright stars), plus real noise; the target keeps the
same noise, so the network removes stars only.

```powershell
..\..\.local\ml\venv\Scripts\python starless.py --backgrounds <workspace>\train\starless --bundles <workspace>\train --denoiser ..\..\.local\ml\runs\r2-holdout-stl-qhy9\model.pt --out ..\..\.local\ml\runs\s2 --max-peak 20000 --bright
python starless_check.py original.fits classic.fits network.fits --png cmp.png
```

## Putting a model into crux

Copy the `.onnx` and a trimmed `.json` into `crates/crux-core/models/` with
a new version number (e.g. `sharpen-3`), and name it in `models/extra.rs`
(the denoiser is in `render/net.rs`). The version tag is part of every
cache key, so results from the old model are remade. `export.py` turns any
checkpoint into ONNX + JSON.

