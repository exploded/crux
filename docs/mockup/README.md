# crux UI mockup

Source for the clickable mockup described in [../ui-design.md](../ui-design.md):
https://claude.ai/artifact/PvV1W8cHpb4FpSRNhQVNub

| File | What |
|---|---|
| `src.html` | The page: HTML, CSS and vanilla JS, with `/*DATA*/null` where the data goes. |
| `build.py` | Extracts compact data from the crux report JSON; renders stretched thumbnails (one sprite sheet), featured frames, `crux preview` marks and gallery images. |
| `build_page.py` | Inlines the data into `src.html` as `out/index.html`. |

Generated files stay in a work folder outside the repo. Nothing is written
under `C:\PixInsight`: the cache and flat library go to the work folder.

## Rebuild

Needs `target\release\crux.exe`, Python with numpy and Pillow.

PowerShell on your PC, in `C:\Projects\rust\crux`:

```powershell
$w = "$env:TEMP\crux-mockup"
New-Item -ItemType Directory -Force "$w\hdr" | Out-Null
.\target\release\crux.exe report C:\PixInsight\2026-11-05_LMC_Mosaic --cache-dir "$w\cache" --library "$w\lib" --json "$w\lmc.json"
Get-ChildItem C:\PixInsight -Directory | Where-Object Name -ne '2026-11-05_LMC_Mosaic' | ForEach-Object {
  .\target\release\crux.exe report $_.FullName --headers-only --cache-dir "$w\cache-hdr\$($_.Name)" --library "$w\lib" --json "$w\hdr\$($_.Name).json"
}
python -I docs\mockup\build.py $w
python -I docs\mockup\build_page.py $w
```

The page is then `$w\out\index.html` with images in `$w\out\img\`. Open it
over HTTP, not `file://`: Chrome blocks reading canvas pixels from `file://`
images, which blanks the Image view.

## Notes

- At default limits the page's regrading (`flagsOf` in `src.html`)
  reproduces crux's grades on all 302 LMC lights. Trail brightness is parsed
  from the reason text, rounded to 0.1×, so an exact tie defers to crux's
  grade.
- The Moon, Sun and LMC altitude curves use low-precision formulas (about 1°),
  fine for a chart background. Per-frame values come from crux.
- Space-map sizes in the Library health tab are hard-coded from `du` on
  2026-11-18.
