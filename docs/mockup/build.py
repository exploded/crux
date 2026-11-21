"""Build the mockup's data.json and images from the crux report JSON.

Usage: python -I build.py <work dir>
The work dir holds lmc.json and hdr/*.json (see README.md). Reads only those,
FITS subs and JPGs under C:\\PixInsight. Writes only under <work dir>.
"""
import json, math, os, re, subprocess, sys
from datetime import datetime, timedelta
from multiprocessing import Pool
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else sys.exit("usage: build.py <work dir>")
MOCK = S
OUT = os.path.join(S, "out")
IMG = os.path.join(OUT, "img")
ROOT = r"C:\PixInsight\2026-11-05_LMC_Mosaic"
CRUX = os.path.join(HERE, "..", "..", "target", "release", "crux.exe")
LAT, LON, TZ = -37.81, 144.96, 11  # AEDT in November

# ---------- FITS + stretch ----------
def read_fits(path):
    with open(path, "rb") as f:
        cards = {}
        done = False
        while not done:
            blk = f.read(2880)
            for i in range(0, 2880, 80):
                c = blk[i:i + 80].decode("ascii", "replace")
                k = c[:8].strip()
                if k == "END":
                    done = True
                    break
                if c[8:10] == "= ":
                    cards[k] = c[10:].split("/")[0].strip().strip("'").strip()
        w, h = int(cards["NAXIS1"]), int(cards["NAXIS2"])
        bz = float(cards.get("BZERO", 0))
        a = np.frombuffer(f.read(w * h * 2), ">i2").reshape(h, w).astype(np.float32) + bz
    return a / 65535.0

def mtf(m, x):
    return ((m - 1) * x) / ((2 * m - 1) * x - m)

def stretch(full, factor=4):
    h, w = full.shape
    hh, ww = h // factor, w // factor
    small = full[:hh * factor, :ww * factor].reshape(hh, factor, ww, factor).mean(axis=(1, 3))
    sample = full[::4, ::4]
    med = float(np.median(sample))
    madn = float(np.median(np.abs(sample - med))) * 1.4826
    # Binning averages noise down, so use the binned image's own spread for the black point.
    smed = float(np.median(small))
    smad = float(np.median(np.abs(small - smed))) * 1.4826
    c0 = max(0.0, smed - 2.8 * smad)
    m = mtf(0.25, smed - c0)
    x = np.clip((small - c0) / (1 - c0), 0, 1)
    return (np.clip(mtf(m, x), 0, 1) * 255).astype(np.uint8)

def thumb_job(args):
    idx, path, featured = args
    try:
        img = stretch(read_fits(path))
    except Exception as e:  # noqa
        return idx, None, str(e)
    pil = Image.fromarray(img)
    if featured:
        pil.save(os.path.join(IMG, f"f{idx}.jpg"), quality=84)
    t = pil.resize((216, 144), Image.LANCZOS)
    return idx, np.asarray(t), None

# ---------- astronomy (low precision, for background curves only) ----------
def jd(dt_utc):
    return dt_utc.timestamp() / 86400.0 + 2440587.5

def sun_radec(n):
    L = math.radians((280.460 + 0.9856474 * n) % 360)
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    return ra, dec

def moon_radec(n):
    Lp = math.radians((218.316 + 13.176396 * n) % 360)
    Mp = math.radians((134.963 + 13.064993 * n) % 360)
    F = math.radians((93.272 + 13.229350 * n) % 360)
    lam = Lp + math.radians(6.289) * math.sin(Mp)
    beta = math.radians(5.128) * math.sin(F)
    eps = math.radians(23.439)
    ra = math.atan2(math.sin(lam) * math.cos(eps) - math.tan(beta) * math.sin(eps), math.cos(lam))
    dec = math.asin(math.sin(beta) * math.cos(eps) + math.cos(beta) * math.sin(eps) * math.sin(lam))
    return ra, dec

def alt(ra, dec, n):
    gmst = math.radians((280.46061837 + 360.98564736629 * n) % 360)
    H = gmst + math.radians(LON) - ra
    phi = math.radians(LAT)
    return math.degrees(math.asin(math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(H)))

def night_curves(night, ra_deg, dec_deg):
    d = datetime.strptime(night, "%Y-%m-%d")
    start = d + timedelta(hours=19)  # local
    pts = []
    for k in range(0, 12 * 6 + 1):  # 19:00 -> 07:00 every 10 min
        loc = start + timedelta(minutes=10 * k)
        utc = loc - timedelta(hours=TZ)
        n =(utc - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0
        s = sun_radec(n)
        m = moon_radec(n)
        pts.append([round(10 * k / 60, 3),
                    round(alt(math.radians(ra_deg), math.radians(dec_deg), n), 1),
                    round(alt(*m, n), 1),
                    round(alt(*s, n), 1)])
    return pts

# ---------- helpers ----------
def panel_of(obj):
    m = re.search(r"(\d+)$", obj or "")
    return int(m.group(1)) if m else None

def hm(t):
    return t[11:16] if t else None

def trail_x(reason):
    xs = [float(x) for x in re.findall(r"([\d.]+)× the pixel noise", reason)]
    return max(xs) if xs else None

def main():
    os.makedirs(IMG, exist_ok=True)
    d = json.load(open(os.path.join(MOCK, "lmc.json"), encoding="utf-8"))
    frames = d["scan"]["frames"]
    metrics = d["metrics"]
    groups = {g["id"]: g for g in d["grouping"]["groups"]}

    lights = []
    for r in d["reports"]:
        i = r["frame"]
        fr = frames[i]
        me, cl = fr["meta"], fr["class"]
        mt = metrics.get(str(i), {})
        tr = r.get("transparency") or {}
        sky = r.get("sky") or {}
        flags = [{"c": f["check"], "g": f["grade"], "r": f["reason"]} for f in r["flags"]]
        tx = None
        for f in flags:
            if f["c"] == "Trail":
                tx = trail_x(f["r"])
        lights.append({
            "id": i,
            "path": os.path.relpath(me["path"], ROOT),
            "size": me["file_size"],
            "panel": panel_of(me["object"]),
            "obj": me["object"],
            "filter": me["filter"],
            "night": cl["night"],
            "time": cl["local_time"][:19],
            "exp": me["exptime"],
            "grade": r["grade"],
            "bad": cl["in_reject_dir"],
            "flags": flags,
            "notes": r.get("notes", []),
            "T": None if tr.get("value") is None else round(tr["value"], 3),
            "cells": [None if c is None else round(c, 2) for c in tr.get("cells", [])] if tr else None,
            "fwhm": None if mt.get("fwhm") is None else round(mt["fwhm"], 2),
            "fr": None if r.get("fwhm_ratio") is None else round(r["fwhm_ratio"], 3),
            "ecc": None if mt.get("ecc") is None else round(mt["ecc"], 3),
            "stars": mt.get("n_stars"),
            "sky": None if r.get("sky_e_s") is None else round(r["sky_e_s"], 3),
            "skyr": None if r.get("sky_ratio") is None else round(r["sky_ratio"], 2),
            "grad": None if r.get("gradient_pct") is None else round(r["gradient_pct"], 1),
            "trails": mt.get("trails", 0),
            "tx": tx,
            "talt": None if sky.get("target_alt") is None else round(sky["target_alt"], 1),
            "malt": None if sky.get("moon_alt") is None else round(sky["moon_alt"], 1),
            "millum": None if sky.get("moon_illum") is None else round(sky["moon_illum"], 2),
            "salt": None if sky.get("sun_alt") is None else round(sky["sun_alt"], 1),
            "ra": me.get("ra"), "dec": me.get("dec"),
            "pier": me.get("pierside"),
            "focpos": me.get("focpos"),
        })
    lights.sort(key=lambda x: x["time"])

    # all files for Organise
    files = []
    for i, fr in enumerate(frames):
        me, cl = fr["meta"], fr["class"]
        files.append({
            "id": i, "path": os.path.relpath(me["path"], ROOT), "size": me["file_size"],
            "type": cl["frame_type"], "role": cl["role"], "night": cl["night"],
            "filter": me.get("filter"), "obj": me.get("object"), "bad": cl["in_reject_dir"],
            "exp": me.get("exptime"),
        })

    # calibration
    cal = d["calibration"]
    sets = []
    for s in cal["sets"]:
        g = groups[s["group"]]
        vl = s.get("vs_lights") or {}
        e = vl.get("error") or {}
        sets.append({
            "group": s["group"], "kind": g["kind"], "label": g["label"], "filter": g.get("filter"),
            "night": g.get("night"), "grade": s["grade"], "retired": g.get("retired"),
            "master": g.get("master"), "n": len(g.get("members", [])), "measured": s.get("measured"),
            "level": s.get("flat_level"), "worst": s.get("worst_change"),
            "changes": s.get("changes"),
            "vsl": None if not e else {"spread": e.get("spread"), "edges": e.get("edges"),
                                        "masked": e.get("masked"), "n": vl.get("n_lights")},
            "flags": [{"c": f["check"], "g": f["grade"], "r": f["reason"]} for f in s["flags"]],
            "notes": s.get("notes", []),
        })
    pairs = [{"a": p["a"], "b": p["b"], "spread": round(p["spread"], 2), "ok": p["combinable"]} for p in cal["pairs"]]
    choices = [{"filter": c["filter"], "night": c["night"], "n": c["n_lights"], "chosen": c["chosen"]["label"],
                "days": c["chosen"]["days"], "reason": c["reason"], "grade": c["grade"]} for c in cal["choices"]]

    # curves per night (LMC centre)
    ras = [l["ra"] for l in lights if l["ra"]]
    decs = [l["dec"] for l in lights if l["dec"]]
    ra0, dec0 = float(np.mean(ras)), float(np.mean(decs))
    nights = sorted({l["night"] for l in lights})
    curves = {n: night_curves(n, ra0, dec0) for n in nights}

    # other targets from header-only scans
    others = []
    hdr = os.path.join(MOCK, "hdr")
    for fn in sorted(os.listdir(hdr)):
        if not fn.endswith(".json"):
            continue
        h = json.load(open(os.path.join(hdr, fn), encoding="utf-8"))
        fr = h["scan"]["frames"]
        kinds = {}
        lt = {}
        nset = set()
        objs = set()
        for f in fr:
            ty = f["class"]["frame_type"]
            role = f["class"]["role"]
            kinds[ty + ("" if role == "Raw" else "-" + role)] = kinds.get(ty + ("" if role == "Raw" else "-" + role), 0) + 1
            if ty == "Light" and role == "Raw":
                filt = f["meta"].get("filter") or "?"
                lt[filt] = lt.get(filt, 0) + (f["meta"].get("exptime") or 0)
                nset.add(f["class"]["night"])
                if f["meta"].get("object"):
                    objs.add(f["meta"]["object"])
            if ty == "Flat" and role == "Raw":
                nset.add(f["class"]["night"])
        others.append({"folder": fn[:-5], "kinds": kinds, "lightSecs": lt, "nights": sorted(n for n in nset if n),
                       "objects": sorted(objs)})

    out = {
        "root": ROOT, "lights": lights, "files": files, "sets": sets, "pairs": pairs,
        "choices": choices, "curves": curves, "centre": [ra0, dec0], "others": others,
        "skipped": d["scan"].get("stats", {}),
    }
    json.dump(out, open(os.path.join(OUT, "data.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print("lights", len(lights), "files", len(files), "nights", nights)

    # featured frames
    want = [("LMC Panel 1", "2026-11-05T21:26"), ("LMC Panel 1", "2026-11-05T21:37"),
            ("LMC Panel 26", "2026-11-12T02:13"), ("LMC Panel 31", "2026-11-12T03:59"),
            ("LMC Panel 31", "2026-11-17T03:24"), ("LMC Panel 6", "2026-11-18T04:15"),
            ("LMC Panel 8", "2026-11-08T22:53"), ("LMC Panel 26", "2026-11-12T01:3")]
    featured = {}
    for obj, t in want:
        for l in lights:
            if l["obj"] == obj and l["time"].startswith(t):
                featured[l["id"]] = l
                break
    # Panel 26 reference for blink: best-T good frame that night
    p26 = [l for l in lights if l["obj"] == "LMC Panel 26" and l["night"] == "2026-11-11" and l["grade"] == "Good" and l["T"]]
    if p26:
        best = max(p26, key=lambda l: l["T"])
        featured[best["id"]] = best
    print("featured", [(l["obj"], l["time"]) for l in featured.values()])
    out["featured"] = sorted(featured)
    json.dump(out, open(os.path.join(OUT, "data.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))

    jobs = [(k, os.path.join(ROOT, l["path"]), l["id"] in featured) for k, l in enumerate(lights)]
    cols, tw, th = 20, 216, 144
    rows = math.ceil(len(jobs) / cols)
    sheet = np.zeros((rows * th, cols * tw), np.uint8)
    with Pool(8) as pool:
        for k, arr, err in pool.imap_unordered(thumb_job, jobs):
            if err:
                print("ERR", k, err)
                continue
            r, c = divmod(k, cols)
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = arr
    Image.fromarray(sheet).save(os.path.join(IMG, "sprite.jpg"), quality=72)
    # rename featured from sprite index to frame id
    for k, l in enumerate(lights):
        p = os.path.join(IMG, f"f{k}.jpg")
        if os.path.exists(p):
            os.replace(p, os.path.join(IMG, f"frame{l['id']}.jpg"))

    # marked previews from crux itself (stars green, trails red)
    for fid, l in featured.items():
        dst = os.path.join(IMG, f"mark{fid}.png")
        subprocess.run([CRUX, "preview", os.path.join(ROOT, l["path"]), dst, "--size", "1562",
                        "--cache-dir", os.path.join(S, "cache"), "--library", os.path.join(S, "lib")],
                       check=False, capture_output=True)
        if os.path.exists(dst):
            Image.open(dst).convert("RGB").save(dst[:-4] + ".jpg", quality=84)
            os.remove(dst)

    # gallery
    gal = {
        "lmc": r"C:\PixInsight\2026-11-05_LMC_Mosaic\mosaic_claude_2026-11-18\LMC_Ha_preview_4000px.jpg",
        "m42": r"C:\PixInsight\2026-01-24 M42\Mosaic\Final_Foraxx.jpg",
        "ngc2070": r"C:\PixInsight\2026-09-12_NGC 2070 LRGB\master\ngc2070.jpg",
        "ngc7293": r"C:\PixInsight\2026-10-14_NGC 7293\master\ngc7293.jpg",
    }
    for k, p in gal.items():
        im = Image.open(p).convert("RGB")
        im.thumbnail((1400, 1400), Image.LANCZOS)
        im.save(os.path.join(IMG, f"g_{k}.jpg"), quality=82)
    print("done")

if __name__ == "__main__":
    main()
