"""Draws crux's app icon: the rail's mark (four accent strokes round a star)
on the app's dark panel, as crux.ico (16-256 px) and crux.png (256 px).

    python crates/crux-gui/assets/make_icon.py

Needs Pillow. Each size is drawn at 8x and scaled down, so small sizes stay
crisp rather than being shrunk from the 256 px image.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
SS = 8  # supersampling

BG_TOP = (0x1b, 0x23, 0x36)
BG_BOTTOM = (0x0a, 0x0d, 0x14)
EDGE = (0x2c, 0x37, 0x50)
ACCENT = (0x8f, 0xb3, 0xff)
STAR = (0xf4, 0xf7, 0xff)


def draw(size: int) -> Image.Image:
    n = size * SS
    small = size <= 24
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))

    # Rounded square with a vertical gradient (Windows 11 style margins).
    margin = 0 if small else round(n * 0.06)
    radius = round(n * (0.18 if small else 0.2))
    box = [margin, margin, n - 1 - margin, n - 1 - margin]
    grad = Image.new("RGBA", (n, n))
    gd = ImageDraw.Draw(grad)
    for y in range(n):
        t = y / (n - 1)
        c = tuple(round(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM))
        gd.line([(0, y), (n, y)], fill=c + (255,))
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius, fill=255)
    img.paste(grad, (0, 0), mask)
    if not small:
        ImageDraw.Draw(img).rounded_rectangle(box, radius, outline=EDGE + (255,), width=max(SS, round(n * 0.008)))

    # The mark: the rail's 24-unit design (strokes 2..8 and 16..22, dot r 2.2),
    # scaled into the square. Thicker at small sizes so it survives 16 px.
    inner = n - 2 * margin
    c = n / 2
    u = inner * (0.8 if small else 0.72) / 24
    w = u * (2.6 if small else 2.0)
    gap_in, gap_out = 4 * u, 10 * u

    # Glow behind the star.
    if not small:
        glow = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        r = 5.5 * u
        ImageDraw.Draw(glow).ellipse([c - r, c - r, c + r, c + r], fill=ACCENT + (120,))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(3.5 * u)))

    d = ImageDraw.Draw(img)
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        x0, y0 = c + dx * gap_in, c + dy * gap_in
        x1, y1 = c + dx * gap_out, c + dy * gap_out
        d.line([(x0, y0), (x1, y1)], fill=ACCENT + (255,), width=round(w))
        for x, y in ((x0, y0), (x1, y1)):  # round caps
            d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=ACCENT + (255,))
    r = u * (2.8 if small else 2.4)
    d.ellipse([c - r, c - r, c + r, c + r], fill=STAR + (255,))

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    images = [draw(s) for s in SIZES]
    images[-1].save(HERE / "crux.png")
    images[-1].save(HERE / "crux.ico", sizes=[(s, s) for s in SIZES], append_images=images[:-1])
    print("wrote", HERE / "crux.ico", HERE / "crux.png")


if __name__ == "__main__":
    main()
