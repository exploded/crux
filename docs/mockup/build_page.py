"""Inline <work dir>/out/data.json into src.html, writing <work dir>/out/index.html."""
import os, sys

here = os.path.dirname(os.path.abspath(__file__))
work = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else sys.exit("usage: build_page.py <work dir>")
src = open(os.path.join(here, "src.html"), encoding="utf-8").read()
data = open(os.path.join(work, "out", "data.json"), encoding="utf-8").read().replace("</", r"<\/")
assert src.count("/*DATA*/null") == 1
out = os.path.join(work, "out", "index.html")
open(out, "w", encoding="utf-8").write(src.replace("/*DATA*/null", data))
print("wrote", out, os.path.getsize(out), "bytes")
