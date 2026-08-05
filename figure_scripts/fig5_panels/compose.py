#!/usr/bin/env python3
"""Assemble the five panels into one figure at Nature two-column width.

Three-column grid, 183 mm wide.  Top row carries the employment evidence, the
bottom row the money.  No panel labels are drawn: the author assigns them.

The panels are embedded as SVG, so the composite stays vector and nothing is
re-rendered: what is checked here is exactly what the individual files contain.
"""
import re
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "figures" / "05"
MM = 72 / 25.4        # matplotlib writes SVG in POINTS, not 96-dpi pixels;
                      # assuming px put every panel at 75% of its true size
W = 183 * MM
COL = W / 3

# (stem, x in mm from the left, row).  Panels are placed at their NATIVE size:
# each was rendered at its target width, so nothing is rescaled here and the
# type stays at the size it was set in.
PANELS = [("f05a_spillover", 0.0, 0), ("f05b_forest", 61.0, 0),
          ("f05c_substitution", 122.0, 0),
          ("f05d_fiscal", 0.0, 1), ("f05e_burden", 91.5, 1)]
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def size_of(root):
    def num(v):
        m = re.match(r"([\d.]+)", v or "")
        return float(m.group(1)) if m else None
    w, h = num(root.get("width")), num(root.get("height"))
    if w is None or h is None:
        vb = [float(x) for x in root.get("viewBox").split()]
        w, h = vb[2], vb[3]
    return w, h


def namespace_ids(root, prefix):
    """Prefix every id and every reference to one, for a single panel.

    Matplotlib names its clip paths and glyphs line2d_1, clip1, xtick_3 and so
    on, restarting from 1 in every file.  Merging five panels left 308 ids
    defined more than once, and a renderer resolves url(#clip1) to the first
    definition it meets, so four panels were clipped by the first panel's axes
    box and their spines and contents disappeared.  Prefixing makes every id
    unique before the merge.
    """
    for el in root.iter():
        if "id" in el.attrib:
            el.set("id", prefix + el.attrib["id"])
        for k, v in list(el.attrib.items()):
            if not isinstance(v, str):
                continue
            if "url(#" in v:
                el.set(k, re.sub(r"url\(#([^)]+)\)", lambda m: "url(#" + prefix + m.group(1) + ")", v))
            elif k.split("}")[-1] == "href" and v.startswith("#"):
                el.set(k, "#" + prefix + v[1:])
    return root


rows_h = [0.0, 0.0]
loaded = []
for _i, (stem, cx, cy) in enumerate(PANELS):
    r = namespace_ids(ET.parse(SRC / f"{stem}.svg").getroot(), "p%d_" % _i)
    w, h = size_of(r)
    loaded.append((stem, cx, cy, r, w, h))
    rows_h[cy] = max(rows_h[cy], h)

GAP = 3 * MM
H = rows_h[0] + GAP + rows_h[1]
out = ET.Element("{http://www.w3.org/2000/svg}svg",
                 {"width": f"{W}", "height": f"{H}", "viewBox": f"0 0 {W} {H}"})
for stem, cx, cy, r, w, h in loaded:
    y0 = 0.0 if cy == 0 else rows_h[0] + GAP
    g = ET.SubElement(out, "{http://www.w3.org/2000/svg}g",
                      {"transform": f"translate({cx*MM},{y0})"})
    for child in list(r):
        g.append(child)

ET.ElementTree(out).write(SRC / "f05_composite.svg", xml_declaration=True,
                          encoding="utf-8")
print(f"wrote {SRC / 'f05_composite.svg'}  ({W/MM:.0f} x {H/MM:.0f} mm)")
for i, (stem, cx, cy) in enumerate(PANELS, 1):
    w, h = size_of(ET.parse(SRC / f"{stem}.svg").getroot())
    print(f"  {i}  row {cy+1}  x={cx:5.1f} mm   {w/MM:5.1f} x {h/MM:5.1f} mm   {stem}")
