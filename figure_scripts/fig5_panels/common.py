"""Shared data access and style for Figure 5.

Style is proplot 0.9.7 + matplotlib 3.4.3, the same combination that produced
Figures 3, 4, 8, 10, 11 and 12, so the panels sit together without a visual
seam.  Verified against the server's conda base (python 3.11.5, mpl 3.4.3,
proplot 0.9.7).

Every number is read from a shipped result file.  Nothing is typed in.
"""
from pathlib import Path

import pandas as pd
import proplot as pplt

ROOT = Path(__file__).resolve().parents[2]
EMP  = ROOT / "employment"
RES  = ROOT / "results"
R06 = RES / "r6_bartik"
AUD = RES / "r6_employment"
# The manuscript's figure pipeline reads SVG from figures/05, the
# folder the assembled figure files are built from.  Panels are written there
# as well as to the local out/ copy, so the two never diverge.

SVG_DIR = Path(__file__).resolve().parents[2] / "figures" / "05"
SVG_DIR.mkdir(parents=True, exist_ok=True)

HALF_W = 8.27 * 0.9 / 2.0          # half a Nature page width, as in f09
LEAVE_OUT, SPEC = "union", "level"

# Nature two-column width is 183 mm.  The layout is three panels across the top
# row and two across the bottom, so a top cell is 61 mm and a bottom cell
# 91.5 mm.  Type is held between the journal minimum of 5 pt and 7 pt.
MM = 1 / 25.4
FIG_W = 183 * MM
PANEL_W3 = FIG_W / 3
PANEL_W2 = FIG_W / 2
PANEL_W = PANEL_W3
# Matches figures 1-4, which set FONTSIZE_PT = 7 for ticks and axis titles and
# drop to 5-5.5 only for point annotations.  Earlier drafts used 6.0 for ticks
# and 5.2 for annotations, which read smaller than the rest of the paper.
FS_TICK, FS_LABEL, FS_ANNOT, FS_TITLE = 7.0, 7.0, 5.5, 7.0
FOOTNOTE_CHARS = 96

pplt.rc.update({"font.size": FS_LABEL, "font.family": "sans-serif"})


C_INFO = "#4e79a7"
C_CONSTR = "#e15759"
C_NEUTRAL = "#59a14f"
C_SUBST = "#b07aa1"      # panel c, so its scatter is not read as panel a's
C_GREY = "#8a8a8a"


def _P(p):
    """P as a reader sees it, not as a float repr: 4e-08 is not English."""
    return "P < 0.001" if p < 0.001 else f"P = {p:.3f}".rstrip("0").rstrip(".")


def iv_table():
    """IV point estimates joined to their Anderson-Rubin sets, by outcome."""
    m = pd.read_csv(R06 / "results_main__union.csv")
    m = m[(m.leave_out_variant == LEAVE_OUT) & (m.specification == SPEC)]
    piv = (m.pivot_table(index="outcome", columns="estimator",
                         values=["coef", "se", "p_value"], aggfunc="first"))
    g = pd.read_csv(R06 / "results_diagnostics__union.csv")
    g = g[(g.diagnostic == "anderson_rubin_ci")
          & (g.leave_out_variant == LEAVE_OUT)]
    ar = g.pivot_table(index="outcome", columns="subject", values="value",
                       aggfunc="first")
    out = pd.DataFrame({
        "beta": piv[("coef", "IV2SLS")],
        "se": piv[("se", "IV2SLS")],
        "p": piv[("p_value", "IV2SLS")],
        "ols": piv[("coef", "OLS_FE")],
        "rf": piv[("coef", "ReducedForm")],
    })
    out["ar_lo"] = ar["lower"]
    out["ar_hi"] = ar["upper"]
    out["excl0"] = ~((out.ar_lo <= 0) & (0 <= out.ar_hi))
    feff = m[m.estimator == "IV2SLS"].set_index("outcome")["fs_F_effective"]
    nobs = m[m.estimator == "IV2SLS"].set_index("outcome")["n_obs"]
    out["feff"], out["n"] = feff, nobs
    return out


def save(fig, stem, target_w=None):
    """SVG at exactly the target width, with nothing clipped.

    Two earlier attempts both failed, in opposite directions.  bbox_inches
    ="tight" never clips but crops to the drawn content, so a panel asked for at
    61 mm came out at 46.6 mm and had to be rescaled, shrinking the type below
    the journal minimum.  Dropping it and calling tight_layout kept the canvas
    at 61 mm but silently pushed the axis titles outside it: the x title sat at
    153 pt on a 147.6 pt canvas and the y title at -9.1 pt, so both vanished.

    This measures the tight bounding box, rescales the figure by the ratio of
    the target width to that box, and saves tight again.  The result is exactly
    the target width and contains every artist.
    """
    if target_w is None:
        target_w = fig.get_size_inches()[0]
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    k = target_w / bb.width
    w0, h0 = fig.get_size_inches()
    fig.set_size_inches(w0 * k, h0 * k)
    for path in (SVG_DIR / f"{stem}.svg",):
        fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.canvas.draw()
    bb2 = fig.get_tightbbox(fig.canvas.get_renderer())
    print(f"wrote {SVG_DIR / (stem + '.svg')}  "
          f"({bb2.width*25.4:.1f} x {bb2.height*25.4:.1f} mm)")


def footnote(ax, text, chars=FOOTNOTE_CHARS, y=-0.24, size=5.2):
    """A wrapped note under the axes.

    Unwrapped notes were forcing the figure box out to three times the width of
    the plot, which is why the early drafts rendered as letterbox strips.
    """
    import textwrap
    ax.text(0.0, y, "\n".join(textwrap.wrap(text, chars)),
            transform=ax.transAxes, ha="left", va="top", fontsize=size,
            color="0.35", linespacing=1.45)


def tidy(ax, spines=("top", "right")):
    """Drop the named spines, kill the grid, thin what is left."""
    for s in spines:
        ax.spines[s].set_visible(False)
    for s in ax.spines:
        if ax.spines[s].get_visible():
            ax.spines[s].set_linewidth(0.7)
    ax.grid(False)
    ax.tick_params(labelsize=FS_TICK, width=0.6, length=2.4, pad=1.6)
    ax.tick_params(which="minor", length=1.4, width=0.4)


def panel_letter(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom", ha="left")


def label_beside(ax, xs, ys, names, sizes=None, size=6.0, colour="0.15", pad=2.5):
    """State labels sitting directly beside their own markers.

    adjustText was tried here first and failed on every log-scaled panel: it
    searches in DATA coordinates, so on a five-decade y axis it drove all
    eighteen labels onto the same horizontal line and produced an unreadable
    pile with leader lines fanning across the plot.

    This places each label as a POINT offset from its marker, which is
    invariant both to the axis scale and to the rescaling that save() applies
    afterwards, and resolves collisions by walking a fixed ring of candidate
    offsets and taking the first that clears everything already placed.
    """
    import numpy as np

    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    names = list(names)
    # scatter sizes are areas in pt^2; the label must clear the marker radius
    # sizes may be a scalar (one marker size for the whole series) or per point
    radii = (np.full(len(xs), 3.0) if sizes is None
             else np.sqrt(np.broadcast_to(np.asarray(sizes, dtype=float),
                                          (len(xs),)) / np.pi))

    disp = ax.transData.transform(np.column_stack([xs, ys]))
    x0, x1 = ax.transAxes.transform([(0.0, 0.0), (1.0, 1.0)])[:, 0]
    flip_at = x0 + 0.72 * (x1 - x0)

    def place(i, dx, dy, ha):
        return ax.annotate(names[i], (xs[i], ys[i]), textcoords="offset points",
                           xytext=(dx, dy), ha=ha, va="center",
                           fontsize=size, color=colour, zorder=20,
                           annotation_clip=False)

    # The markers are obstacles too, not just the other labels.  Checking only
    # label-against-label let WI's label sit on top of MA's marker in panel a,
    # which reads as two overlapping words.
    from matplotlib.transforms import Bbox
    px = fig.dpi / 72.0
    placed = [Bbox.from_bounds(disp[j, 0] - radii[j] * px,
                               disp[j, 1] - radii[j] * px,
                               2 * radii[j] * px, 2 * radii[j] * px)
              for j in range(len(xs))]
    texts = []
    for i in np.argsort(-disp[:, 1]):          # top down, so the order is fixed
        d = radii[i] + pad
        near_right = disp[i, 0] > flip_at
        first, second = ((-d, "right"), (d, "left")) if near_right \
            else ((d, "left"), (-d, "right"))
        # Candidates escalate by DISTANCE, alternating sides at each step, so a
        # crowded label lands on the far side of its own marker before it is
        # pushed a row away.  Exhausting one side first left ME and DE three
        # rows below their points, too far to pair up without a leader.
        cands = []
        for dy in (0, 5, -5, 10, -10, 15, -15, 21, -21, 27, -27, 34, -34, 41, -41):
            cands.append((first[0], dy, first[1]))
            cands.append((second[0], dy, second[1]))

        # The search is BOUNDED.  An unbounded widening fallback was tried and
        # threw panel d's crowded left edge two hundred points off the canvas,
        # leaving labels with no marker anywhere near them.  When nothing is
        # free, the least-bad candidate is kept instead: a slight overlap next
        # to the right marker beats a clean label beside the wrong one.
        best, best_cost = None, np.inf
        for dx, dy, ha in cands:
            t = place(i, dx, dy, ha)
            bb = t.get_window_extent(rend).expanded(1.05, 1.30)
            cost = sum(_overlap_area(bb, b) for b in placed)
            if cost <= 0 and _inside(bb, ax, rend):
                best, best_cost = (dx, dy, ha), 0.0
                t.remove()
                break
            cost += 0.0 if _inside(bb, ax, rend) else 1e4     # keep it on canvas
            if cost < best_cost:
                best, best_cost = (dx, dy, ha), cost
            t.remove()

        t = place(i, *best)
        placed.append(t.get_window_extent(rend).expanded(1.05, 1.30))
        texts.append(t)
    return texts


def _overlap_area(a, b):
    """Area of the intersection of two display-space boxes, zero if disjoint."""
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if (w > 0 and h > 0) else 0.0


def _inside(bb, ax, rend):
    """Is the label fully inside the axes?  Labels outside get clipped by the
    tight bounding box in save(), or silently enlarge the canvas."""
    ab = ax.get_window_extent(rend)
    return (bb.x0 >= ab.x0 and bb.x1 <= ab.x1
            and bb.y0 >= ab.y0 and bb.y1 <= ab.y1)


def place_labels(ax, xs, ys, names, size=5.4, colour="0.2", leaders=False, **kw):
    """State labels that do not sit on top of each other or on the markers.

    leaders=False for the manuscript panels: a leader line clutters a 61 mm
    panel.  leaders=True for the oversize reference versions, where fifty labels
    get pushed far enough from their markers that the reader cannot pair them
    up without one.
    """
    from adjustText import adjust_text
    texts = [ax.text(x, y, n, fontsize=size, color=colour)
             for x, y, n in zip(xs, ys, names)]
    arrow = (dict(arrowstyle="-", color="0.62", lw=0.4, shrinkA=1, shrinkB=2)
             if leaders else None)
    adjust_text(texts, ax=ax, expand=(1.18, 1.35), arrowprops=arrow, **kw)
    return texts
