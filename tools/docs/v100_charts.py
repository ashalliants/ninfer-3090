#!/usr/bin/env python3
"""Render the comparison charts in docs/assets/ from a chart data file.

  python tools/docs/v100_charts.py                 the Tesla V100 charts (v100-*.svg, v100_charts.json)
  python tools/docs/v100_charts.py --data tools/docs/rtx3090_charts.json
                                                   the RTX 3090 chart (rtx3090-*.svg)
  python tools/docs/v100_charts.py --out DIR       write them somewhere else

Every number comes from the JSON, which names the build, PR or pinned source behind it
(docs/tesla-v100.md#methodology-and-sources has the methodology). Speed-ups on the headline chart are
computed here from the raw per-prompt figures in the JSON (geometric mean of ours / theirs), so they
cannot drift from the per-model charts. The SVGs carry their own light and dark colours
(prefers-color-scheme) on an explicit background, so they read on both GitHub themes and on Hugging
Face. After regenerating, run `python tools/sync_model_cards.py --check` and copy any chart a model
card uses into that card's assets/. Standard library only.
"""
import argparse
import json
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = Path(__file__).with_suffix(".json")

W = 960
PAD = 32
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

# Reference categorical palette (dataviz skill, slots 1-3, validated all-pairs in both modes):
# blue = ours (main configuration), orange = a competitor measured on this box, aqua = ours (second
# configuration). Neutral grey = our own earlier build; grey hatching = a third-party reported figure.
STYLE = """
  <style>
    .surface {{ fill: #fcfcfb; stroke: rgba(11,11,11,0.10); }}
    .t1 {{ fill: #0b0b0b; }} .t2 {{ fill: #52514e; }} .t3 {{ fill: #6f6e69; }}
    .grid {{ stroke: #e6e5df; }} .base {{ stroke: #c3c2b7; }} .ref {{ stroke: #52514e; stroke-dasharray: 4 3; }}
    .ours {{ fill: #2a78d6; }} .alt {{ fill: #1baf7a; }} .same {{ fill: #eb6834; }} .prev {{ fill: #b5b3aa; }}
    .rep {{ fill: url(#hatch-{uid}); }} .hatchbg {{ fill: #e4e2da; }} .hatchln {{ stroke: #8f8d86; }}
    .win {{ fill: #1d5fb0; }} .loss {{ fill: #b8471d; }}
    text {{ font-family: {font}; }}
    @media (prefers-color-scheme: dark) {{
      .surface {{ fill: #1a1a19; stroke: rgba(255,255,255,0.12); }}
      .t1 {{ fill: #f4f3ef; }} .t2 {{ fill: #c3c2b7; }} .t3 {{ fill: #a3a29b; }}
      .grid {{ stroke: #2e2e2b; }} .base {{ stroke: #4a4a46; }} .ref {{ stroke: #c3c2b7; }}
      .ours {{ fill: #3987e5; }} .alt {{ fill: #199e70; }} .same {{ fill: #d95926; }} .prev {{ fill: #6f6e69; }}
      .hatchbg {{ fill: #3a3935; }} .hatchln {{ stroke: #8f8d86; }}
      .win {{ fill: #79b0f0; }} .loss {{ fill: #f08a5d; }}
    }}
  </style>"""

KIND_CLASS = {"ours": "ours", "alt": "alt", "same": "same", "reported": "rep", "prev": "prev"}


def fmt(v):
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 100:
        return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"
    return f"{v:.1f}"


def text_w(s, size, weight=400):
    # Generous estimate (DejaVu Sans, the widest likely fallback), so labels never collide.
    return len(s) * size * (0.66 if weight >= 600 else 0.58)


def nice_max(v):
    step = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * step >= v:
            return m * step
    return 10 * step


def ticks(vmax):
    raw = vmax / 5
    step = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * step >= raw:
            step = m * step
            break
    return [i * step for i in range(int(vmax / step + 1e-9) + 1)]


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


class Svg:
    def __init__(self, uid):
        self.uid = uid
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, cls="t1", size=13, weight=400, anchor="start"):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" font-weight="{weight}" '
                 f'text-anchor="{anchor}">{escape(s)}</text>')

    def wrap(self, x, y, s, width, cls="t3", size=11.5, line=15, weight=400):
        words, cur, lines = s.split(), "", []
        for w in words:
            if cur and text_w(cur + " " + w, size, weight) > width:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            lines.append(cur)
        for i, ln in enumerate(lines):
            self.text(x, y + i * line, ln, cls, size, weight)
        return y + len(lines) * line

    def bar(self, x, y, w, h, kind, tip):
        cls = KIND_CLASS[kind]
        r = min(4, h / 2, max(w, 0.1) / 2)
        # square at the baseline, rounded at the data end
        self.add(f'<path class="{cls}" d="M{x:.1f},{y:.1f} h{max(w - r, 0):.1f} a{r},{r} 0 0 1 {r},{r} '
                 f'v{h - 2 * r:.1f} a{r},{r} 0 0 1 {-r},{r} h{-max(w - r, 0):.1f} z"><title>{escape(tip)}</title></path>')

    def render(self, height, title, desc):
        defs = (f'<defs><pattern id="hatch-{self.uid}" width="6" height="6" patternUnits="userSpaceOnUse" '
                f'patternTransform="rotate(45)"><rect width="6" height="6" class="hatchbg"/>'
                f'<line x1="0" y1="0" x2="0" y2="6" class="hatchln" stroke-width="2"/></pattern></defs>')
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height:.0f}" width="{W}" height="{height:.0f}" '
                f'role="img" aria-label="{escape(desc)}">\n  <title>{escape(title)}</title>\n  <desc>{escape(desc)}</desc>')
        style = STYLE.format(uid=self.uid, font=FONT)
        bg = f'<rect x="1" y="1" width="{W - 2}" height="{height - 2:.0f}" rx="14" class="surface" stroke-width="1"/>'
        return "\n".join([head, style, "  " + defs, "  " + bg] + ["  " + p for p in self.parts] + ["</svg>", ""])


def legend(svg, y, series):
    x = PAD
    for s in series:
        label = s["name"]
        wdt = 14 + 6 + text_w(label, 12.5) + 18
        if x + wdt > W - PAD:
            x = PAD
            y += 20
        svg.bar(x, y - 10, 14, 12, s["kind"], label)
        svg.text(x + 20, y, label, "t2", 12.5)
        x += wdt
    return y + 12


def panel_head(svg, y, panel):
    if panel.get("title"):
        svg.text(PAD, y, panel["title"], "t1", 15, 600)
        y += 8
    if panel.get("note"):
        y = svg.wrap(PAD, y + 12, panel["note"], W - 2 * PAD, "t3", 11.5) - 4
    return y


def hbar_panel(svg, y, panel):
    """Grouped horizontal bars: one group per prompt, one bar per series (None = not drawn)."""
    y = panel_head(svg, y, panel)
    y = legend(svg, y + 20, panel["series"]) + 8
    label_w = panel.get("label_w", 120)
    x0, x1 = PAD + label_w, W - PAD - 120
    vmax = nice_max(max(v for s in panel["series"] for v in s["values"] if v is not None) * 1.02)
    bh, gap, ggap = 13, 2, 13
    top = y
    layout, gy = [], top
    for gi, _ in enumerate(panel["groups"]):
        present = [si for si, s in enumerate(panel["series"]) if s["values"][gi] is not None]
        gh = len(present) * bh + (len(present) - 1) * gap
        layout.append((gy, present, gh))
        gy += gh + ggap
    height = gy - ggap - top
    for t in ticks(vmax):
        x = x0 + (x1 - x0) * t / vmax
        svg.add(f'<line x1="{x:.1f}" y1="{top - 4:.1f}" x2="{x:.1f}" y2="{top + height + 4:.1f}" class="{"base" if t == 0 else "grid"}" stroke-width="1"/>')
        svg.text(x, top + height + 18, f"{t:,.0f}", "t3", 11, anchor="middle")
    for gi, g in enumerate(panel["groups"]):
        gy, present, gh = layout[gi]
        svg.text(x0 - 10, gy + gh / 2 + 4.5, g, "t2", 12.5, anchor="end")
        for k, si in enumerate(present):
            s = panel["series"][si]
            v = s["values"][gi]
            by = gy + k * (bh + gap)
            w = (x1 - x0) * v / vmax
            extra = (s.get("labels") or [None] * len(s["values"]))[gi]
            svg.bar(x0, by, w, bh, s["kind"], f'{s["name"]}: {g} {fmt(v)} {panel.get("unit", "tok/s")}' + (f" ({extra})" if extra else ""))
            strong = s["kind"] in ("ours", "alt")
            lbl = fmt(v)
            svg.text(x0 + w + 5, by + bh - 2.5, lbl, "t1" if strong else "t2", 11.5, 600 if strong else 400)
            if extra:
                svg.text(x0 + w + 5 + text_w(lbl, 11.5, 600 if strong else 400) + 4, by + bh - 2.5, extra, "t3", 10.5)
    svg.text(x1, top + height + 34, panel.get("axis", "tokens per second, higher is better"), "t3", 11, anchor="end")
    return top + height + 40


def rows_panel(svg, y, panel):
    """One bar per row, label above the bar (long labels stay readable at phone width)."""
    y = panel_head(svg, y, panel)
    y += 16
    x0, x1 = PAD, W - PAD - 90
    vmax = nice_max(max(r["value"] for r in panel["rows"]) * 1.02)
    for r in panel["rows"]:
        strong = r["kind"] in ("ours", "alt")
        wt = 600 if strong else 400
        svg.text(x0, y + 12, r["label"], "t1" if strong else "t2", 12.5, wt)
        if r.get("detail"):
            svg.text(x0 + text_w(r["label"], 12.5, wt) + 8, y + 12, r["detail"], "t3", 11.5)
        y += 18
        w = (x1 - x0) * r["value"] / vmax
        svg.bar(x0, y, w, 14, r["kind"], f'{r["label"]} {r.get("detail", "")}: {fmt(r["value"])}')
        svg.text(x0 + w + 6, y + 11.5, fmt(r["value"]) + (" reported" if r["kind"] == "reported" else ""),
                 "t1" if strong else "t2", 12, wt)
        y += 14 + 12
    return y + 2


def ratio_cell(cell):
    if "ratio" in cell:
        return cell["ratio"], None, None
    rs = [a / b for a, b in zip(cell["ours"], cell["theirs"])]
    return geomean(rs), min(rs), max(rs)


def ratio_panel(svg, y, panel):
    """Speed-up of ours over the best other engine: one row per model, Decode and Prefill columns."""
    y = panel_head(svg, y, panel)
    cols = panel["columns"]
    vmax = panel.get("max", 1.6)
    name_w = 270
    col_gap = 28
    col_w = (W - 2 * PAD - name_w - col_gap) / 2
    bar_w = col_w - 70
    cx = [PAD + name_w + ci * (col_w + col_gap) for ci in range(len(cols))]
    y += 12
    for ci, c in enumerate(cols):
        svg.text(cx[ci], y, c, "t1", 14, 700)
    svg.text(W - PAD, y, "dashed line: 1.0x, same speed", "t3", 11, anchor="end")
    y += 14
    first = True
    for r in panel["rows"]:
        if r.get("section"):
            y += 6
            svg.text(PAD, y + 10, r["section"], "t3", 11.5, 600)
            y += 18
            continue
        if not first:
            svg.add(f'<line x1="{PAD}" y1="{y:.1f}" x2="{W - PAD}" y2="{y:.1f}" class="grid" stroke-width="1"/>')
        first = False
        y += 14
        ny = svg.wrap(PAD, y + 13, r["model"], name_w - 16, "t1", 15, 18, 700)
        ny = svg.wrap(PAD, ny + 1, r["versus"], name_w - 16, "t3", 11.5, 15)
        cell_bottom = y
        for ci, cell in enumerate(r["cells"]):
            x0 = cx[ci]
            if cell is None or cell.get("none"):
                msg = (cell or {}).get("none", "no comparable figure")
                yy = svg.wrap(x0, y + 15, msg, col_w, "t3", 11.5)
                cell_bottom = max(cell_bottom, yy)
                continue
            ratio, lo, hi = ratio_cell(cell)
            kind = "rep" if cell.get("reported") else "ours"
            w = bar_w * min(ratio, vmax) / vmax
            tip = f'{r["model"]}, {cols[ci]}: {ratio:.2f}x'
            svg.bar(x0, y + 2, w, 18, "reported" if kind == "rep" else "ours", tip)
            ref = x0 + bar_w / vmax
            svg.add(f'<line x1="{ref:.1f}" y1="{y - 3:.1f}" x2="{ref:.1f}" y2="{y + 25:.1f}" class="ref" stroke-width="1.5"/>')
            lx = max(x0 + w, ref) + 7
            svg.text(lx, y + 17, f"{ratio:.2f}x", "loss" if ratio < 1 else ("t1" if kind == "rep" else "win"), 17, 700)
            detail = cell["detail"]
            if lo is not None and len(cell["ours"]) > 1:
                detail = f"{detail} (range {lo:.2f}-{hi:.2f}x)"
            yy = svg.wrap(x0, y + 38, detail, col_w, "t3", 11, 14)
            cell_bottom = max(cell_bottom, yy)
        y = max(ny, cell_bottom) + 4
    return y + 4


def chart(spec, meta):
    svg = Svg(spec["file"].replace(".svg", "").replace(".", "-"))
    y = PAD + 16
    svg.text(PAD, y, spec["title"], "t1", 22, 700)
    y += 24
    y = svg.wrap(PAD, y, spec["subtitle"], W - 2 * PAD, "t2", 13.5, 19) + 6
    for p in spec["panels"]:
        y += 16
        kind = p["type"]
        y = {"grouped": hbar_panel, "rows": rows_panel, "ratio": ratio_panel}[kind](svg, y, p)
    y += 6
    svg.add(f'<line x1="{PAD}" y1="{y:.1f}" x2="{W - PAD}" y2="{y:.1f}" class="grid" stroke-width="1"/>')
    y += 18
    for n in spec.get("notes", []):
        y = svg.wrap(PAD, y, n, W - 2 * PAD, "t2", 11.5) + 3
    foot = (f'{meta["gpu"]} · {meta["power"]} · {spec.get("date", meta["date"])} · ours: {spec["build"]} · '
            f'competitors measured on this box unless marked “reported”')
    y = svg.wrap(PAD, y + 3, foot, W - 2 * PAD, "t3", 11.5)
    return svg.render(y + 14, spec["title"], spec.get("desc", spec["title"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "assets")
    ap.add_argument("--data", type=Path, default=DATA)
    args = ap.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in data["charts"]:
        (args.out / spec["file"]).write_text(chart(spec, data["meta"]), encoding="utf-8")
        print(args.out / spec["file"])


if __name__ == "__main__":
    main()
