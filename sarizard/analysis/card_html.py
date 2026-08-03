"""HTML rendering of a report card, alongside the PNG that :mod:`report_card` draws.

The HTML is a real table rather than an embedded image: cells carry the same colors, values,
and error bars as the PNG, but the text stays selectable and searchable, the figure scales to
the window instead of to 300 dpi, and each cell gets a hover tooltip naming its endpoint and
column. Long cards keep their column headers and row labels pinned while the grid scrolls.

The layout mirrors the PNG deliberately, so the two read as the same card: the dataset group
boxes down the left margin, the heavy rule dividing the baseline from the block it is compared
against, the AVERAGE row under its rule, and the colorbar as a gradient legend. Nothing
here recomputes a number or a color; :func:`report_card.plot_card` passes the values it has
already resolved, so the two renderings cannot drift.

The page is self-contained (inline CSS, no scripts or external assets), so it opens from the
filesystem and survives being copied somewhere else on its own.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# grid geometry in CSS pixels, in the same proportion as the PNG's cell inches
_CELL_WIDTH_PX = 84
_CELL_HEIGHT_PX = 44
_GROUP_COL_PX = 34
_LABEL_COL_PX = 190

# the rule dividing the reference column from the block compared against it, in the PNG's weight
_DIVIDER_PX = 3

# gradient legend: how finely to sample the colormap, and the bar's size in CSS pixels
_LEGEND_SAMPLES = 48
_LEGEND_WIDTH_PX = 260
_LEGEND_HEIGHT_PX = 14

_MISSING_COLOR = "#d3d3d3"  # matplotlib's lightgrey, for a (column, endpoint) with no result

# fitting a dataset bracket's upright label to its own height, as the PNG does: the label's
# nominal size, the floor it may shrink to, the average width of a bold glyph in em, and the
# fraction of the box it may fill. A single-row group has one row's height to work with, which a
# full-size label overruns into its neighbours'
_GROUP_LABEL_PX = 11
_GROUP_LABEL_MIN_PX = 6
_BOLD_EM_WIDTH = 0.68
_GROUP_LABEL_SLACK = 0.85


@dataclass(frozen=True)
class HtmlCard:
    """One card's fully resolved contents, ready to write as HTML.

    Every field is what the PNG renderer already computed, so the two renderings agree by
    construction. Cell fields are row-major and indexed identically to the plotted matrix.

    Attributes
    ----------
    row_labels, col_labels : list of str
        Tick labels as drawn, underscores already spaced out.
    text : list of list of str
        Cell annotation, ``""`` where the cell carries no value. A newline separates the value
        from its error bar or p-value, as in the PNG.
    color : list of list of str
        Cell background as a hex string; a missing value takes :data:`_MISSING_COLOR`.
    light_text : list of list of bool
        Whether the cell's text flips to white, by the PNG's own contrast rule.
    groups : list of (int, int, str)
        ``(start, end, label)`` runs over the endpoint rows, the label carrying the dataset
        display name and its split strategy on two lines.
    divider_cols : list of int
        Columns whose left edge carries the heavy dividing rule.
    average_row : int
        Row index of the AVERAGE row, which is set bold and takes its own bracket compartment.
    emphasis_rows : list of int
        Row indices after which a heavier rule is drawn (the endpoint the study leans on).
    legend_stops : list of (float, str)
        Gradient stops as ``(position in 0-1, hex)``, ordered from the ramp's low end.
    legend_ticks : list of (float, str)
        Legend tick positions in 0-1 with their labels.
    title : str
        Document title, shown in the browser tab only; the page itself carries no heading, so
        it matches the PNG.
    """

    row_labels: list[str]
    col_labels: list[str]
    text: list[list[str]]
    color: list[list[str]]
    light_text: list[list[bool]]
    groups: list[tuple[int, int, str]]
    divider_cols: list[int]
    average_row: int
    emphasis_rows: list[int]
    legend_stops: list[tuple[float, str]]
    legend_ticks: list[tuple[float, str]]
    title: str


_STYLE = f"""
:root {{ color-scheme: light; }}
body {{
  margin: 0; padding: 24px;
  background: #ffffff; color: #111111;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}}
.legend {{ margin: 0 0 18px 0; }}
.legend-bar {{
  width: {_LEGEND_WIDTH_PX}px; height: {_LEGEND_HEIGHT_PX}px;
  border: 1px solid #111111;
}}
.legend-ticks {{
  position: relative; width: {_LEGEND_WIDTH_PX}px; height: 1.2em;
  font-size: 11px; color: #444444;
}}
.legend-ticks span {{ position: absolute; white-space: nowrap; }}
/* the grid scrolls inside its own box, so the page itself never scrolls sideways */
.card {{ overflow: auto; max-height: 90vh; max-width: 100%; }}
table {{ border-collapse: separate; border-spacing: 0; }}
th, td {{ padding: 0; margin: 0; }}
/* column headers and the two label columns stay put while the grid scrolls under them */
thead th {{
  position: sticky; top: 0; z-index: 3;
  background: #ffffff; vertical-align: bottom;
  font-size: 12px; font-weight: 700; text-align: center;
  padding: 0 4px 6px 4px;
}}
th.group, th.label {{ position: sticky; z-index: 2; background: #ffffff; }}
th.group {{ left: 0; width: {_GROUP_COL_PX}px; min-width: {_GROUP_COL_PX}px; }}
th.label {{
  left: {_GROUP_COL_PX}px; width: {_LABEL_COL_PX}px; min-width: {_LABEL_COL_PX}px;
  text-align: right; font-weight: 400; font-size: 12px; padding-right: 8px;
}}
thead th.group, thead th.label {{ z-index: 4; }}
/* the dataset bracket: name over split strategy, set upright in the left margin */
th.group.box {{
  border: 1px solid #111111;
  writing-mode: vertical-rl; transform: rotate(180deg);
  font-weight: 700; white-space: pre; text-align: center; overflow: hidden;
}}
td.cell {{
  width: {_CELL_WIDTH_PX}px; min-width: {_CELL_WIDTH_PX}px; height: {_CELL_HEIGHT_PX}px;
  border: 1px solid #ffffff;
  text-align: center; font-size: 12px; line-height: 1.15;
  font-variant-numeric: tabular-nums;
}}
td.cell .aux {{ font-size: 11px; }}
td.cell.light {{ color: #ffffff; }}
td.missing {{ background: {_MISSING_COLOR}; }}
/* the reference column's divider, matching the PNG's rule rather than a gap */
td.divide, th.divide {{ border-left: {_DIVIDER_PX}px solid #111111; }}
/* group boundaries, the AVERAGE rule, and the emphasis rule, as on the PNG */
tr.group-top td.cell, tr.group-top th.label {{ border-top: 1px solid #111111; }}
tr.rule td.cell, tr.rule th.label {{ border-top: 2px solid #111111; }}
tr.average th.label, tr.average td.cell {{ font-weight: 700; }}
/* AVERAGE's compartment is open at the top: the group above it closes the endpoint block */
th.group.box.open {{ border-top: none; }}
/* hover: ring the cell under the pointer without moving anything */
td.cell:hover {{ outline: 2px solid #111111; outline-offset: -2px; }}
"""


def _group_label_px(label: str, span: int) -> float:
    """Shrink a bracket's label until its longest line fits the rows the bracket covers."""
    longest = max((len(line) for line in label.splitlines()), default=1)
    available = span * _CELL_HEIGHT_PX * _GROUP_LABEL_SLACK
    return max(_GROUP_LABEL_MIN_PX, min(_GROUP_LABEL_PX, available / (_BOLD_EM_WIDTH * longest)))


def _cell_html(card: HtmlCard, row: int, col: int) -> str:
    """Render one grid cell, blank where the card has no value there."""
    divide = " divide" if col in card.divider_cols else ""
    text = card.text[row][col]
    color = card.color[row][col]
    if not text and not color:
        return f'<td class="cell missing{divide}" style="background:{_MISSING_COLOR}"></td>'

    # value on the first line, error bar or p-value under it, as the PNG stacks them
    value, _, aux = text.partition("\n")
    body = html.escape(value)
    if aux:
        body += f'<br><span class="aux">{html.escape(aux)}</span>'
    classes = ("cell light" if card.light_text[row][col] else "cell") + divide
    # the tooltip is one line, so the line breaks that stack a two-line column header or a cell's
    # error bar become spaces rather than riding into the attribute
    row_label, col_label = card.row_labels[row], card.col_labels[col]
    tooltip = " ".join(f"{row_label} · {col_label}: {text}".split())
    return (
        f'<td class="{classes}" style="background:{color}" title="{html.escape(tooltip)}">'
        f"{body}</td>"
    )


def _legend_html(card: HtmlCard) -> str:
    """Render the colorbar as a gradient bar with its tick labels underneath."""
    stops = ", ".join(f"{color} {position * 100:.1f}%" for position, color in card.legend_stops)
    # a tick centers on its position, except at the ends, where centering would hang the label
    # off the bar and the page edge clips it
    ticks = ""
    for position, label in card.legend_ticks:
        shift = "0" if position <= 0.0 else ("-100%" if position >= 1.0 else "-50%")
        ticks += (
            f'<span style="left:{position * 100:.1f}%;transform:translateX({shift})">'
            f"{html.escape(label)}</span>"
        )
    return (
        '<div class="legend">'
        f'<div class="legend-bar" style="background:linear-gradient(to right, {stops})"></div>'
        f'<div class="legend-ticks">{ticks}</div>'
        "</div>"
    )


def _header_html(card: HtmlCard) -> str:
    """Render the sticky header row: two label columns, then the column names."""
    cells = ['<th class="group"></th>', '<th class="label"></th>']
    for col, label in enumerate(card.col_labels):
        divide = ' class="divide"' if col in card.divider_cols else ""
        cells.append(f"<th{divide}>{html.escape(label).replace(chr(10), '<br>')}</th>")
    return "<thead><tr>" + "".join(cells) + "</tr></thead>"


def _row_html(card: HtmlCard, row: int, group_starts: dict[int, tuple[int, str]]) -> str:
    """Render one table row, opening a dataset bracket where a group starts.

    Below the last group the bracket column carries on unlabelled around AVERAGE, so the summary
    label is enclosed by the same column as the endpoint labels.
    """
    below_groups = row >= max((end for _, end, _ in card.groups), default=0)
    classes = []
    if row in group_starts:
        classes.append("group-top")
    if row == card.average_row:
        classes.append("average")
    if row - 1 in card.emphasis_rows:
        classes.append("rule")
    cells = []
    if row in group_starts:
        span, label = group_starts[row]
        cells.append(
            f'<th class="group box" rowspan="{span}" '
            f'style="font-size:{_group_label_px(label, span):.1f}px">{html.escape(label)}</th>'
        )
    elif below_groups:
        cells.append('<th class="group box open"></th>')
    elif not any(start <= row < end for start, end, _ in card.groups):
        cells.append('<th class="group"></th>')
    cells.append(f'<th class="label">{html.escape(card.row_labels[row])}</th>')
    cells.extend(_cell_html(card, row, col) for col in range(len(card.col_labels)))
    attrs = f' class="{" ".join(classes)}"' if classes else ""
    return f"<tr{attrs}>" + "".join(cells) + "</tr>"


def render(card: HtmlCard) -> str:
    """Return the complete HTML document for one card.

    Parameters
    ----------
    card : HtmlCard
        The card's resolved cells, labels, and legend.

    Returns
    -------
    str
        A self-contained HTML document: inline styles, no scripts, no external assets.
    """
    # a group's bracket is opened by its first row and spans the rest, matching the PNG's box
    group_starts = {start: (end - start, label) for start, end, label in card.groups}
    rows = "".join(_row_html(card, row, group_starts) for row in range(len(card.row_labels)))
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(card.title)}</title>\n"
        f"<style>{_STYLE}</style>\n</head>\n<body>\n"
        f"{_legend_html(card)}\n"
        f'<div class="card"><table>{_header_html(card)}<tbody>{rows}</tbody></table></div>\n'
        "</body>\n</html>\n"
    )


def write(card: HtmlCard, out_html: Path) -> None:
    """Write one card's HTML document to ``out_html``."""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render(card), encoding="utf-8")
    logger.info("wrote %s", out_html)
