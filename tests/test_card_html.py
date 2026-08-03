"""Tests for the HTML rendering of a report card: structure, escaping, and label fitting."""

from dataclasses import replace

import pytest

from sarizard.analysis import card_html
from sarizard.analysis.card_html import HtmlCard


@pytest.fixture
def card() -> HtmlCard:
    """A two-column card with one dataset group, a divider rule, and an AVERAGE row."""
    return HtmlCard(
        row_labels=["CLint HLM", "AVERAGE"],
        col_labels=["chemeleon\nbaseline", "osmordred"],
        text=[["0.500\n±0.010", "0.600\n±0.020"], ["0.550", "0.650"]],
        color=[["#ffffff", "#00ff00"], ["#eeeeee", "#00ee00"]],
        light_text=[[False, True], [False] * 2],
        groups=[(0, 1, "ASAP\n(predefined)")],
        divider_cols=[1],
        average_row=1,
        emphasis_rows=[],
        legend_stops=[(0.0, "#ff0000"), (1.0, "#00ff00")],
        legend_ticks=[(0.0, "0.0"), (1.0, "1.0")],
        title="report card r2 reduced",
    )


def test_renders_one_cell_per_value(card):
    assert card_html.render(card).count('class="cell') == 4


def test_divider_rules_the_flavor_block(card):
    doc = card_html.render(card)

    assert doc.count('divide"') == 3  # the header cell and the two body cells under it


def test_group_bracket_spans_its_rows(card):
    assert '<th class="group box" rowspan="1"' in card_html.render(card)


def test_average_row_is_bold_and_carries_no_rule_above_it(card):
    assert '<tr class="average">' in card_html.render(card)


def test_emphasis_row_rules_the_row_below_it(card):
    emphasized = replace(card, emphasis_rows=[0])

    assert '<tr class="average rule">' in card_html.render(emphasized)


def test_bracket_column_encloses_the_average_row(card):
    # an unlabelled bracket cell below the dataset groups, open where the group above closes
    assert '<th class="group box open"></th>' in card_html.render(card)


def test_cell_carries_a_hover_tooltip_naming_its_row_and_column(card):
    assert "CLint HLM · osmordred: 0.600 ±0.020" in card_html.render(card)


def test_tooltip_keeps_a_two_line_column_label_on_one_line(card):
    assert "CLint HLM · chemeleon baseline: 0.500 ±0.010" in card_html.render(card)


def test_dark_cell_text_flips_to_white(card):
    assert 'class="cell light' in card_html.render(card)


def test_missing_value_renders_as_a_grey_cell():
    blank = HtmlCard(
        row_labels=["CLint HLM"],
        col_labels=["osmordred"],
        text=[[""]],
        color=[[""]],
        light_text=[[False]],
        groups=[(0, 1, "ASAP\n(predefined)")],
        divider_cols=[],
        average_row=0,
        emphasis_rows=[],
        legend_stops=[(0.0, "#ff0000")],
        legend_ticks=[(0.0, "0.0")],
        title="t",
    )
    assert "cell missing" in card_html.render(blank)


def test_labels_are_html_escaped():
    escaped = HtmlCard(
        row_labels=["a<b> & c"],
        col_labels=["x"],
        text=[["1"]],
        color=[["#ffffff"]],
        light_text=[[False]],
        groups=[],
        divider_cols=[],
        average_row=0,
        emphasis_rows=[],
        legend_stops=[(0.0, "#ff0000")],
        legend_ticks=[(0.0, "0.0")],
        title="t",
    )
    doc = card_html.render(escaped)
    assert "a&lt;b&gt; &amp; c" in doc
    assert "<b>" not in doc


@pytest.mark.parametrize(
    ("span", "expect_shrunk"),
    [(1, True), (12, False)],
)
def test_group_label_shrinks_only_when_its_box_is_short(span, expect_shrunk):
    size = card_html._group_label_px("ChEMBL 37\n(cluster)", span)
    assert (size < card_html._GROUP_LABEL_PX) is expect_shrunk


def test_write_creates_the_file(card, tmp_path):
    out = tmp_path / "nested" / "card.html"
    card_html.write(card, out)
    assert out.read_text().startswith("<!doctype html>")
