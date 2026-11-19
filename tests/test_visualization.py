"""Abbildungen: alle Achsen gesperrt, die Balken zeigen die Spannen, die Experimentbilder enthalten die gemessenen Zeilen."""

import numpy as np
import pytest

import spf_evaluation as ev
import spf_visualization as V


@pytest.fixture(scope="module")
def analysis():
    return ev.analyse(ev.Settings(n=4, m=100, l=24, s=500))


def _locked(fig):
    for name in fig.layout:
        if name.startswith(("xaxis", "yaxis")):
            assert fig.layout[name].fixedrange is True, name


def test_band_chart_bars_have_the_height_of_the_spans(analysis):
    fig = V.build_bands(analysis)
    _locked(fig)
    bar = fig.data[0]
    assert list(bar.y) == pytest.approx(list(ev.team_spans(analysis.main_points)))
    assert list(bar.x) == ["M1", "M2", "M3", "M4"]
    lo, hi = fig.layout.yaxis.range
    assert lo < min(bar.base) and hi > max(np.array(bar.base) + np.array(bar.y))


def test_band_chart_without_form_has_zero_height_bars_and_a_valid_range():
    a = ev.analyse(ev.Settings(n=4, m=0, l=24, s=500))
    fig = V.build_bands(a)
    assert max(fig.data[0].y) == 0.0
    assert fig.layout.yaxis.range[0] < 0 < fig.layout.yaxis.range[1]


def test_histogram_counts_every_assignment(analysis):
    fig = V.build_hist(analysis)
    _locked(fig)
    assert len(fig.data[0].x) == len(analysis.perms)


def test_sweep_charts_hold_the_rows_and_lock_axes():
    rows_m = [{"m": 0, "exakt": 0.0, "gemeinsam": 0.0, "naiv": 0.4}, {"m": 50, "exakt": 0.4, "gemeinsam": 0.38, "naiv": 0.6}]
    fig = V.build_m_sweep(rows_m)
    _locked(fig)
    assert [list(t.y) for t in fig.data] == [[0.0, 0.4], [0.0, 0.38], [0.4, 0.6]]
    rows_r = [{"start": s, "rho": r, "top": 0.5 - r / 4, "max": 0.9, "var": 0.65} for s in ("stationär", "Nullzustand") for r in (0.0, 0.5)]
    fig = V.build_rho_sweep(rows_r)
    _locked(fig)
    assert len(fig.data) == 4
