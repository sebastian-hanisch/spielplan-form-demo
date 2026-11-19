"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import spf_algorithm as A
import spf_constants as C
import spf_evaluation as ev
import spf_scenario as S


def test_exactly_two_teams():
    """n = 2: der kleinste Fall; ein Spieltag, beide Zuordnungen geben dieselbe Partie, die Spanne ist 0 auch mit Form."""
    liga = S.Liga(n=2, span=300.0, draw=0.25)
    pts, _ = ev.exact_plans(liga, A.Form(m=100.0, rho=0.5), S.assignments(2, 5, 1))
    assert ev.team_spans(pts).max() == 0.0
    assert pts[0, 0] > pts[0, 1]


def test_odd_number_of_teams_is_rejected():
    for n in (3, 5, 9):
        with pytest.raises(ValueError):
            S.schedule(tuple(range(n)))


def test_form_at_the_slider_limits():
    for m, rho in ((C.M_MIN, C.RHO_MIN), (C.M_MAX, C.RHO_MAX), (C.M_MAX, C.RHO_MIN), (C.M_MIN, C.RHO_MAX)):
        a = ev.analyse(ev.Settings(n=4, m=m, rho=rho, l=20, s=500))
        assert np.isfinite(a.main_points).all() and a.top_span >= 0


def test_rho_must_stay_below_one():
    assert C.RHO_MAX < 1.0
    with pytest.raises(ValueError):
        A.Form(m=C.DEFAULT_M, rho=1.0)


def test_draw_share_at_the_limits():
    for draw in (C.DRAW_MIN, C.DRAW_MAX):
        a = ev.analyse(ev.Settings(n=4, draw=draw, l=20, s=500))
        assert np.isfinite(a.main_points).all()
    pw, d, pl = S.outcome_probs(0.0, 0.0, 0.0)
    assert (float(pw), float(d), float(pl)) == (0.5, 0.0, 0.5)          # kein Remis: nur Sieg und Niederlage


def test_equal_ratings_at_span_zero():
    a = ev.analyse(ev.Settings(n=4, span=0, l=24, s=500))
    assert a.main_points.min() > 0 and a.max_span < 0.005


def test_n_ten_has_no_exact_branch_and_still_gives_a_result():
    a = ev.analyse(ev.Settings(n=10, l=20, s=500))
    assert a.exact_points is None and a.exact_state_var is None
    assert a.main_points is a.sim_points and a.sim_points.shape == (20, 10)


def test_double_round_at_n4_has_six_rounds_and_twice_the_points():
    a1 = ev.analyse(ev.Settings(n=4, m=0, l=24, s=500))
    a2 = ev.analyse(ev.Settings(n=4, m=0, rounds=C.ROUNDS_OPTIONS[1], l=24, s=500))
    assert a2.exact_points[0] == pytest.approx(2 * a1.exact_points[0], abs=1e-9)       # ohne Form ist jede Partie unabhängig von der anderen
    assert a2.settings.max_points == 2 * a1.settings.max_points


def test_minimum_and_maximum_number_of_assignments():
    assert len(ev.analyse(ev.Settings(n=8, l=C.L_MIN, s=C.S_MIN)).perms) == C.L_MIN
    assert len(ev.analyse(ev.Settings(n=4, l=C.L_MIN, s=C.S_MIN)).perms) == C.L_MIN        # weniger als 24 verlangt: nur eine Stichprobe
    assert len(ev.analyse(ev.Settings(n=4, l=C.L_MAX, s=C.S_MIN)).perms) == 24


def test_span_of_ignores_rounding_dust():
    assert ev.span_of([5.0, 5.0 + 1e-13, 5.0 - 1e-13]) == 0.0
    assert ev.span_of([1.0, 2.5]) == 1.5
