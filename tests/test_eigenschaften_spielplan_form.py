"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, Simulationen und Rangfolgen): Aussagen, die gelten müssen, ohne dass man das richtige Ergebnis kennt.

Jede Eigenschaft ist eine belegte Aussage über das Modell (Beweis oder Symmetrie im Kommentar), nicht nur das heutige Verhalten des Codes."""

import numpy as np
import pytest

import spf_algorithm as A
import spf_evaluation as ev
import spf_scenario as S
from spf_rng import SplitMix64


def _random_leagues(count, seed):
    """Zufällige Ligen (feste Zufalls-Seeds): n, Elo-Abstand, Remisanteil, Gedächtnis, Rundenart."""
    rng = SplitMix64(seed)
    out = []
    for _ in range(count):
        out.append(dict(n=(4, 6)[rng.below(2)], span=25 * rng.below(25), draw=0.05 * rng.below(11), rho=0.05 * rng.below(20), double=bool(rng.below(2))))
    return out


def _exact_spans(n, span, draw, rho, m, double, perms=8, stationary=True):
    liga = S.Liga(n=n, span=float(span), draw=draw, double=double)
    pts, _ = ev.exact_plans(liga, A.Form(m=float(m), rho=rho, stationary=stationary), S.assignments(n, perms, 5))
    return ev.team_spans(pts)


def test_without_form_the_spread_is_zero_in_every_random_league():
    """m = 0: die Partie hängt nur von den Grundstärken ab, die Summe über die (feste) Gegnermenge ist von der Reihenfolge unabhängig (siehe 📐 'Der Haken')."""
    for lg in _random_leagues(14, 101):
        assert _exact_spans(m=0, **lg).max() < 1e-9, lg


def test_without_form_common_random_numbers_give_zero_spread_in_every_random_league():
    for lg in _random_leagues(8, 202):
        liga = S.Liga(n=lg["n"], span=float(lg["span"]), draw=lg["draw"], double=lg["double"])
        pts, _, _ = ev.sim_plans(liga, A.Form(m=0.0, rho=lg["rho"]), S.assignments(lg["n"], 6, 3), 200, 7)
        assert ev.team_spans(pts).max() == 0.0, lg


def test_equal_strengths_have_zero_span_with_form_unless_the_draw_cap_binds():
    """Gleich starke Mannschaften sind austauschbar: bei Remisanteil höchstens 0,25 ist die Spanne auch mit Form 0 (Gleitkomma-Rest unter 1e-9, gemessen rund 1e-14).
    Nur bei Remisanteil höchstens 0,5 und m = 100 (Elo-Abstand zweier Zustandsstufen 200, die Obergrenze des Remisanteils bindet) bleibt eine winzige Spanne
    (gemessen 0,0007, 8 Zuordnungen; die README nennt die Werte für alle 720 Zuordnungen, siehe test_claims)."""
    for draw in (0.0, 0.25):
        for rho in (0.0, 0.6):
            assert _exact_spans(n=6, span=0, draw=draw, rho=rho, m=100, double=False).max() == 0.0, (draw, rho)
    worst = max(_exact_spans(n=6, span=0, draw=0.5, rho=rho, m=100, double=False).max() for rho in (0.0, 0.6))
    assert 1e-5 < worst < 0.001                                  # gemessen 0,0007


def test_every_assignment_is_a_complete_round_robin():
    for n in (8, 10):
        for perm in S.assignments(n, 40, 9):
            for double in (False, True):
                sched = S.schedule(perm, double)
                legs = 2 if double else 1
                assert len(sched) == legs * (n - 1)
                pairs = [g for r in sched for g in r]
                assert sorted(set(pairs)) == sorted(itertools_pairs(n)) and len(pairs) == legs * n * (n - 1) // 2
                for r in sched:
                    assert sorted(x for g in r for x in g) == list(range(n))


def itertools_pairs(n):
    return [(a, b) for a in range(n) for b in range(a + 1, n)]


def test_shifting_all_ratings_changes_nothing():
    """Elo-Erwartung und Remisanteil hängen nur von Differenzen ab."""
    R = S.ratings(6, 300)
    form = A.Form(m=60.0, rho=0.4)
    a = A.exact_points(S.schedule((1, 0, 3, 2, 5, 4)), R, form, 0.25)
    b = A.exact_points(S.schedule((1, 0, 3, 2, 5, 4)), R + 777.0, form, 0.25)
    assert a.points == pytest.approx(b.points, abs=1e-9)
    assert A.stationary_distribution(R, form, 0.25) == pytest.approx(A.stationary_distribution(R + 777.0, form, 0.25), abs=1e-12)


def test_simulation_of_a_plan_does_not_depend_on_the_order_of_the_plans():
    R = S.ratings(6, 300)
    form = A.Form(m=50.0, rho=0.5)
    rnd = A.draw_randomness(6, 300, 2, 5)
    pi0 = A.start_distribution(R, form, 0.25)
    perms = S.assignments(6, 5, 4)
    forward = [A.simulate(S.schedule(p), R, form, 0.25, rnd, pi0).points for p in perms]
    backward = [A.simulate(S.schedule(p), R, form, 0.25, rnd, pi0).points for p in reversed(perms)][::-1]
    assert all(np.array_equal(f, b) for f, b in zip(forward, backward))


def test_spread_grows_with_the_effect_of_form():
    """Gemessen für ρ = 0, 0.5, 0.8 und je 40 Zuordnungen bei n = 8 (steht in README und test_claims): strikt wachsend in m."""
    for rho in (0.0, 0.5, 0.8):
        spans = [_exact_spans(n=8, span=300, draw=0.25, rho=rho, m=m, double=False, perms=40)[0] for m in (0, 25, 50, 100)]
        assert spans[0] == 0 and spans[0] < spans[1] < spans[2] < spans[3], (rho, spans)


def test_expected_points_are_monotone_in_strength_for_every_plan_without_form():
    R = S.ratings(8, 300)
    for perm in S.assignments(8, 6, 1):
        p = A.exact_points(S.schedule(perm), R, A.Form(m=0.0, rho=0.5), 0.25).points
        assert (np.diff(p) < 0).all()
