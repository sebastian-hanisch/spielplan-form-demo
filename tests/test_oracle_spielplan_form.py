"""Orakel-Tests (ORACLE-PLAYBOOK §2, Verfahrenstyp Simulation / Spieltheorie-nah): ein unabhängiger zweiter Rechenweg für die Kernroutinen.

Der Prüfling rechnet mit NumPy-Tensoren (`exact_points`: gemeinsame Zustandsverteilung als Tensor mit n Achsen) und mit einer vektorisierten Simulation.
Die Orakel hier sind reines Python ohne NumPy und ohne einen Aufruf des Prüflings:
* `enumerate_expected_points`: Aufzählung aller Ergebnisbäume (Sieg/Remis/Niederlage je Partie, dazu die Münzwürfe bleibt/wechselt je Mannschaft) mit dem Zustandsverlauf in einem
  Dictionary; für n = 4 sind das 3^6 = 729 Ergebnisfolgen, jeweils mit Münzwurf-Verzweigung.
* `enumerate_simple_model`: das einfache ±m-Modell (rho = 0, Start in 0) als reiner Baum über die 3^6 Ergebnisfolgen, ohne Zustandsverteilung.
* `power_iteration_stationary`: die stationäre Verteilung als Grenzwert der Matrixpotenz P^k (anderer Weg als der lineare Löser des Prüflings).
Vorher von Hand geprüft: n = 2, m = 0 gibt 1,375 Punkte je Mannschaft (siehe test_algorithm.py)."""

import itertools

import numpy as np
import pytest

import spf_algorithm as A
import spf_scenario as S

N4 = 4
R4 = [150.0, 50.0, -50.0, -150.0]
DRAW = 0.25


def probs(ra, rb, draw=DRAW):
    e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    d = min(draw, 2.0 * min(e, 1.0 - e))
    return e - d / 2.0, d, 1.0 - e - d / 2.0


def power_iteration_stationary(R, m, draw=DRAW, rho=0.5, rounds=256):
    """Fixpunkt: pi_i = Grenzwert der Zeilen von (rho·I + (1-rho)·Q_i)^k, Q_i aus den Verteilungen der anderen (mittleres Feld)."""
    n = len(R)
    pi = [[1 / 3.0] * 3 for _ in range(n)]
    for _ in range(60):
        new = []
        for i in range(n):
            Q = [[0.0] * 3 for _ in range(3)]                        # Zeilen: Zustand -1, 0, +1; Spalten: Ergebnis Niederlage, Remis, Sieg
            for j in range(n):
                if j == i:
                    continue
                for si, sv in enumerate((-1, 0, 1)):
                    for sj, sw in enumerate((-1, 0, 1)):
                        pw, d, pl = probs(R[i] + m * sv, R[j] + m * sw, draw)
                        w = pi[j][sj] / (n - 1)
                        Q[si][0] += w * pl
                        Q[si][1] += w * d
                        Q[si][2] += w * pw
            P = np.array([[rho * (1.0 if r == c else 0.0) + (1 - rho) * Q[r][c] for c in range(3)] for r in range(3)])
            row = np.linalg.matrix_power(P, rounds)[0]                   # jede Zeile konvergiert gegen pi
            new.append((row / row.sum()).tolist())                       # Rundungsreste der Potenz nicht in die nächste Runde tragen
        pi = new
    return pi


def enumerate_expected_points(sched, R, m, rho, draw, pi0):
    """Erwartete Punkte und Zustandsverteilung je Spieltag durch Aufzählung: Dictionary {Zustandsfolge aller Mannschaften: Wahrscheinlichkeit}."""
    n = len(R)
    dist = {}
    for combo in itertools.product((-1, 0, 1), repeat=n):
        w = 1.0
        for i, s in enumerate(combo):
            w *= pi0[i][s + 1]
        dist[combo] = w
    points = [0.0] * n
    marginals = []
    for games in sched:
        marginals.append([[sum(w for st, w in dist.items() if st[i] == s) for s in (-1, 0, 1)] for i in range(n)])
        for a, b in games:
            nxt = {}
            for st, w in dist.items():
                pw, d, pl = probs(R[a] + m * st[a], R[b] + m * st[b])
                for p, ra_res, rb_res, pa_pts, pb_pts in ((pw, 1, -1, 3, 0), (d, 0, 0, 1, 1), (pl, -1, 1, 0, 3)):
                    if p == 0.0:
                        continue
                    points[a] += w * p * pa_pts
                    points[b] += w * p * pb_pts
                    for sa, qa in ((st[a], rho), (ra_res, 1 - rho)):
                        for sb, qb in ((st[b], rho), (rb_res, 1 - rho)):
                            new = list(st)
                            new[a], new[b] = sa, sb
                            key = tuple(new)
                            nxt[key] = nxt.get(key, 0.0) + w * p * qa * qb
            dist = nxt
    return points, marginals


def enumerate_simple_model(sched, R, m, draw=DRAW):
    """Das einfache ±m-Modell als Ergebnisbaum: Start im Zustand 0, nach jeder Partie ist der Zustand das Ergebnis (+1 / 0 / -1); 3^6 Blätter bei n = 4."""
    games = [g for rnd in sched for g in rnd]
    n = len(R)
    points = [0.0] * n

    def walk(k, state, w):
        if k == len(games):
            return
        a, b = games[k]
        pw, d, pl = probs(R[a] + m * state[a], R[b] + m * state[b], draw)
        for p, ra, rb, pa_pts, pb_pts in ((pw, 1, -1, 3, 0), (d, 0, 0, 1, 1), (pl, -1, 1, 0, 3)):
            if p == 0.0:
                continue
            points[a] += w * p * pa_pts
            points[b] += w * p * pb_pts
            nxt = list(state)
            nxt[a], nxt[b] = ra, rb
            walk(k + 1, nxt, w * p)

    walk(0, [0] * n, 1.0)
    return points


def _stationary_start(m):
    return power_iteration_stationary(R4, m)


# --- Aufzählung gegen die Tensor-Rechnung ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("perm", [(0, 1, 2, 3), (2, 0, 3, 1), (3, 1, 0, 2)])
@pytest.mark.parametrize("m, rho", [(0.0, 0.5), (50.0, 0.0), (100.0, 0.5), (100.0, 0.9)])
def test_tensor_expectation_equals_enumeration_for_four_teams(perm, m, rho):
    sched = S.schedule(perm)
    pi0 = _stationary_start(m)
    ref_points, ref_marg = enumerate_expected_points(sched, R4, m, rho, DRAW, pi0)
    ex = A.exact_points(sched, np.array(R4), A.Form(m=m, rho=rho), DRAW)
    assert ex.points == pytest.approx(ref_points, abs=1e-9)
    assert ex.state_marginals == pytest.approx(np.array(ref_marg), abs=1e-9)


def test_tensor_expectation_equals_enumeration_in_a_double_round():
    sched = S.schedule((1, 3, 0, 2), double=True)
    pi0 = _stationary_start(80.0)
    ref_points, _ = enumerate_expected_points(sched, R4, 80.0, 0.6, DRAW, pi0)
    ex = A.exact_points(sched, np.array(R4), A.Form(m=80.0, rho=0.6), DRAW)
    assert ex.points == pytest.approx(ref_points, abs=1e-9)


def test_naive_start_in_zero_equals_enumeration():
    pi0 = [[0.0, 1.0, 0.0]] * N4
    sched = S.schedule((0, 2, 1, 3))
    ref_points, _ = enumerate_expected_points(sched, R4, 70.0, 0.5, DRAW, pi0)
    ex = A.exact_points(sched, np.array(R4), A.Form(m=70.0, rho=0.5, stationary=False), DRAW)
    assert ex.points == pytest.approx(ref_points, abs=1e-9)


@pytest.mark.parametrize("perm", list(itertools.islice(itertools.permutations(range(4)), 0, 24, 5)))
def test_simple_pm_m_model_is_the_special_case_rho_zero_start_zero(perm):
    """Der Sonderfall: rho = 0 mit Start in 0 ist das einfache ±m-Modell; Baum über 3^6 Ergebnisfolgen gegen die Tensor-Rechnung und gegen die Simulation."""
    sched = S.schedule(perm)
    tree = enumerate_simple_model(sched, R4, 100.0)
    ex = A.exact_points(sched, np.array(R4), A.Form(m=100.0, rho=0.0, stationary=False), DRAW)
    assert ex.points == pytest.approx(tree, abs=1e-9)
    sim = A.simulate(sched, np.array(R4), A.Form(m=100.0, rho=0.0, stationary=False), DRAW, A.draw_randomness(4, 60000, 17, 3))
    assert sim.points == pytest.approx(tree, abs=0.06)


# --- Simulation gegen die Aufzählung --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("m, rho", [(0.0, 0.5), (100.0, 0.5), (100.0, 0.9)])
def test_simulation_matches_the_enumeration(m, rho):
    sched = S.schedule((2, 0, 3, 1))
    pi0 = _stationary_start(m)
    ref_points, _ = enumerate_expected_points(sched, R4, m, rho, DRAW, pi0)
    sim = A.simulate(sched, np.array(R4), A.Form(m=m, rho=rho), DRAW, A.draw_randomness(4, 80000, 23, 3))
    # Standardfehler eines Mittels über 80 000 Läufe: Standardabweichung der Saisonpunkte (< 2,5) / 283 < 0,009; Band: 6 Standardfehler
    assert sim.points == pytest.approx(ref_points, abs=0.055)


def test_simulated_spread_over_all_24_assignments_tracks_the_exact_spread():
    """Mit gemeinsamen Zufallszahlen schwankt die simulierte Punktzahl über die Zuordnungen fast nur durch die Reihenfolge: Spanne ≈ exakte Spanne."""
    R = np.array(R4)
    perms = S.assignments(4, 24, 1)
    form = A.Form(m=100.0, rho=0.5)
    pi0 = A.start_distribution(R, form, DRAW)
    rnd = A.draw_randomness(4, 20000, 31, 3)
    sim = np.array([A.simulate(S.schedule(p), R, form, DRAW, rnd, pi0).points for p in perms])
    ref = np.array([enumerate_expected_points(S.schedule(p), R4, 100.0, 0.5, DRAW, _stationary_start(100.0))[0] for p in perms])
    assert np.ptp(ref, axis=0).max() > 0.2                                    # es gibt etwas zu messen
    assert np.ptp(sim, axis=0) == pytest.approx(np.ptp(ref, axis=0), abs=0.12)


# --- Stationäre Verteilung --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("m", [0.0, 50.0, 100.0])
def test_stationary_distribution_equals_the_matrix_power_limit(m):
    ref = np.array(power_iteration_stationary(R4, m))
    mine = A.stationary_distribution(np.array(R4), A.Form(m=m, rho=0.0), DRAW)
    assert mine == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_without_form_the_average_state_distribution_over_all_assignments_is_stationary_in_every_round(rho):
    """Die Auflage: bei m = 0 ist die Zustandsverteilung jeder Mannschaft, gemittelt über alle 24 Zuordnungen, in JEDEM Spieltag gleich der Startverteilung (exakt)."""
    pi0 = power_iteration_stationary(R4, 0.0, rho=rho)
    total = None
    for perm in itertools.permutations(range(4)):
        _, marg = enumerate_expected_points(S.schedule(perm), R4, 0.0, rho, DRAW, pi0)
        marg = np.array(marg)
        total = marg if total is None else total + marg
    total /= 24
    for t in range(3):
        assert total[t] == pytest.approx(np.array(pi0), abs=1e-12)
