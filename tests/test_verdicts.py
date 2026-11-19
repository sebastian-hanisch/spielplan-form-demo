"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text); die Tests bauen die Analyse aus Hand-Werten statt aus Läufen."""

import itertools

import numpy as np

import spf_evaluation as ev


def _analysis(m, points, sd=2.0, rho=0.5, exact=True):
    settings = ev.Settings(n=4, m=m, rho=rho, l=2)
    pts = np.array(points, dtype=float)
    return ev.Analysis(settings, np.zeros(4), [(0, 1, 2, 3), (1, 0, 2, 3)], pts, pts if exact else None, 0.65 if exact else None, 0.65, np.array([sd, 1, 1, 1]), np.zeros((4, 3)))


def test_core_verdict_without_form_is_success_and_names_zero():
    kind, text = ev.verdict_core(_analysis(0, [[10, 8, 6, 4], [10, 8, 6, 4]]))
    assert kind == "success"
    assert "m = 0" in text and "Spanne des Stärksten 0.00 Punkte" in text and "exakt, 2 Zuordnungen" in text and "Rauschen" in text


def test_core_verdict_with_form_names_best_worst_span_share_and_noise():
    kind, text = ev.verdict_core(_analysis(50, [[10.0, 8, 6, 4], [10.5, 8, 6, 4]]))      # Spanne 0,5 von höchstens 9 Punkten: 5,6 %; sd 2,0: 25 %
    assert kind == "warning"
    assert "im besten der 2 Zuordnungen 10.50, im schlechtesten 10.00 Punkte" in text
    assert "Spanne **0.50** Punkte, das sind 5.6 % der höchstens 9" in text
    assert "um 2.00 (Standardabweichung" in text and "25 % davon" in text
    assert "m = 50 Elo" in text and "ρ = 0.50" in text
    assert "Die Spanne gilt für diese 2 von 24 Zuordnungen; über alle ist sie mindestens so groß." in text         # Stichprobe von 2 der 4! = 24 Zuordnungen


def test_core_verdict_has_no_sample_note_when_all_assignments_are_used():
    perms = list(itertools.permutations(range(4)))
    pts = np.array([[10.0 + 0.01 * k, 8, 6, 4] for k in range(len(perms))])
    a = ev.Analysis(ev.Settings(n=4, m=50, l=24), np.zeros(4), perms, pts, pts, 0.65, 0.65, np.array([2.0, 1, 1, 1]), np.zeros((4, 3)))
    assert "Zuordnungen; über alle ist sie" not in ev.verdict_core(a)[1] and "der 24 Zuordnungen" in ev.verdict_core(a)[1]


def test_core_verdict_threshold_two_percent_of_the_maximum_points():
    assert ev.verdict_core(_analysis(50, [[10.0, 8, 6, 4], [10.17, 8, 6, 4]]))[0] == "info"         # 1,9 %
    assert ev.verdict_core(_analysis(50, [[10.0, 8, 6, 4], [10.19, 8, 6, 4]]))[0] == "warning"      # 2,1 %


def test_core_verdict_says_simulation_without_exact_points():
    _, text = ev.verdict_core(_analysis(50, [[10.0, 8, 6, 4], [10.5, 8, 6, 4]], exact=False))
    assert "(in der Simulation)" in text


def test_m_verdict_names_zero_naive_and_last_values():
    rows = [{"m": 0, "exakt": 0.0, "gemeinsam": 0.0, "naiv": 0.47}, {"m": 100, "exakt": 0.93, "gemeinsam": 0.88, "naiv": 1.02}]
    kind, text = ev.verdict_m(rows, True)
    assert kind == "info"
    assert "exakt **0.00** Punkte" in text and "(0.00)" in text and "**0.47** Punkte Spanne: reines Rauschen (die Zahl hängt an der Zahl der Läufe und der Zuordnungen)" in text and "Bei m = 100 sind es 0.93 Punkte" in text
    assert "in der Simulation mit gemeinsamen Zufallszahlen" in ev.verdict_m(rows, False)[1]
    rows[0]["exakt"] = 0.2
    assert ev.verdict_m(rows, True)[0] == "warning"        # wäre die Spanne bei m = 0 nicht null, wäre der Befund ein Fehler


def test_rho_verdict_names_variance_band_and_span_fall():
    rows = [{"start": "stationär", "rho": 0.0, "top": 0.75, "max": 1.2, "var": 0.650}, {"start": "stationär", "rho": 0.95, "top": 0.05, "max": 0.1, "var": 0.649},
            {"start": "Nullzustand", "rho": 0.0, "top": 0.80, "max": 1.3, "var": 0.572}, {"start": "Nullzustand", "rho": 0.95, "top": 0.04, "max": 0.06, "var": 0.102}]
    kind, text = ev.verdict_rho(rows)
    assert kind == "info"
    assert "zwischen 0.649 und 0.650 (Abstand 0.001)" in text and "zwischen 0.102 und 0.572 (Abstand 0.470)" in text
    assert "von 0.75 (ρ = 0.00) auf 0.05 (ρ = 0.95)" in text and "von 0.80 auf 0.04" in text


def test_rho_verdict_without_form_says_nothing_to_distribute():
    rows = [{"start": s, "rho": r, "top": 0.0, "max": 0.0, "var": 0.66} for s in ("stationär", "Nullzustand") for r in (0.0, 0.95)]
    assert "in allen Zeilen 0" in ev.verdict_rho(rows)[1]
