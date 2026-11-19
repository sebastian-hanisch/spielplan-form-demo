"""Jede Zahl der README steht hier als Rechnung oder Band (DEMO-PLAYBOOK §5, §11a).

Messungen: Mannschaften n = 8 (falls nicht anders genannt), Elo-Abstand 300, Remisanteil höchstens 0,25, einfache Runde, 100 zufällige Zuordnungen
(Zufalls-Seed 2026, SplitMix64), Erwartung exakt (Tensor-Rechnung); Simulation mit 2000 Läufen je Spielplan. Bänder statt Gleichheit: die Zahlen hängen an der letzten Binärstelle
der BLAS-Summen und der Potenzfunktion (DEMO-PLAYBOOK §5)."""

import itertools
import os
import re
from pathlib import Path

import numpy as np
import pytest

import spf_algorithm as A
import spf_constants as C
import spf_evaluation as ev
import spf_scenario as S

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def _exact(n=8, m=50, rho=0.5, span=300, draw=0.25, double=False, stationary=True, count=100, seed=2026):
    liga = S.Liga(n=n, span=float(span), draw=draw, double=double)
    return ev.exact_plans(liga, A.Form(m=float(m), rho=rho, stationary=stationary), S.assignments(n, count, seed))


@pytest.fixture(scope="module")
def default():
    return ev.analyse(ev.Settings())


@pytest.fixture(scope="module")
def sweep_m_rows():
    return ev.sweep_m(ev.Settings())


@pytest.fixture(scope="module")
def sweep_rho_rows():
    return ev.sweep_rho(ev.Settings())


# --- Der Haken (m = 0) --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n, double", [(4, False), (6, False), (8, False), (8, True), (4, True)])
def test_without_form_every_team_has_zero_spread_exactly(n, double):
    pts, _ = _exact(n=n, m=0, double=double, count=60)
    assert ev.team_spans(pts).max() == 0.0


def test_without_form_simulation_with_common_numbers_has_zero_spread_for_ten_teams_too():
    a = ev.analyse(ev.Settings(n=10, m=0, l=40, s=500))
    assert ev.team_spans(a.sim_points).max() == 0.0 and a.exact_points is None
    b = ev.analyse(ev.Settings(n=8, m=0))
    assert ev.team_spans(b.sim_points).max() == 0.0 and ev.team_spans(b.exact_points).max() == 0.0


def test_naive_independent_numbers_show_spread_without_form(sweep_m_rows):
    row = sweep_m_rows[0]
    assert row["m"] == 0 and row["exakt"] == 0.0 and row["gemeinsam"] == 0.0
    assert 0.35 <= row["naiv"] <= 0.60                          # gemessen 0,47 (Median über 3 Messreihen, je 40 Zuordnungen, 1000 Läufe)


# --- Größe der Spanne ---------------------------------------------------------------------------------------------------------------------

def test_default_spans(default):
    assert default.exact_points is not None and len(default.perms) == 100
    assert default.top_span == pytest.approx(0.43, abs=0.03)             # gemessen 0,432
    assert default.max_span == pytest.approx(0.80, abs=0.05)             # gemessen 0,795
    spans = ev.team_spans(default.main_points)
    assert int(spans.argmax()) == 3                                      # in der Stichprobe von 100 Zuordnungen hat M4 die größte Spanne, nicht der Stärkste (über alle 40.320: M4 und M5)
    assert spans[3] > spans[0] and spans[3] > spans[7]
    assert spans[0] == pytest.approx(0.43, abs=0.03) and spans[7] == pytest.approx(0.42, abs=0.03)
    assert default.settings.max_points == 21
    assert 100 * default.top_span / 21 == pytest.approx(2.06, abs=0.15)  # 2,1 % der möglichen Punkte
    assert default.main_points[:, 0].mean() == pytest.approx(14.63, abs=0.1)
    assert default.top_sd == pytest.approx(3.24, abs=0.15)
    assert 100 * default.top_span / default.top_sd == pytest.approx(13.3, abs=1.5)


def test_simulation_against_exact_at_default(default):
    dev = np.abs(default.sim_points - default.exact_points)
    assert dev.mean() == pytest.approx(0.047, abs=0.02) and dev.max() < 0.25      # Mittel 0,047, Maximum 0,168
    assert default.sim_top_span == pytest.approx(0.38, abs=0.08)                   # Simulation 0,38 gegen exakt 0,43: enthält Rauschen


@pytest.mark.parametrize("rho, expected", [(0.0, (0.363, 0.755, 1.693)), (0.5, (0.197, 0.432, 0.973)), (0.8, (0.091, 0.193, 0.402)), (0.95, (0.025, 0.052, 0.106))])
def test_span_of_the_strongest_by_form_effect_and_memory(rho, expected):
    for m, value in zip((25, 50, 100), expected):
        pts, _ = _exact(m=m, rho=rho)
        assert pts[:, 0].max() - pts[:, 0].min() == pytest.approx(value, rel=0.04, abs=0.004), (m, rho)


def test_elo_range_zero_and_six_hundred():
    pts0, _ = _exact(span=0)
    top0 = ev.team_spans(pts0).max()
    pts600, _ = _exact(span=600)
    assert top0 == 0.0                                                # gleich starke Mannschaften, Standardfall: Spanne 0 (Gleitkomma-Rest unter 1e-9 zählt als 0)
    assert float(np.max(pts0.max(axis=0) - pts0.min(axis=0))) < 1e-9   # der Rest selbst (gemessen rund 1e-14)
    assert ev.team_spans(pts600).max() == pytest.approx(1.04, abs=0.08)   # Abstand 600: größte Spanne 1,04, die des Stärksten 0,39
    assert ev.span_of(pts600[:, 0]) == pytest.approx(0.39, abs=0.04)


def test_double_round_and_league_size():
    pts, _ = _exact(double=True)
    assert ev.span_of(pts[:, 0]) == pytest.approx(0.83, abs=0.05)     # Doppelrunde (8 Mannschaften): 0,83 von höchstens 42 Punkten, 2,0 %
    assert 100 * 0.83 / 42 == pytest.approx(2.0, abs=0.1)
    pts4, _ = _exact(n=4, m=100, count=24)
    assert ev.team_spans(pts4).max() == pytest.approx(0.415, abs=0.03) and ev.span_of(pts4[:, 0]) == pytest.approx(0.143, abs=0.02)
    pts6, _ = _exact(n=6, m=50)
    assert ev.span_of(pts6[:, 0]) == pytest.approx(0.282, abs=0.03)
    a10 = ev.analyse(ev.Settings(n=10))
    assert a10.top_span == pytest.approx(0.73, abs=0.15)             # n = 10: nur Simulation (Rauschen inbegriffen)


def test_equal_strengths_keep_a_tiny_span_only_where_the_draw_cap_binds():
    """README (Befunde und Korrekturen): Elo-Abstand 0, n = 6, alle 720 Zuordnungen, ρ = 0,5: Remisanteil höchstens 0,5 gibt 0,0002 bei m = 100 und 0,003 bei m = 150;
    bei Remisanteil höchstens 0,3 und m = 150 ist die Spanne 0 (Gleitkomma-Rest). Die Spannen sind hier roh (ohne Nullsetzen), sonst bliebe der Rest unsichtbar."""
    def raw(draw, m):
        pts, _ = _exact(n=6, span=0, draw=draw, m=m, count=720)
        return float((pts.max(axis=0) - pts.min(axis=0)).max())
    assert raw(0.5, 100) == pytest.approx(1.7e-4, abs=0.6e-4)
    assert raw(0.5, 150) == pytest.approx(3.2e-3, abs=0.6e-3)
    assert raw(0.3, 150) < 1e-9


def test_more_assignments_contain_the_first_100_and_do_not_settle_the_span():
    assert S.assignments(8, 300, 2026)[:100] == S.assignments(8, 100, 2026)       # die 300 Zuordnungen enthalten dieselben ersten 100
    pts, _ = _exact(count=300)
    assert ev.team_spans(pts).max() == pytest.approx(0.82, abs=0.05)        # 300 statt 100 Zuordnungen: größte Spanne 0,82 (statt 0,80)
    assert ev.span_of(pts[:, 0]) == pytest.approx(0.43, abs=0.03)           # die des Stärksten bleibt bei 0,43 (Zufall: über alle 40.320 sind es 0,59)


@pytest.mark.skipif(os.environ.get("SPF_FULL") != "1", reason="vollständige Aufzählung aller 40.320 Zuordnungen, rund vier Minuten: nur mit SPF_FULL=1")
def test_full_enumeration_of_all_assignments_default_case():
    """README: über alle 40.320 Zuordnungen (Standardfall, exakt) Spanne des Stärksten 0,59 (2,8 % von 21), M2 0,74, M3 und M6 0,84, M4 und M5 0,89, M7 0,74, M8 0,58;
    Spanne des Stärksten aus 100 Zuordnungen über die Zufalls-Seeds 2026 bis 2055: 0,38 bis 0,57, Mittel 0,47 (300 Zuordnungen: Mittel 0,51);
    die größte Spanne hatte M4 in 18, M5 in 6, M3 und M6 in je 3 von 30 Fällen."""
    liga = S.Liga(n=8, span=300.0, draw=0.25, double=False)
    perms = list(itertools.permutations(range(8)))
    pts, _ = ev.exact_plans(liga, A.Form(m=50.0, rho=0.5, stationary=True), perms)
    spans = ev.team_spans(pts)
    assert spans == pytest.approx([0.588, 0.738, 0.844, 0.894, 0.894, 0.844, 0.737, 0.578], abs=0.002)
    assert abs(spans[3] - spans[4]) < 1e-9 and 100 * spans[0] / 21 == pytest.approx(2.8, abs=0.05)
    index = {p: i for i, p in enumerate(perms)}

    def over_seeds(count):
        tops, winners = [], []
        for seed in range(2026, 2056):
            sub = pts[[index[p] for p in S.assignments(8, count, seed)]]
            s = ev.team_spans(sub)
            tops.append(s[0])
            winners.append(int(s.argmax()) + 1)
        return np.array(tops), winners
    tops100, winners100 = over_seeds(100)
    assert tops100.min() == pytest.approx(0.383, abs=0.003) and tops100.max() == pytest.approx(0.565, abs=0.003) and tops100.mean() == pytest.approx(0.470, abs=0.003)
    assert {t: winners100.count(t) for t in set(winners100)} == {4: 18, 5: 6, 3: 3, 6: 3}
    assert over_seeds(300)[0].mean() == pytest.approx(0.508, abs=0.003)


# --- Zustandsstreuung konstant (die Bauauflage) -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("m, limit, lo, hi", [(50, 0.005, 0.649, 0.653), (100, 0.010, 0.644, 0.652)])
def test_state_variance_is_nearly_constant_in_rho_with_stationary_start(m, limit, lo, hi):
    """100 Zuordnungen, n = 8: Abstand der Zustandsstreuung über ρ = 0, 0,5, 0,8, 0,95 gemessen 0,0031 (m = 50) und 0,0070 (m = 100)."""
    vs = [_exact(m=m, rho=rho)[1] for rho in (0.0, 0.5, 0.8, 0.95)]
    assert max(vs) - min(vs) < limit
    assert all(lo <= v <= hi for v in vs)


def test_state_variance_is_not_constant_with_start_in_zero():
    vs = [_exact(m=50, rho=rho, stationary=False)[1] for rho in (0.0, 0.5, 0.8, 0.95)]
    assert vs[0] == pytest.approx(0.574, abs=0.01) and vs[1] == pytest.approx(0.491, abs=0.01)
    assert vs[2] == pytest.approx(0.310, abs=0.01) and vs[3] == pytest.approx(0.102, abs=0.01)
    assert max(vs) - min(vs) > 0.45                                   # Abstand 0,47


def test_without_form_the_state_variance_does_not_depend_on_rho():
    """m = 0, n = 4, alle 24 Zuordnungen: die Zustandsstreuung (in einem zufällig gezogenen Spielplan) ist für jedes ρ gleich, exakt (der Orakel-Test zeigt es für die Verteilung selbst);
    n = 8 mit 100 von 40 320 Zuordnungen: Abstand höchstens 0,001 (gemessen 0,0006)."""
    v4 = [_exact(n=4, m=0, rho=rho, count=24)[1] for rho in (0.0, 0.5, 0.8, 0.95)]
    assert max(v4) - min(v4) < 1e-9 and v4[0] == pytest.approx(0.6104, abs=0.001)
    v8 = [_exact(m=0, rho=rho)[1] for rho in (0.0, 0.5, 0.8, 0.95)]
    assert max(v8) - min(v8) < 0.001 and v8[0] == pytest.approx(0.6665, abs=0.002)


def test_spread_still_falls_with_memory_even_with_constant_state_spread(sweep_rho_rows):
    stat = [r for r in sweep_rho_rows if r["start"] == "stationär"]
    zero = [r for r in sweep_rho_rows if r["start"] == "Nullzustand"]
    assert [round(r["top"], 2) for r in stat] == pytest.approx([0.72, 0.41, 0.19, 0.05], abs=0.03)
    assert [round(r["top"], 2) for r in zero] == pytest.approx([0.72, 0.37, 0.14, 0.03], abs=0.03)
    assert max(r["var"] for r in stat) - min(r["var"] for r in stat) < 0.005
    assert zero[-1]["var"] == pytest.approx(0.102, abs=0.01)
    assert all(z["top"] <= s["top"] + 0.02 for z, s in zip(zero[1:], stat[1:]))     # naiver Start unterschätzt die Spanne bei ρ >= 0,5


def test_m_sweep_values(sweep_m_rows):
    assert [round(r["exakt"], 2) for r in sweep_m_rows] == pytest.approx([0.0, 0.19, 0.41, 0.93], abs=0.04)
    assert [r["m"] for r in sweep_m_rows] == list(C.M_SWEEP)


# --- README: genannte Testzahl und Platzhalter ------------------------------------------------------------------------------------------------

def test_readme_has_no_placeholders_and_no_planning_words():
    assert "TODO" not in README
    for word in ("Saatwert", "Schwanz", "geplant", "kommt noch"):
        assert word not in README, word


def test_readme_test_count_matches_the_suite():
    m = re.search(r"\*\*(\d+) Tests\*\*", README)
    assert m, "die README nennt die Testzahl als '**N Tests**'"
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests", "-p", "no:cacheprovider"], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parent.parent)).stdout
    counted = int(re.search(r"(\d+) tests? collected", out).group(1))
    assert int(m.group(1)) == counted
