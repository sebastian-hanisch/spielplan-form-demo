"""Einheiten gegen Handrechnung auf Mini-Instanzen (DEMO-PLAYBOOK §1b): Zirkeltafel, Partie, Übergang, stationäre Verteilung, Zufallszahlen, Simulation, exakte Erwartung."""

import itertools

import numpy as np
import pytest

import spf_algorithm as A
import spf_scenario as S
from spf_rng import SplitMix64


def _e(diff):
    """Elo-Erwartung bei Stärkeunterschied diff = r_a - r_b."""
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


# --- Zirkeltafel und Zuordnung ---------------------------------------------------------------------------------------------------------

def test_circle_table_of_four_by_hand():
    """Plätze 0..3: Spieltag 1 (0,3),(1,2); Platz 0 bleibt, die übrigen rücken: Spieltag 2 (0,2),(3,1); Spieltag 3 (0,1),(2,3)."""
    assert S.circle_rounds(4) == [[(0, 3), (1, 2)], [(0, 2), (3, 1)], [(0, 1), (2, 3)]]


@pytest.mark.parametrize("n", [2, 4, 6, 8, 10])
def test_circle_table_is_a_complete_round_robin(n):
    rounds = S.circle_rounds(n)
    assert len(rounds) == n - 1
    pairs = [frozenset(g) for r in rounds for g in r]
    assert len(set(pairs)) == n * (n - 1) // 2 == len(pairs)            # jedes Paar genau einmal
    for r in rounds:
        assert sorted(x for g in r for x in g) == list(range(n))        # jede Mannschaft genau einmal je Spieltag


def test_circle_table_rejects_odd_and_tiny_n():
    for n in (0, 1, 3, 5):
        with pytest.raises(ValueError):
            S.circle_rounds(n)


def test_schedule_maps_seats_to_teams_with_smaller_number_first():
    perm = (3, 2, 1, 0)                                                 # Platz 0 -> Mannschaft 3, ...
    assert S.schedule(perm) == [[(0, 3), (1, 2)], [(1, 3), (0, 2)], [(2, 3), (0, 1)]]


def test_double_round_repeats_the_first_leg_in_the_same_order():
    single = S.schedule((0, 1, 2, 3))
    assert S.schedule((0, 1, 2, 3), double=True) == single + single


def test_assignments_all_permutations_for_n4_and_sample_otherwise():
    everything = S.assignments(4, 100, 1)
    assert len(everything) == 24 and set(everything) == set(itertools.permutations(range(4)))
    sample = S.assignments(8, 50, 1)
    assert len(sample) == 50 and all(sorted(p) == list(range(8)) for p in sample)
    assert sample == S.assignments(8, 50, 1)                            # gleicher Zufalls-Seed, gleiche Zuordnungen
    assert sample != S.assignments(8, 50, 2)
    assert len(S.assignments(4, 10, 1)) == 10                           # weniger als 24 verlangt: eine Stichprobe


def test_ratings_form_an_even_ladder():
    assert list(S.ratings(4, 300)) == pytest.approx([150.0, 50.0, -50.0, -150.0])
    assert list(S.ratings(3, 0)) == [0.0, 0.0, 0.0]


# --- Partie -------------------------------------------------------------------------------------------------------------------------------

def test_outcome_probabilities_equal_teams_by_hand():
    pw, d, pl = S.outcome_probs(0.0, 0.0, 0.25)
    assert (float(pw), float(d), float(pl)) == pytest.approx((0.375, 0.25, 0.375))


def test_outcome_probabilities_with_400_elo_difference_by_hand():
    """e = 10/11; der Remisanteil ist durch 2·(1 - e) = 2/11 begrenzt, nicht durch 0,25; Niederlage 1 - e - d/2 = 0."""
    pw, d, pl = (float(x) for x in S.outcome_probs(400.0, 0.0, 0.25))
    assert d == pytest.approx(2 / 11) and pw == pytest.approx(9 / 11) and pl == pytest.approx(0.0, abs=1e-12)


def test_expected_score_is_kept_whatever_the_draw_share():
    for dm in (0.0, 0.25, 0.5):
        pw, d, _ = S.outcome_probs(120.0, -30.0, dm)
        assert float(pw + d / 2) == pytest.approx(_e(150.0))


# --- Zustandsübergang ---------------------------------------------------------------------------------------------------------------------

def test_kernel_rows_sum_to_one_for_every_start_state():
    K = A.transition_kernel(80.0, -20.0, A.Form(m=40, rho=0.3), 0.25)
    assert K.shape == (3, 3, 3, 3)
    assert K.sum(axis=(2, 3)) == pytest.approx(np.ones((3, 3)))


def test_kernel_without_memory_sets_the_state_to_the_result_by_hand():
    """rho = 0, gleich starke Mannschaften im Zustand 0: Sieg von a -> (a heiß, b kalt) mit 0,375; Remis -> (0, 0) mit 0,25; Sieg von b -> (a kalt, b heiß)."""
    K = A.transition_kernel(0.0, 0.0, A.Form(m=0, rho=0.0), 0.25)
    assert K[1, 1, 2, 0] == pytest.approx(0.375)
    assert K[1, 1, 1, 1] == pytest.approx(0.25)
    assert K[1, 1, 0, 2] == pytest.approx(0.375)
    assert K[1, 1].sum() == pytest.approx(1.0)


def test_kernel_with_memory_one_half_by_hand():
    """rho = 0,5: im Zustand 0 beider Mannschaften bleibt (0, 0) nach Sieg mit 0,375·0,25, nach Remis mit 0,25·1, nach Niederlage mit 0,375·0,25 = 0,4375."""
    K = A.transition_kernel(0.0, 0.0, A.Form(m=0, rho=0.5), 0.25)
    assert K[1, 1, 1, 1] == pytest.approx(0.375 * 0.25 + 0.25 + 0.375 * 0.25)
    assert K[1, 1, 2, 1] == pytest.approx(0.375 * 0.5 * 0.5)       # a gewinnt und setzt zurück, b hat die Niederlage nicht übernommen: bleibt im Zustand 0
    assert K[1, 1, 2, 2] == pytest.approx(0.0, abs=1e-12)          # beide heiß ist unmöglich: ein Sieg von a heißt Niederlage von b


def test_form_rejects_rho_one_and_negative_effect():
    with pytest.raises(ValueError):
        A.Form(m=10, rho=1.0)
    with pytest.raises(ValueError):
        A.Form(m=-1, rho=0.5)


# --- Stationäre Verteilung -----------------------------------------------------------------------------------------------------------------

def test_stationary_two_teams_without_form_is_the_outcome_distribution():
    """n = 2, m = 0: die Zustandsverteilung ist die Ergebnisverteilung (Niederlage, Remis, Sieg) gegen den einzigen Gegner."""
    R = np.array([100.0, -100.0])
    e = _e(200.0)
    d = min(0.25, 2 * (1 - e))
    pi = A.stationary_distribution(R, A.Form(m=0, rho=0.5), 0.25)
    assert pi[0] == pytest.approx([1 - e - d / 2, d, e - d / 2], abs=1e-12)
    assert pi[1] == pytest.approx([e - d / 2, d, 1 - e - d / 2], abs=1e-12)


def test_stationary_rows_are_distributions_and_the_strong_team_is_hotter():
    pi = A.stationary_distribution(S.ratings(8, 300), A.Form(m=50, rho=0.5), 0.25)
    assert pi.shape == (8, 3) and pi.sum(axis=1) == pytest.approx(np.ones(8)) and (pi >= 0).all()
    assert (np.diff(pi[:, 2]) < 0).all() and (np.diff(pi[:, 0]) > 0).all()      # je schwächer, desto seltener heiß und desto öfter kalt


def test_stationary_distribution_does_not_depend_on_rho():
    R = S.ratings(6, 300)
    base = A.stationary_distribution(R, A.Form(m=80, rho=0.0), 0.25)
    for rho in (0.3, 0.8, 0.95):
        assert A.stationary_distribution(R, A.Form(m=80, rho=rho), 0.25) == pytest.approx(base, abs=1e-12)


def test_start_distribution_options():
    R = S.ratings(4, 300)
    assert A.start_distribution(R, A.Form(m=50, rho=0.5, stationary=False), 0.25).tolist() == [[0.0, 1.0, 0.0]] * 4
    assert A.start_distribution(R, A.Form(m=50, rho=0.5, stationary=True), 0.25) == pytest.approx(A.stationary_distribution(R, A.Form(m=50, rho=0.5), 0.25))


# --- Zufallszahlen ------------------------------------------------------------------------------------------------------------------------

def test_uniform_block_is_bit_identical_to_splitmix64():
    ref = SplitMix64(12345)
    assert np.array_equal(S.uniform_block(12345, 200), np.array([ref.random() for _ in range(200)]))
    assert S.uniform_block(7, 0).size == 0


def test_randomness_is_reproducible_and_shaped():
    r1 = A.draw_randomness(4, 50, 9, 3)
    r2 = A.draw_randomness(4, 50, 9, 3)
    assert r1.pair.shape == (1, 4, 4, 50) and r1.coin.shape == (3, 4, 50) and r1.start.shape == (4, 50) and r1.runs == 50
    assert np.array_equal(r1.pair, r2.pair) and np.array_equal(r1.coin, r2.coin) and np.array_equal(r1.start, r2.start)
    assert not np.array_equal(r1.pair, A.draw_randomness(4, 50, 10, 3).pair)
    assert A.draw_randomness(4, 50, 9, 6, legs=2).pair.shape[0] == 2


def test_initial_states_follow_the_start_distribution():
    pi0 = np.array([[0.2, 0.3, 0.5]] * 2)
    u = np.array([[0.0, 0.19, 0.2, 0.49, 0.5, 0.99]] * 2)
    assert A._initial_states(pi0, u)[0].tolist() == [-1, -1, 0, 0, 1, 1]      # Schwellen bei 0,2 und 0,5


# --- Simulation und exakte Erwartung -------------------------------------------------------------------------------------------------------

def test_two_teams_one_game_expected_points_by_hand():
    """n = 2 gleich stark, m = 0: 3·0,375 + 0,25 = 1,375 für jede Mannschaft; die Summe der Punkte ist 3 - d = 2,75."""
    R = np.zeros(2)
    ex = A.exact_points([[(0, 1)]], R, A.Form(m=0, rho=0.5), 0.25)
    assert ex.points == pytest.approx([1.375, 1.375])
    rnd = A.draw_randomness(2, 40000, 3, 1)
    sim = A.simulate([[(0, 1)]], R, A.Form(m=0, rho=0.5), 0.25, rnd)
    assert sim.points == pytest.approx([1.375, 1.375], abs=0.03)


def test_points_without_form_are_the_sum_of_the_pair_expectations():
    """Handrechnung ohne den Kern: Erwartung je Paar aus der Elo-Formel; Summe über alle Gegner."""
    R = S.ratings(4, 300)
    ex = A.exact_points(S.schedule((0, 1, 2, 3)), R, A.Form(m=0, rho=0.5), 0.25)
    for i in range(4):
        total = 0.0
        for j in range(4):
            if j == i:
                continue
            e = _e(R[i] - R[j])
            d = min(0.25, 2 * min(e, 1 - e))
            total += 3 * (e - d / 2) + d
        assert ex.points[i] == pytest.approx(total)


def test_simulation_is_deterministic_for_the_same_seed():
    R = S.ratings(4, 300)
    rnd = A.draw_randomness(4, 300, 5, 3)
    a = A.simulate(S.schedule((0, 1, 2, 3)), R, A.Form(m=50, rho=0.5), 0.25, rnd)
    b = A.simulate(S.schedule((0, 1, 2, 3)), R, A.Form(m=50, rho=0.5), 0.25, rnd)
    assert np.array_equal(a.points, b.points) and np.array_equal(a.state_mean, b.state_mean)


def test_common_random_numbers_give_exactly_equal_points_in_every_plan_without_form():
    """Der Haken: m = 0 und gemeinsame Zufallszahlen -> dieselben Punkte in jedem Spielplan, Spanne exakt 0 (Ganzzahlen, keine Rundung)."""
    R = S.ratings(6, 300)
    rnd = A.draw_randomness(6, 400, 4, 5)
    pts = [A.simulate(S.schedule(p), R, A.Form(m=0, rho=0.5), 0.25, rnd).points for p in S.assignments(6, 30, 1)]
    assert all(np.array_equal(pts[0], q) for q in pts)


def test_common_random_numbers_exact_zero_also_for_the_double_round():
    R = S.ratings(4, 300)
    rnd = A.draw_randomness(4, 300, 4, 6, legs=2)
    pts = [A.simulate(S.schedule(p, True), R, A.Form(m=0, rho=0.0), 0.25, rnd).points for p in S.assignments(4, 30, 1)]
    assert all(np.array_equal(pts[0], q) for q in pts)


def test_independent_random_numbers_show_a_spread_without_form():
    """Die naive Messung: je Plan eigene Zufallszahlen -> Spanne > 0 ohne jede Wirkung der Reihenfolge (das ist Rauschen)."""
    R = S.ratings(4, 300)
    pts = [A.simulate(S.schedule(p), R, A.Form(m=0, rho=0.5), 0.25, A.draw_randomness(4, 300, 100 + k, 3)).points for k, p in enumerate(S.assignments(4, 12, 1))]
    assert np.ptp(np.array(pts)[:, 0]) > 0.1


def test_exact_points_without_form_are_equal_in_every_plan():
    R = S.ratings(6, 300)
    pts = np.array([A.exact_points(S.schedule(p), R, A.Form(m=0, rho=0.7), 0.25).points for p in S.assignments(6, 12, 1)])
    assert np.ptp(pts, axis=0).max() < 1e-9


def test_state_marginals_are_distributions():
    R = S.ratings(4, 300)
    ex = A.exact_points(S.schedule((0, 1, 2, 3)), R, A.Form(m=60, rho=0.4), 0.25)
    assert ex.state_marginals.shape == (3, 4, 3)
    assert ex.state_marginals.sum(axis=2) == pytest.approx(np.ones((3, 4)))


def test_points_never_exceed_the_maximum_and_total_is_below_three_per_game():
    R = S.ratings(4, 300)
    ex = A.exact_points(S.schedule((0, 1, 2, 3)), R, A.Form(m=100, rho=0.5), 0.25)
    assert (ex.points <= 9).all() and (ex.points >= 0).all()
    assert ex.points.sum() < 3 * 6 and ex.points.sum() >= 2 * 6          # je Partie 3 - d (Sieg/Niederlage) oder 2 (Remis)


def test_without_form_the_strongest_team_leads_and_with_equal_strengths_all_are_equal():
    R = S.ratings(4, 300)
    p = A.exact_points(S.schedule((0, 1, 2, 3)), R, A.Form(m=0, rho=0.5), 0.25).points
    assert (np.diff(p) < 0).all()
    equal = A.exact_points(S.schedule((0, 1, 2, 3)), np.zeros(4), A.Form(m=0, rho=0.5), 0.25).points
    assert equal == pytest.approx([equal[0]] * 4)
    assert equal[0] == pytest.approx(3 * 1.375)                      # drei Partien zu je 1,375 Punkten
