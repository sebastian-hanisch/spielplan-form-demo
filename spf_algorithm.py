"""Kern: Form als Markow-Zustand mit Gedächtnis, stationäre Startverteilung, Simulation mit gemeinsamen Zufallszahlen und exakte Erwartung.

Zustand je Mannschaft s in {-1, 0, +1} (kalt, normal, heiß); Stärke in einer Partie = Grundstärke + m·s. Nach jeder Partie bleibt der Zustand mit
Wahrscheinlichkeit rho, sonst wird er auf das Ergebnis gesetzt (Sieg +1, Remis 0, Niederlage -1); jede Mannschaft wirft ihre Münze für sich.
rho = 0 und Start in 0 ist das einfache ±m-Modell.

Zwei Rechenwege, bewusst verschieden gebaut:
* `simulate`: Monte-Carlo mit gemeinsamen Zufallszahlen (eine Zahl je Paar und Hin-/Rückspiel, eine je Mannschaft und Spieltag, eine je Mannschaft
  für den Start), damit alle Spielpläne dieselben Zufallszahlen sehen.
* `exact_points`: die gemeinsame Verteilung aller n Zustände (3^n Einträge) Partie für Partie weitergerechnet; liefert die Erwartung ohne Rauschen.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import spf_constants as C
from spf_scenario import outcome_probs, uniform_block


@dataclass(frozen=True)
class Form:
    """Zustandsmodell: Wirkung m (Elo je Zustandsstufe), Gedächtnis rho in [0, 1), Start stationär oder in 0."""

    m: float = C.DEFAULT_M
    rho: float = C.DEFAULT_RHO
    stationary: bool = True

    def __post_init__(self):
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("das Gedächtnis rho muss in [0, 1) liegen (bei rho = 1 wechselt der Zustand nie)")
        if self.m < 0:
            raise ValueError("die Wirkung m darf nicht negativ sein")


# --- Zustandsübergang einer Partie (3 x 3 x 3 x 3) -------------------------------------------------------------------------------------------

def transition_kernel(ra: float, rb: float, form: Form, draw_max: float) -> np.ndarray:
    """K[sa, sb, sa', sb'] (Indizes 0, 1, 2 für die Zustände -1, 0, +1): Übergangswahrscheinlichkeit der Zustände der beiden Mannschaften einer Partie."""
    K = np.zeros((3, 3, 3, 3))
    stay = form.rho
    for ia in range(3):
        for ib in range(3):
            pw, d, pl = (float(x) for x in outcome_probs(ra + form.m * (ia - 1), rb + form.m * (ib - 1), draw_max))
            for p, res_a in ((pw, 2), (d, 1), (pl, 0)):   # Zustandsindex von a nach dem Ergebnis; b hat das Spiegelbild 2 - res_a
                res_b = 2 - res_a
                for ja, pa in ((ia, stay), (res_a, 1.0 - stay)):
                    for jb, pb in ((ib, stay), (res_b, 1.0 - stay)):
                        K[ia, ib, ja, jb] += p * pa * pb
    return K


def stationary_distribution(R, form: Form, draw_max: float, iterations: int = 2000, tol: float = 1e-14) -> np.ndarray:
    """Stationäre Verteilung des Zustands je Mannschaft (Zeilen: Mannschaften, Spalten: -1, 0, +1), im Mittelfeld-Sinn.

    Die Zustandskette einer Mannschaft ist P = rho·I + (1 - rho)·Q mit Q[s, x] = Wahrscheinlichkeit, dass das Ergebnis x ist, wenn die Mannschaft im
    Zustand s antritt; der Gegner ist gleichverteilt unter den anderen Mannschaften und hat seinen eigenen stationären Zustand (unabhängig gezogen).
    Es gilt pi·P = pi genau dann, wenn pi·Q = pi: die stationäre Verteilung hängt nicht an rho. Das ist der Kern der Konstruktion: Gedächtnis ändert,
    wann der Zustand wechselt, nicht, wie er verteilt ist. Berechnet als Fixpunkt (gedämpfte Iteration)."""
    R = np.asarray(R, dtype=float)
    n = len(R)
    sv = np.array([-1.0, 0.0, 1.0])
    pi = np.full((n, 3), 1.0 / 3.0)
    for _ in range(iterations):
        new = np.empty_like(pi)
        for i in range(n):
            Q = np.zeros((3, 3))
            for j in range(n):
                if j == i:
                    continue
                pw, d, pl = outcome_probs(R[i] + form.m * sv[:, None], R[j] + form.m * sv[None, :], draw_max)   # Zeilen: Zustand von i, Spalten: von j
                w = pi[j] / (n - 1)
                Q[:, 2] += pw @ w      # Ergebnis Sieg -> Zustand +1
                Q[:, 1] += d @ w
                Q[:, 0] += pl @ w
            new[i] = _left_stationary(Q)
        step = np.abs(new - pi).max()
        pi = 0.5 * pi + 0.5 * new
        if step < tol:
            break
    return pi


def _left_stationary(Q: np.ndarray) -> np.ndarray:
    """Linker Eigenvektor zum Eigenwert 1 einer stochastischen 3 x 3-Matrix (pi Q = pi, Summe 1)."""
    A = np.vstack([Q.T - np.eye(3), np.ones((1, 3))])
    b = np.array([0.0, 0.0, 0.0, 1.0])
    return np.linalg.lstsq(A, b, rcond=None)[0]


@lru_cache(maxsize=64)
def _stationary_cached(R: tuple, m: float, draw_max: float) -> np.ndarray:
    # rho spielt für die stationäre Verteilung keine Rolle (siehe stationary_distribution): der Zwischenspeicher hängt nur an (R, m, draw_max)
    return stationary_distribution(R, Form(m=m, rho=0.0), draw_max)


def start_distribution(R, form: Form, draw_max: float) -> np.ndarray:
    """Startverteilung des Zustands je Mannschaft: stationär (Standard) oder alle im Zustand 0 (naiv, das einfache Modell)."""
    n = len(R)
    if form.stationary:
        return _stationary_cached(tuple(float(x) for x in R), float(form.m), float(draw_max)).copy()
    pi = np.zeros((n, 3))
    pi[:, 1] = 1.0
    return pi


# --- Zufallszahlen: gemeinsam für alle Spielpläne -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Randomness:
    """Alle Zufallszahlen eines Messlaufs. `pair[leg, a, b]` (a < b) ist die Zahl der Partie a gegen b im Hin- (leg 0) bzw. Rückspiel (leg 1),
    `coin[round, team]` entscheidet, ob der Zustand bleibt, `start[team]` zieht den Anfangszustand; jeweils ein Vektor der Länge `runs`."""

    pair: np.ndarray
    coin: np.ndarray
    start: np.ndarray

    @property
    def runs(self) -> int:
        return self.start.shape[1]


def draw_randomness(n: int, runs: int, seed: int, rounds_total: int, legs: int = 1) -> Randomness:
    """Aus SplitMix64(seed) (plattformfest): gleiche Eingabe, gleiche Zufallszahlen."""
    block = uniform_block(seed, runs * (legs * n * n + rounds_total * n + n))
    pos = legs * n * n * runs
    pair = block[:pos].reshape(legs, n, n, runs)
    coin = block[pos:pos + rounds_total * n * runs].reshape(rounds_total, n, runs)
    start = block[pos + rounds_total * n * runs:].reshape(n, runs)
    return Randomness(pair, coin, start)


def _initial_states(pi0: np.ndarray, start_u: np.ndarray) -> np.ndarray:
    """Anfangszustände (n x runs) mit Werten -1, 0, +1 aus den Zufallszahlen und der Startverteilung (Inversionsverfahren)."""
    c0 = pi0[:, 0:1]
    c1 = pi0[:, 0:1] + pi0[:, 1:2]
    return (start_u >= c0).astype(float) + (start_u >= c1).astype(float) - 1.0


# --- Simulation ---------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SimResult:
    points: np.ndarray        # mittlere Punkte je Mannschaft (n,)
    state_mean: np.ndarray    # mittlerer Zustand vor einer Partie, je Mannschaft
    state_sq: np.ndarray      # mittleres Quadrat des Zustands vor einer Partie, je Mannschaft
    points_sd: np.ndarray     # Standardabweichung der Saisonpunkte über die Läufe, je Mannschaft

    @property
    def state_var(self) -> np.ndarray:
        return self.state_sq - self.state_mean ** 2


def simulate(sched, R, form: Form, draw_max: float, rnd: Randomness, pi0: np.ndarray | None = None) -> SimResult:
    """Ein Spielplan, `rnd.runs` Läufe, alles vektorisiert über die Läufe. Punkte 3 / 1 / 0."""
    R = np.asarray(R, dtype=float)
    n = len(R)
    runs = rnd.runs
    if pi0 is None:
        pi0 = start_distribution(R, form, draw_max)
    state = _initial_states(pi0, rnd.start)
    points = np.zeros((n, runs))
    s_sum = np.zeros(n)
    sq_sum = np.zeros(n)
    per_leg = len(sched) // rnd.pair.shape[0]
    for r, games in enumerate(sched):
        leg = r // per_leg
        s_sum += state.sum(axis=1)
        sq_sum += (state ** 2).sum(axis=1)
        new = state.copy()
        for a, b in games:              # a < b: die Ausrichtung je Paar ist fest, damit dieselbe Zufallszahl überall dasselbe Ergebnis liefert
            pw, d, _ = outcome_probs(R[a] + form.m * state[a], R[b] + form.m * state[b], draw_max)
            u = rnd.pair[leg, a, b]
            win = u < pw
            draw = (~win) & (u < pw + d)
            points[a] += C.POINTS_WIN * win + C.POINTS_DRAW * draw
            points[b] += C.POINTS_WIN * (~win & ~draw) + C.POINTS_DRAW * draw
            res_a = np.where(win, 1.0, np.where(draw, 0.0, -1.0))
            new[a] = np.where(rnd.coin[r, a] < form.rho, state[a], res_a)
            new[b] = np.where(rnd.coin[r, b] < form.rho, state[b], -res_a)
        state = new
    count = len(sched) * runs
    mean = s_sum / count
    return SimResult(points.mean(axis=1), mean, sq_sum / count, points.std(axis=1))


# --- Exakte Erwartung -----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ExactResult:
    points: np.ndarray        # erwartete Punkte je Mannschaft
    state_marginals: np.ndarray   # Zustandsverteilung je Mannschaft vor jedem Spieltag (Spieltage x n x 3)


def exact_points(sched, R, form: Form, draw_max: float, pi0: np.ndarray | None = None, kernels: dict | None = None) -> ExactResult:
    """Erwartete Punkte aus der gemeinsamen Verteilung aller Zustände (Tensor mit n Achsen der Länge 3), Partie für Partie.

    Kosten 3^n·(Zahl der Partien): für n bis 8 in Millisekunden, für n = 10 spürbar langsamer (die App rechnet dort nur die Simulation)."""
    R = np.asarray(R, dtype=float)
    n = len(R)
    if pi0 is None:
        pi0 = start_distribution(R, form, draw_max)
    joint = pi0[0]
    for i in range(1, n):
        joint = np.multiply.outer(joint, pi0[i])
    points = np.zeros(n)
    marginals = np.zeros((len(sched), n, 3))
    cache = {} if kernels is None else kernels   # Übergangsmatrizen je Paar; für viele Spielpläne derselben Liga von außen mitgeben
    for r, games in enumerate(sched):
        for i in range(n):
            marginals[r, i] = joint.sum(axis=tuple(k for k in range(n) if k != i))
        for a, b in games:
            pair_marg = joint.sum(axis=tuple(k for k in range(n) if k not in (a, b)))   # Achsen in der Reihenfolge a, b (a < b)
            sa = np.array([-1.0, 0.0, 1.0])
            pw, d, pl = outcome_probs(R[a] + form.m * sa[:, None], R[b] + form.m * sa[None, :], draw_max)
            points[a] += float((pair_marg * (C.POINTS_WIN * pw + C.POINTS_DRAW * d)).sum())
            points[b] += float((pair_marg * (C.POINTS_WIN * pl + C.POINTS_DRAW * d)).sum())
            if (a, b) not in cache:
                cache[(a, b)] = transition_kernel(R[a], R[b], form, draw_max).reshape(9, 9)
            front = np.moveaxis(joint, (a, b), (0, 1))
            shape = front.shape
            updated = cache[(a, b)].T @ front.reshape(9, -1)
            joint = np.moveaxis(updated.reshape(shape), (0, 1), (a, b))
    return ExactResult(points, marginals)
