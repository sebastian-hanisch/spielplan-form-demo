"""Auswertung: Messläufe über viele Zuordnungen, Kennzahlen, Experimente und Verdikte.

Messgröße: die Spanne (größter minus kleinster Wert) der erwarteten Punkte einer Mannschaft über die Zuordnungen der Mannschaften zur Zirkeltafel.
Erwartete Punkte kommen aus der Simulation mit gemeinsamen Zufallszahlen und, bis n = 8, zusätzlich exakt (ohne Rauschen)."""

import math
from dataclasses import dataclass

import numpy as np

import spf_algorithm as A
import spf_constants as C
import spf_scenario as S

ZERO_TOL = 1e-9   # Spannen darunter gelten als null (Gleitkomma-Rest der exakten Rechnung)


@dataclass(frozen=True)
class Settings:
    """Alle Einstellungen der App. Immer mit Schlüsselwörtern aufrufen, nie positional."""

    n: int = C.DEFAULT_N
    span: int = C.DEFAULT_SPAN
    draw: float = C.DEFAULT_DRAW
    rounds: str = C.DEFAULT_ROUNDS
    m: int = C.DEFAULT_M
    rho: float = C.DEFAULT_RHO
    start: str = C.DEFAULT_START
    l: int = C.DEFAULT_L
    s: int = C.DEFAULT_S
    seed: int = C.DEFAULT_SEED

    @property
    def double(self) -> bool:
        return self.rounds == C.ROUNDS_OPTIONS[1]

    @property
    def form(self) -> A.Form:
        return A.Form(m=float(self.m), rho=float(self.rho), stationary=self.start == C.START_OPTIONS[0])

    @property
    def exact_possible(self) -> bool:
        return self.n <= C.EXACT_MAX_N

    @property
    def max_points(self) -> int:
        """Höchstzahl der Punkte einer Mannschaft in der Saison (alle Partien gewonnen)."""
        return C.POINTS_WIN * (self.n - 1) * (2 if self.double else 1)


def span_of(values) -> float:
    """Spanne (max - min) eines Vektors; Gleitkomma-Reste unter ZERO_TOL zählen als 0."""
    v = np.asarray(values, dtype=float)
    d = float(v.max() - v.min())
    return 0.0 if d < ZERO_TOL else d


def team_spans(P) -> np.ndarray:
    """Spanne je Mannschaft (Spalten von P: Mannschaften, Zeilen: Zuordnungen)."""
    return np.array([span_of(P[:, i]) for i in range(P.shape[1])])


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    ratings: np.ndarray
    perms: list
    sim_points: np.ndarray               # Zuordnungen x Mannschaften (Simulation, gemeinsame Zufallszahlen)
    exact_points: np.ndarray | None      # dasselbe exakt (nur n <= 8)
    exact_state_var: float | None        # mittlere Zustandsstreuung (exakt, in einem zufällig gezogenen Spielplan und Spieltag, gemittelt über die Mannschaften)
    sim_state_var: float
    sim_points_sd: np.ndarray            # Standardabweichung der Saisonpunkte je Mannschaft (Mittel über die Zuordnungen)
    pi0: np.ndarray

    @property
    def main_points(self) -> np.ndarray:
        """Erwartete Punkte: exakt, wo möglich, sonst die Simulation."""
        return self.exact_points if self.exact_points is not None else self.sim_points

    @property
    def top_span(self) -> float:
        return span_of(self.main_points[:, 0])

    @property
    def max_span(self) -> float:
        return float(team_spans(self.main_points).max())

    @property
    def sim_top_span(self) -> float:
        return span_of(self.sim_points[:, 0])

    @property
    def top_sd(self) -> float:
        return float(self.sim_points_sd[0])


def state_variance(marginals: np.ndarray) -> np.ndarray:
    """Zustandsstreuung je Mannschaft aus Zustandsverteilungen (letzte Achse: -1, 0, +1; davor beliebig viele Achsen, über die gemittelt wird).

    Aufruf mit den Verteilungen aller Spieltage und aller Zuordnungen (Zuordnungen x Spieltage x n x 3): die Streuung des Zustands in einem zufällig
    gezogenen Spielplan und Spieltag. Nur dieses Maß ist bei m = 0 für jedes rho exakt gleich (über alle Zuordnungen gemittelt ist die Verteilung stationär);
    die Streuung innerhalb eines festen Spielplans ist bei kleinem rho etwas niedriger, weil dort der letzte Gegner die Verteilung prägt."""
    sv = np.array([-1.0, 0.0, 1.0])
    mg = marginals.reshape(-1, marginals.shape[-2], 3).mean(axis=0)
    return mg @ sv ** 2 - (mg @ sv) ** 2


def sim_plans(liga: S.Liga, form: A.Form, perms, runs: int, seed: int, independent: bool = False, pi0=None):
    """Simulation aller Zuordnungen. Gemeinsame Zufallszahlen (Standard): alle Spielpläne sehen dieselben Zahlen.
    `independent=True` zieht für jede Zuordnung eigene Zufallszahlen (die naive Messung, die bei m = 0 Rauschen als Plan-Glück zeigt).
    Rückgabe: Punkte (Zuordnungen x n), mittlere Zustandsstreuung (wie `state_variance`), mittlere Standardabweichung der Saisonpunkte je Mannschaft."""
    R = S.ratings(liga.n, liga.span)
    rounds_total = (liga.n - 1) * (2 if liga.double else 1)
    legs = 2 if liga.double else 1
    if pi0 is None:
        pi0 = A.start_distribution(R, form, liga.draw)
    shared = None if independent else A.draw_randomness(liga.n, runs, seed + 1, rounds_total, legs)
    pts, means, squares, sd = [], [], [], []
    for k, perm in enumerate(perms):
        rnd = shared if shared is not None else A.draw_randomness(liga.n, runs, seed + 1 + 7919 * (k + 1), rounds_total, legs)
        res = A.simulate(S.schedule(perm, liga.double), R, form, liga.draw, rnd, pi0)
        pts.append(res.points)
        means.append(res.state_mean)
        squares.append(res.state_sq)
        sd.append(res.points_sd)
    var = np.mean(squares, axis=0) - np.mean(means, axis=0) ** 2          # Streuung über alle Zuordnungen, Spieltage und Läufe
    return np.array(pts), float(var.mean()), np.mean(sd, axis=0)


def exact_plans(liga: S.Liga, form: A.Form, perms, pi0=None):
    """Exakte erwartete Punkte aller Zuordnungen (n <= 8): Punkte (Zuordnungen x n) und mittlere Zustandsstreuung."""
    R = S.ratings(liga.n, liga.span)
    if pi0 is None:
        pi0 = A.start_distribution(R, form, liga.draw)
    pts, marg = [], []
    for perm in perms:
        ex = A.exact_points(S.schedule(perm, liga.double), R, form, liga.draw, pi0)
        pts.append(ex.points)
        marg.append(ex.state_marginals)
    return np.array(pts), float(state_variance(np.array(marg)).mean())


def analyse(settings: Settings) -> Analysis:
    liga = S.Liga(n=settings.n, span=float(settings.span), draw=settings.draw, double=settings.double)
    form = settings.form
    R = S.ratings(liga.n, liga.span)
    pi0 = A.start_distribution(R, form, liga.draw)
    perms = S.assignments(liga.n, settings.l, settings.seed)
    sim, sim_var, sd = sim_plans(liga, form, perms, settings.s, settings.seed, pi0=pi0)
    ex, ex_var = (exact_plans(liga, form, perms, pi0) if settings.exact_possible else (None, None))
    return Analysis(settings, R, perms, sim, ex, ex_var, sim_var, sd, pi0)


# --- Beispielplan und Gegnerfolge ---------------------------------------------------------------------------------------------------------

def team_name(i: int) -> str:
    """Anzeigename: M1 ist die stärkste Mannschaft."""
    return f"M{i + 1}"


def schedule_rows(perm, double: bool = False) -> list[dict]:
    """Zeilen für die Tabelle eines Spielplans: Spieltag und die Paarungen (stärkere Mannschaft zuerst)."""
    return [{"Spieltag": r + 1, "Paarungen": ", ".join(f"{team_name(a)}–{team_name(b)}" for a, b in games)} for r, games in enumerate(S.schedule(perm, double))]


def opponent_order(perm, team: int, double: bool = False) -> list[int]:
    """Gegner von `team` Spieltag für Spieltag (als Mannschaftsnummern, 0 = stärkste)."""
    out = []
    for games in S.schedule(perm, double):
        for a, b in games:
            if team == a:
                out.append(b)
            elif team == b:
                out.append(a)
    return out


def extreme_plans(P, perms, team: int = 0) -> tuple[int, int]:
    """Indizes der Zuordnung mit den meisten und den wenigsten erwarteten Punkten für `team`."""
    col = P[:, team]
    return int(np.argmax(col)), int(np.argmin(col))


# --- Experimente --------------------------------------------------------------------------------------------------------------------------

def _measure(liga: S.Liga, form: A.Form, perms, runs: int, seed: int):
    """Erwartete Punkte aller Zuordnungen und Zustandsstreuung: exakt bis n = 8, sonst Simulation."""
    if liga.n <= C.EXACT_MAX_N:
        return exact_plans(liga, form, perms)
    pts, var, _ = sim_plans(liga, form, perms, runs, seed)
    return pts, var


def _liga(settings: Settings) -> S.Liga:
    return S.Liga(n=settings.n, span=float(settings.span), draw=settings.draw, double=settings.double)


def sweep_m(settings: Settings) -> list[dict]:
    """Experiment 1: Wirkung m der Form. Je m und je Messreihe (SWEEP_SEEDS): Spanne des Stärksten (exakt bzw. Simulation mit gemeinsamen Zufallszahlen)
    und die Spanne, die eine naive Simulation mit je Zuordnung eigenen Zufallszahlen zeigt. Median über die Messreihen."""
    liga = _liga(settings)
    rows = []
    for m in C.M_SWEEP:
        form = A.Form(m=float(m), rho=float(settings.rho), stationary=settings.form.stationary)
        exact, shared, naive = [], [], []
        for seed in C.SWEEP_SEEDS:
            perms = S.assignments(liga.n, C.SWEEP_L, seed)
            pts, _ = _measure(liga, form, perms, C.SWEEP_S, seed)
            exact.append(span_of(pts[:, 0]))
            sh, _, _ = sim_plans(liga, form, perms, C.SWEEP_S, seed)
            shared.append(span_of(sh[:, 0]))
            nv, _, _ = sim_plans(liga, form, perms, C.SWEEP_S, seed, independent=True)
            naive.append(span_of(nv[:, 0]))
        rows.append({"m": m, "exakt": float(np.median(exact)), "gemeinsam": float(np.median(shared)), "naiv": float(np.median(naive))})
    return rows


def sweep_rho(settings: Settings) -> list[dict]:
    """Experiment 2: Gedächtnis rho, bei stationärem Start und bei Start in 0 (naiv). Je Zeile die Spanne des Stärksten, die größte Spanne
    irgendeiner Mannschaft und die mittlere Zustandsstreuung. Median über die Messreihen."""
    liga = _liga(settings)
    rows = []
    for stationary in (True, False):
        for rho in C.RHO_SWEEP:
            form = A.Form(m=float(settings.m), rho=float(rho), stationary=stationary)
            top, mx, var = [], [], []
            for seed in C.SWEEP_SEEDS:
                perms = S.assignments(liga.n, C.SWEEP_L, seed)
                pts, v = _measure(liga, form, perms, C.SWEEP_S, seed)
                top.append(span_of(pts[:, 0]))
                mx.append(float(team_spans(pts).max()))
                var.append(v)
            rows.append({"start": "stationär" if stationary else "Nullzustand", "rho": rho, "top": float(np.median(top)),
                         "max": float(np.median(mx)), "var": float(np.median(var))})
    return rows


# --- Verdikte ------------------------------------------------------------------------------------------------------------------------------

def verdict_core(a: Analysis) -> tuple[str, str]:
    """Verdikt der Kernfrage: wie viel verschiebt die bloße Reihenfolge die erwartete Punktzahl des Stärksten?"""
    st = a.settings
    top = a.top_span
    way = "exakt" if a.exact_points is not None else "in der Simulation"
    if st.m == 0:
        return ("success",
                f"Ohne Form (m = 0) ist die erwartete Punktzahl jeder Mannschaft in **jedem** Spielplan gleich: Spanne des Stärksten {top:.2f} Punkte "
                f"({way}, {len(a.perms)} Zuordnungen). Die Reihenfolge der Gegner verschiebt hier nichts; was eine Simulation mit je Plan eigenen Zufallszahlen als "
                "„Plan-Glück“ zeigt, ist Rauschen.")
    best, worst = extreme_plans(a.main_points, a.perms)
    hi, lo = a.main_points[best, 0], a.main_points[worst, 0]
    share = 100.0 * top / st.max_points
    sd_share = 100.0 * top / a.top_sd if a.top_sd > 0 else float("nan")
    kind = "warning" if share >= 2.0 else "info"
    total = math.factorial(st.n)
    total_de = f"{total:,}".replace(",", ".")
    sample = f" Die Spanne gilt für diese {len(a.perms)} von {total_de} Zuordnungen; über alle ist sie mindestens so groß." if len(a.perms) < total else ""
    return (kind,
            f"Mit Form (m = {st.m} Elo, Gedächtnis ρ = {st.rho:.2f}) erwartet der Stärkste im besten der {len(a.perms)} Zuordnungen {hi:.2f}, im schlechtesten {lo:.2f} "
            f"Punkte ({way}): Spanne **{top:.2f}** Punkte, das sind {share:.1f} % der höchstens {st.max_points} möglichen. "
            f"Zum Vergleich: die Saisonpunkte des Stärksten streuen um {a.top_sd:.2f} (Standardabweichung über die Läufe); die Spanne ist {sd_share:.0f} % davon." + sample)


def verdict_m(rows: list[dict], exact: bool = True) -> tuple[str, str]:
    """Verdikt zu Experiment 1 (m-Sweep). `exact`: ob die Erwartung exakt gerechnet wurde (n <= 8) oder per Simulation (n = 10)."""
    zero = next(r for r in rows if r["m"] == 0)
    last = rows[-1]
    way = "exakt" if exact else "in der Simulation mit gemeinsamen Zufallszahlen"
    text = (f"Bei m = 0 ist die Spanne des Stärksten {way} **{zero['exakt']:.2f}** Punkte, auch mit gemeinsamen Zufallszahlen in der Simulation ({zero['gemeinsam']:.2f}). "
            f"Eine naive Simulation mit je Zuordnung eigenen Zufallszahlen zeigt dort dennoch **{zero['naiv']:.2f}** Punkte Spanne: reines Rauschen (die Zahl hängt an der Zahl der Läufe und der Zuordnungen). "
            f"Bei m = {last['m']} sind es {last['exakt']:.2f} Punkte (Median über {len(C.SWEEP_SEEDS)} Messreihen).")
    return ("info" if zero["exakt"] == 0 else "warning"), text


def verdict_rho(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu Experiment 2 (rho-Sweep): Zustandsstreuung bei stationärem Start für alle rho gleich, die Spanne fällt mit dem Gedächtnis."""
    stat = [r for r in rows if r["start"] == "stationär"]
    zero = [r for r in rows if r["start"] == "Nullzustand"]
    v_lo, v_hi = min(r["var"] for r in stat), max(r["var"] for r in stat)
    z_lo, z_hi = min(r["var"] for r in zero), max(r["var"] for r in zero)
    first, last = stat[0], stat[-1]
    z_first, z_last = zero[0], zero[-1]
    text = (f"Bei stationärem Start liegt die Zustandsstreuung für alle ρ zwischen {v_lo:.3f} und {v_hi:.3f} (Abstand {v_hi - v_lo:.3f}); beim Start im Nullzustand zwischen "
            f"{z_lo:.3f} und {z_hi:.3f} (Abstand {z_hi - z_lo:.3f}). ")
    if first["top"] == 0 and last["top"] == 0:
        text += "Die Spanne des Stärksten ist in allen Zeilen 0: ohne Wirkung der Form (m = 0) gibt es nichts zu verteilen."
    else:
        text += (f"Trotz gleicher Streuung fällt die Spanne des Stärksten bei stationärem Start von {first['top']:.2f} (ρ = {first['rho']:.2f}) auf {last['top']:.2f} (ρ = {last['rho']:.2f}); "
                 f"beim Start im Nullzustand von {z_first['top']:.2f} auf {z_last['top']:.2f}.")
    return "info", text
