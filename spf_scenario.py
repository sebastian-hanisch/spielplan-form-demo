"""Szenario: Elo-Leiter, Zirkeltafel (Berger-Tafel), Zuordnung der Mannschaften zu den Plätzen, Ausgang einer Partie.

Mannschaft 0 ist die stärkste, Mannschaft n-1 die schwächste. Die Zirkeltafel hat feste Plätze 0 bis n-1; eine Zuordnung `perm` setzt
auf Platz p die Mannschaft `perm[p]`. Verschiedene Zuordnungen sind verschiedene Spielpläne mit denselben Paarungen, nur in anderer Reihenfolge.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np

import spf_constants as C
from spf_rng import SplitMix64


@dataclass(frozen=True)
class Liga:
    """Feste Eigenschaften der Liga (ohne Form): Zahl der Mannschaften, Elo-Leiter, Remisanteil, Rundenart."""

    n: int = C.DEFAULT_N
    span: float = C.DEFAULT_SPAN
    draw: float = C.DEFAULT_DRAW
    double: bool = False


def ratings(n: int, span: float) -> np.ndarray:
    """Elo-Leiter, gleichmäßig von +span/2 (Mannschaft 0) bis -span/2 (Mannschaft n-1)."""
    if n < 2:
        raise ValueError("mindestens zwei Mannschaften")
    return np.linspace(span / 2.0, -span / 2.0, n)


def circle_rounds(n: int) -> list[list[tuple[int, int]]]:
    """Zirkelmethode (Berger-Tafel) für gerades n: n-1 Spieltage mit je n/2 Paarungen von Plätzen.

    Platz 0 bleibt stehen, die übrigen Plätze rücken einen Schritt weiter; in jedem Spieltag spielt Platz i gegen Platz n-1-i der aktuellen Reihe."""
    if n < 2 or n % 2:
        raise ValueError("die Zirkelmethode braucht eine gerade Zahl von Mannschaften (n >= 2)")
    seats = list(range(n))
    rounds = []
    for _ in range(n - 1):
        rounds.append([(seats[i], seats[n - 1 - i]) for i in range(n // 2)])
        seats = [seats[0]] + [seats[-1]] + seats[1:-1]
    return rounds


def schedule(perm, double: bool = False) -> list[list[tuple[int, int]]]:
    """Spielplan zur Zuordnung `perm`: Paarungen als (kleinere, größere) Mannschaftsnummer, Spieltag für Spieltag.

    Doppelrunde: die Rückrunde wiederholt die Hinrunde in derselben Reihenfolge."""
    n = len(perm)
    rounds = [[tuple(sorted((int(perm[a]), int(perm[b])))) for a, b in rnd] for rnd in circle_rounds(n)]
    return rounds + [list(r) for r in rounds] if double else rounds


def assignments(n: int, count: int, seed: int) -> list[tuple[int, ...]]:
    """Zuordnungen der Mannschaften zu den Plätzen: alle n! Zuordnungen, falls n! <= count, sonst `count` zufällige (mit Zurücklegen, SplitMix64)."""
    if math.factorial(n) <= count:
        return list(itertools.permutations(range(n)))
    rng = SplitMix64(seed)
    out = []
    for _ in range(count):
        items = list(range(n))
        rng.shuffle(items)
        out.append(tuple(items))
    return out


def outcome_probs(ra, rb, draw_max):
    """(Sieg von a, Remis, Sieg von b) bei den Elo-Werten ra, rb (auch Arrays).

    Erwartungswert e = 1/(1 + 10^((rb-ra)/400)); Remisanteil d = min(draw_max, 2·min(e, 1-e)); Sieg von a = e - d/2, Sieg von b = 1 - e - d/2.
    Der erwartete Punktanteil von a bleibt dadurch e; der Remisanteil kann nie größer sein, als die Erwartung zulässt."""
    ra = np.asarray(ra, dtype=float)
    rb = np.asarray(rb, dtype=float)
    e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    d = np.minimum(draw_max, 2.0 * np.minimum(e, 1.0 - e))
    return e - d / 2.0, d, 1.0 - e - d / 2.0


def uniform_block(seed: int, count: int) -> np.ndarray:
    """Die ersten `count` Werte von SplitMix64(seed).random() als Array, vektorisiert und bitgleich zu `SplitMix64.random`.

    Zustand k = seed + (k+1)·γ (mod 2^64), danach die Mischfunktion; das ist dieselbe Folge wie `random()` in einer Schleife."""
    gamma = np.uint64(0x9E3779B97F4A7C15)
    k = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed & ((1 << 64) - 1)) + k * gamma
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)
