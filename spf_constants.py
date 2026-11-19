"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Liga -------------------------------------------------------------------------------------------------------------------------------
N_OPTIONS = (4, 6, 8, 10)                  # gerade Mannschaftszahlen (Zirkelmethode ohne Freilos); 4 und 6 sind exakt aufzählbar
DEFAULT_N = 8
SPAN_MIN, SPAN_MAX, SPAN_STEP, DEFAULT_SPAN = 0, 600, 25, 300     # Elo-Abstand zwischen der stärksten und der schwächsten Mannschaft
DRAW_MIN, DRAW_MAX, DRAW_STEP, DEFAULT_DRAW = 0.0, 0.5, 0.05, 0.25  # Höchstwert des Remisanteils einer Partie
ROUNDS_OPTIONS = ("Einfache Runde", "Doppelrunde")
DEFAULT_ROUNDS = "Einfache Runde"
POINTS_WIN, POINTS_DRAW = 3, 1             # Punktwertung 3 / 1 / 0

# --- Form (Zustandsmodell) ----------------------------------------------------------------------------------------------------------------
M_MIN, M_MAX, M_STEP, DEFAULT_M = 0, 150, 5, 50     # Elo-Wirkung je Zustandsstufe: Stärke = Grundstärke + m · Zustand
RHO_MIN, RHO_MAX, RHO_STEP, DEFAULT_RHO = 0.0, 0.95, 0.05, 0.5   # Gedächtnis: Wahrscheinlichkeit, dass der Zustand nach einer Partie bleibt
START_OPTIONS = ("Stationär (Standard)", "Nullzustand (naiv)")
DEFAULT_START = START_OPTIONS[0]
STATES = (-1, 0, 1)                        # kalt, normal, heiß

# --- Messung ----------------------------------------------------------------------------------------------------------------------------
L_MIN, L_MAX, L_STEP, DEFAULT_L = 20, 300, 20, 100     # Zahl der zufälligen Zuordnungen der Mannschaften zur Zirkeltafel
S_MIN, S_MAX, S_STEP, DEFAULT_S = 500, 4000, 500, 2000  # Zahl der Läufe je Plan in der Simulation
SEED_MAX, DEFAULT_SEED = 999999, 2026
EXACT_MAX_N = 8                            # bis hierher rechnet die App die Erwartung auch exakt (3^n Gesamtzustände)
EXACT_SCHEDULES = 24                       # so viele Zuordnungen vergleicht die App Simulation gegen exakt

# Feste Messreihen der Experimente: nie ein einzelner Zufalls-Seed als Befund.
SWEEP_SEEDS = (2026, 2027, 2028)
SWEEP_L, SWEEP_S = 40, 1000                # Experimente: weniger Zuordnungen und Läufe als der Hauptlauf (Laufzeit)
M_SWEEP = (0, 25, 50, 100)
RHO_SWEEP = (0.0, 0.5, 0.8, 0.95)

# --- Farben -------------------------------------------------------------------------------------------------------------------------------
COLOR_BAND = "rgba(47,107,101,0.35)"
COLOR_MAIN = "#2F6B65"
COLOR_ALT = "#c0504d"
COLOR_LINE = "#4c78a8"
COLOR_GREY = "rgba(150,150,150,0.7)"

PRESETS = {
    "Standard (n = 8, m = 50, ρ = 0,5)": {
        "n": DEFAULT_N, "span": DEFAULT_SPAN, "draw": DEFAULT_DRAW, "rounds": DEFAULT_ROUNDS, "m": DEFAULT_M, "rho": DEFAULT_RHO,
        "start": DEFAULT_START, "l": DEFAULT_L, "s": DEFAULT_S, "seed": DEFAULT_SEED,
    },
    "Ohne Form (m = 0): der Haken": {
        "n": DEFAULT_N, "span": DEFAULT_SPAN, "draw": DEFAULT_DRAW, "rounds": DEFAULT_ROUNDS, "m": 0, "rho": DEFAULT_RHO,
        "start": DEFAULT_START, "l": DEFAULT_L, "s": DEFAULT_S, "seed": DEFAULT_SEED,
    },
    "Form ohne Gedächtnis (ρ = 0)": {
        "n": DEFAULT_N, "span": DEFAULT_SPAN, "draw": DEFAULT_DRAW, "rounds": DEFAULT_ROUNDS, "m": DEFAULT_M, "rho": 0.0,
        "start": DEFAULT_START, "l": DEFAULT_L, "s": DEFAULT_S, "seed": DEFAULT_SEED,
    },
    "Starkes Gedächtnis (ρ = 0,8)": {
        "n": DEFAULT_N, "span": DEFAULT_SPAN, "draw": DEFAULT_DRAW, "rounds": DEFAULT_ROUNDS, "m": DEFAULT_M, "rho": 0.8,
        "start": DEFAULT_START, "l": DEFAULT_L, "s": DEFAULT_S, "seed": DEFAULT_SEED,
    },
    "Naiver Start in 0 (Äpfel mit Birnen)": {
        "n": DEFAULT_N, "span": DEFAULT_SPAN, "draw": DEFAULT_DRAW, "rounds": DEFAULT_ROUNDS, "m": DEFAULT_M, "rho": 0.8,
        "start": START_OPTIONS[1], "l": DEFAULT_L, "s": DEFAULT_S, "seed": DEFAULT_SEED,
    },
    "Doppelrunde (Hin und Rück)": {
        "n": DEFAULT_N, "span": DEFAULT_SPAN, "draw": DEFAULT_DRAW, "rounds": ROUNDS_OPTIONS[1], "m": DEFAULT_M, "rho": DEFAULT_RHO,
        "start": DEFAULT_START, "l": DEFAULT_L, "s": DEFAULT_S, "seed": DEFAULT_SEED,
    },
    "Kleine Liga (n = 4, alle 24 Zuordnungen)": {
        "n": 4, "span": DEFAULT_SPAN, "draw": DEFAULT_DRAW, "rounds": DEFAULT_ROUNDS, "m": 100, "rho": DEFAULT_RHO,
        "start": DEFAULT_START, "l": DEFAULT_L, "s": DEFAULT_S, "seed": DEFAULT_SEED,
    },
}
PRESET_HELP = {
    "Standard (n = 8, m = 50, ρ = 0,5)": "Acht Mannschaften, einfache Runde, Elo-Leiter von +150 bis −150, Remisanteil höchstens 0,25, Form mit Wirkung 50 Elo und Gedächtnis 0,5.",
    "Ohne Form (m = 0): der Haken": "Form wirkt nicht. Dann ist die erwartete Punktzahl jeder Mannschaft in jedem Spielplan gleich: die Spanne ist 0, egal wie die Mannschaften zugeordnet sind.",
    "Form ohne Gedächtnis (ρ = 0)": "Der Zustand ist immer das Ergebnis der letzten Partie (Sieg +1, Remis 0, Niederlage −1): das einfache Modell.",
    "Starkes Gedächtnis (ρ = 0,8)": "Der Zustand bleibt nach einer Partie mit Wahrscheinlichkeit 0,8 und wechselt seltener. Die Verteilung des Zustands bleibt dieselbe (stationärer Start).",
    "Naiver Start in 0 (Äpfel mit Birnen)": "Alle starten im Zustand normal: bei großem Gedächtnis dauert es lange, bis Form entsteht. Die Streuung des Zustands hängt dann am Gedächtnis (bei ρ = 0,8 und m = 50 etwa 0,31 statt 0,65), und die Spanne fällt kleiner aus.",
    "Doppelrunde (Hin und Rück)": "Hin- und Rückrunde in gleicher Reihenfolge (14 statt 7 Spieltage bei 8 Mannschaften): mehr Partien, mehr Gelegenheiten für Form.",
    "Kleine Liga (n = 4, alle 24 Zuordnungen)": "Vier Mannschaften, drei Spieltage: Es gibt nur 24 Zuordnungen, die App rechnet alle durch. Gut, um die exakte Rechnung gegen die Simulation zu halten.",
}
