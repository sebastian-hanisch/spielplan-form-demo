"""Spielplanreihenfolge und Form: was die Reihenfolge der Gegner verschiebt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Frage: Wie viele erwartete Punkte kann die bloße Reihenfolge der Gegner (die Zuordnung der Mannschaften zu den Plätzen der Zirkeltafel) verschieben?
Ohne Form (m = 0) in diesem Modell keine; mit Form (Markow-Zustand mit Gedächtnis) wächst die Spanne. Alles simuliert, keine Felddaten.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import spf_constants as C
from spf_evaluation import (
    Settings,
    analyse,
    extreme_plans,
    opponent_order,
    schedule_rows,
    sweep_m,
    sweep_rho,
    team_name,
    team_spans,
    verdict_core,
    verdict_m,
    verdict_rho,
)
from spf_layout import dunkles_design, kopf_css
from spf_presets import (
    bounds,
    choices,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from spf_visualization import build_bands, build_hist, build_m_sweep, build_rho_sweep

st.set_page_config(page_title="Spielplanreihenfolge und Form – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep_m(settings):
    return sweep_m(settings)


@st.cache_data(show_spinner=False)
def _sweep_rho(settings):
    return sweep_rho(settings)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in spf_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🗓️ Spielplanreihenfolge und Form: was die Reihenfolge der Gegner verschiebt")
st.markdown(
    """
Eine Liga spielt nach der **Zirkelmethode** (Berger-Tafel): jede Mannschaft trifft jede andere, nur die **Reihenfolge der Gegner** hängt davon ab, welche Mannschaft auf welchem
Platz der Tafel steht. Frage dieser Demo: **Wie viele erwartete Punkte kann allein diese Reihenfolge verschieben?** Ohne **Form** (der Zustand einer Mannschaft wirkt nicht auf ihre
nächste Partie) in diesem Modell keine: die Spanne ist 0. Mit Form, modelliert als **Markow-Zustand mit Gedächtnis**, wächst sie. Die [Doppelrunde mit minimalen Breaks](https://sebastianhanisch-double-round-robin-demo.streamlit.app/)
konstruiert Pläne, ohne ein Ergebnis zu bewerten; hier wird bewertet, was die Reihenfolge der Gegner kostet oder bringt (Heim und Auswärts sind nicht modelliert).

Weiter unten: **🔬 Experimente** (Haken bei m = 0, Gedächtnis), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Liga**")
    n = st.select_slider("Mannschaften n", options=choices("n_select"), key="n_select",
                         help="Gerade Zahl (Zirkelmethode ohne Freilos). Bis n = 8 rechnet die App die Erwartung zusätzlich exakt; n = 10 nur per Simulation.")
    span = st.slider("Elo-Abstand Stärkster – Schwächster", *bounds("span_slider"), step=C.SPAN_STEP, key="span_slider",
                     help="Die Stärken liegen gleichmäßig auf einer Elo-Leiter von +Abstand/2 (M1) bis −Abstand/2 (letzte Mannschaft). 300 Elo: M1 schlägt die letzte Mannschaft mit etwa 85 % Erwartung.")
    draw = st.slider("Remisanteil (Höchstwert)", *bounds("draw_slider"), step=C.DRAW_STEP, key="draw_slider",
                     help="Obergrenze des Remisanteils einer Partie; bei ungleichen Mannschaften ist er kleiner, damit die Elo-Erwartung erhalten bleibt.")
    rounds = st.radio("Runden", choices("rounds_radio"), key="rounds_radio",
                      help="Doppelrunde: die Rückrunde wiederholt die Hinrunde in gleicher Reihenfolge (jedes Paar spielt zweimal, Hin- und Rückspiel mit eigenen Zufallszahlen).")
    st.markdown("**Form**")
    m = st.slider("Wirkung der Form m (Elo)", *bounds("m_slider"), step=C.M_STEP, key="m_slider",
                  help="Stärke in der Partie = Grundstärke + m · Zustand (heiß +1, normal 0, kalt −1). m = 0: Form wirkt nicht.")
    rho = st.slider("Gedächtnis ρ", *bounds("rho_slider"), step=C.RHO_STEP, key="rho_slider",
                    help="Wahrscheinlichkeit, dass der Zustand nach einer Partie bleibt. Sonst wird er auf das Ergebnis gesetzt (Sieg +1, Remis 0, Niederlage −1). ρ = 0: immer das Ergebnis der letzten Partie.")
    start = st.radio("Start des Zustands", choices("start_radio"), key="start_radio",
                     help="Stationär: Jede Mannschaft startet in der zu ihrer Zustandskette passenden Verteilung (im Mittelfeld-Sinn stationär, hängt nicht an ρ). Nullzustand: alle starten normal (die naive Wahl).")
    st.markdown("**Messung**")
    l = st.slider("Zahl der Zuordnungen", *bounds("l_slider"), step=C.L_STEP, key="l_slider",
                  help="So viele zufällige Zuordnungen der Mannschaften zur Zirkeltafel werden verglichen (bei n = 4 alle 24, sobald die Zahl mindestens 24 ist).")
    runs = st.slider("Läufe je Spielplan (Simulation)", *bounds("s_slider"), step=C.S_STEP, key="s_slider",
                     help="Simulierte Saisons je Spielplan. Alle Spielpläne sehen dieselben Zufallszahlen.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_select": int(n), "span_slider": int(span), "draw_slider": float(draw), "rounds_radio": rounds, "m_slider": int(m),
                   "rho_slider": float(rho), "start_radio": start, "l_slider": int(l), "s_slider": int(runs), "seed_input": int(seed)})

settings = Settings(n=int(n), span=int(span), draw=float(draw), rounds=rounds, m=int(m), rho=float(rho), start=start, l=int(l), s=int(runs), seed=int(seed))

with st.expander("So funktioniert die Messung", expanded=True):
    st.markdown(
        """
1. **Spielplan.** Die Zirkelmethode legt für n Plätze fest, wer an welchem Spieltag gegen wen spielt (n − 1 Spieltage). **Zuordnung** heißt: welche Mannschaft auf welchem Platz steht.
   Jede Zuordnung ist ein eigener Spielplan mit denselben Paarungen in anderer Reihenfolge. M1 ist die stärkste Mannschaft (Elo-Leiter).
2. **Partie.** Aus den aktuellen Elo-Werten folgt die Erwartung; ein Teil davon sind Remis (3 / 1 / 0 Punkte). Die Stärke einer Mannschaft ist **Grundstärke + m · Zustand**.
3. **Form als Zustand.** Jede Mannschaft ist kalt (−1), normal (0) oder heiß (+1). Nach jeder Partie bleibt der Zustand mit Wahrscheinlichkeit ρ, sonst wird er auf das Ergebnis gesetzt.
   **Konstruktion:** der Zustand startet in seiner **stationären Verteilung**, die nicht von ρ abhängt. Das Gedächtnis ändert dadurch vor allem, *wann* der Zustand wechselt, und kaum, *wie stark* er streut (bei m = 0 über alle Zuordnungen gemittelt gleich, sonst fast; Experiment 2 misst es). Die Spanne der erwarteten Punkte fällt trotzdem mit ρ.
4. **Messgröße.** Für jede Zuordnung die **erwarteten Punkte** jeder Mannschaft; die **Spanne** ist das Maximum minus das Minimum über die Zuordnungen. Die Simulation benutzt
   **gemeinsame Zufallszahlen** (eine je Paar, Hin- und Rückspiel), die Erwartung wird bis n = 8 zusätzlich **exakt** gerechnet (alle Zustände aller Mannschaften Partie für Partie).
        """
    )
    st.markdown("**Beispiel: Zuordnung M1 auf Platz 1, M2 auf Platz 2, …**")
    st.dataframe(pd.DataFrame(schedule_rows(tuple(range(settings.n)), settings.double)), hide_index=True, width="stretch")

with st.expander("🧭 Was die Auswahlen bedeuten"):
    st.markdown(
        """
| Auswahl | Bedeutung |
|---|---|
| **Einfache Runde** | Jedes Paar spielt einmal: n − 1 Spieltage. |
| **Doppelrunde** | Hin- und Rückrunde, die Rückrunde in derselben Reihenfolge wie die Hinrunde (2·(n − 1) Spieltage); Hin- und Rückspiel haben eigene Zufallszahlen. |
| **Start stationär** | Jede Mannschaft beginnt in der Zustandsverteilung, die der Zustandskette entspricht, wenn ihre Ergebnisse gegen ein zufälliges anderes Team fallen (unabhängig von ρ). Der Zustand streut dann für jedes ρ fast gleich stark (bei m = 0 über alle Zuordnungen gemittelt gleich, sonst misst Experiment 2 den Abstand). |
| **Start im Nullzustand** | Alle beginnen normal. Bei großem ρ dauert es lange, bis Form entsteht; die Streuung des Zustands wächst erst im Lauf und hängt am ρ. Das ist die naive Wahl; sie vergleicht Gedächtnisstufen mit verschieden starker Form. |
| **Gemeinsame Zufallszahlen** | Dieselbe Zufallszahl entscheidet die Partie eines Paars in jedem Spielplan. Dann unterscheiden sich zwei Spielpläne nur über die Reihenfolge, nicht über Würfelglück. |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

with st.spinner("Rechne die Spielpläne (bis zu etwa 20 Sekunden bei vielen Zuordnungen und Läufen) …"):
    a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
way = "exakt" if a.exact_points is not None else "Simulation"
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Spanne des Stärksten ({way})", f"{a.top_span:.2f} Punkte", delta=f"von höchstens {settings.max_points}", delta_color="off")
m2.metric("Größte Spanne irgendeiner Mannschaft", f"{a.max_span:.2f} Punkte", delta=f"{team_name(int(team_spans(a.main_points).argmax()))}", delta_color="off")
m3.metric("Erwartete Punkte des Stärksten (Mittel)", f"{a.main_points[:, 0].mean():.2f}", delta=f"{len(a.perms)} Zuordnungen", delta_color="off")
m4.metric("Zustandsstreuung (Var s)", f"{(a.exact_state_var if a.exact_state_var is not None else a.sim_state_var):.3f}", delta="mittlere Varianz des Zustands", delta_color="off")
if a.exact_points is not None:
    st.caption(f"Simulation mit gemeinsamen Zufallszahlen ({settings.s} Läufe je Plan): Spanne des Stärksten {a.sim_top_span:.2f} Punkte; sie enthält Rauschen der Simulation, die exakte Zahl nicht.")
st.plotly_chart(build_bands(a), width="stretch", key="bands_chart")
st.caption("Je Balken: kleinster bis größter Wert der erwarteten Punkten dieser Mannschaft über die Zuordnungen, als Abweichung von ihrem Mittel (Punkte). Die Höhe ist die Spanne.")
st.plotly_chart(build_hist(a), width="stretch", key="hist_chart")

best, worst = extreme_plans(a.main_points, a.perms)
st.markdown("**Gegnerfolge des Stärksten im besten und im schlechtesten Spielplan** (Mannschaftsnummer des Gegners je Spieltag)")
order_best = opponent_order(a.perms[best], 0, settings.double)
order_worst = opponent_order(a.perms[worst], 0, settings.double)
st.dataframe(
    pd.DataFrame({
        "Spieltag": range(1, len(order_best) + 1),
        f"Bester Plan ({a.main_points[best, 0]:.2f} Punkte)": [team_name(j) for j in order_best],
        f"Schlechtester Plan ({a.main_points[worst, 0]:.2f} Punkte)": [team_name(j) for j in order_worst],
    }),
    hide_index=True, width="stretch",
)
st.caption("Die Tabelle zeigt nur, wie die beiden Gegnerfolgen aussehen. Warum eine Folge besser ist, untersucht die Demo nicht; bei m = 0 sind beide gleich.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie viele erwartete Punkte verschiebt die Reihenfolge allein?")
kind, text = verdict_core(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Der Haken bei m = 0")
st.caption(
    f"Spanne des Stärksten gegen die Wirkung m, mit den übrigen Einstellungen der Seitenleiste (ρ, Liga, Start). Median über {len(C.SWEEP_SEEDS)} Messreihen (feste Zufalls-Seeds) "
    f"mit je {C.SWEEP_L} Zuordnungen und {C.SWEEP_S} Läufen. Die Rechnung dauert bis etwa 30 Sekunden; sie läuft auf Knopfdruck."
)
key1 = ("m", settings.n, settings.span, settings.draw, settings.rounds, settings.rho, settings.start)
if st.button("▶ Experiment 1 rechnen", key="run_exp1"):
    with st.spinner("Rechne Experiment 1 (bis zu etwa 30 Sekunden) …"):
        st.session_state["exp1"] = (key1, _sweep_m(settings))
exp1 = st.session_state.get("exp1")
if exp1 and exp1[0] == key1:
    rows1 = exp1[1]
    st.plotly_chart(build_m_sweep(rows1), width="stretch", key="m_sweep_chart")
    kind, text = verdict_m(rows1, settings.exact_possible)
    getattr(st, kind)(text)
else:
    st.info("Noch nicht gerechnet für diese Einstellungen: Knopf drücken.")

st.subheader("🔬 Experiment 2: Gedächtnis ändert den Zeitpunkt der Wechsel, kaum die Streuung des Zustands")
st.caption(
    f"Spanne des Stärksten und mittlere Zustandsstreuung gegen ρ, stationärer Start gegen Start im Nullzustand, mit m aus der Seitenleiste. Median über {len(C.SWEEP_SEEDS)} Messreihen "
    f"mit je {C.SWEEP_L} Zuordnungen. Die Rechnung dauert bis etwa 30 Sekunden; sie läuft auf Knopfdruck."
)
key2 = ("rho", settings.n, settings.span, settings.draw, settings.rounds, settings.m)
if st.button("▶ Experiment 2 rechnen", key="run_exp2"):
    with st.spinner("Rechne Experiment 2 (bis zu etwa 30 Sekunden) …"):
        st.session_state["exp2"] = (key2, _sweep_rho(settings))
exp2 = st.session_state.get("exp2")
if exp2 and exp2[0] == key2:
    rows2 = exp2[1]
    st.plotly_chart(build_rho_sweep(rows2), width="stretch", key="rho_sweep_chart")
    kind, text = verdict_rho(rows2)
    getattr(st, kind)(text)
else:
    st.info("Noch nicht gerechnet für diese Einstellungen: Knopf drücken.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Form ist ein Markow-Zustand mit drei Stufen** und einer festen Wirkung m | Ein Modell, an Daten nicht geprüft: die gezeigten Spannen gelten für diese Annahme. Ob echte Ligen Form dieser Art haben, ist offen. | Hot-Hand-Messfehler („Wie misst man Abhängigkeit?“) |
| **Stationärer Start im Mittelfeld-Sinn** (Gegner unabhängig gezogen) | Die Zustandsstreuung ist für alle ρ gleich nur näherungsweise; bei m = 0 gilt es exakt (Mittel über alle Zuordnungen). Siehe Experiment 2. | – |
| **Kein Heimvorteil, keine Breaks** | Heim und Auswärts fehlen. Wo ein Heimvorteil gilt, hat die Reihenfolge der Heimspiele eine eigene Wirkung; die Demo misst sie nicht. | [double-round-robin-demo](https://sebastianhanisch-double-round-robin-demo.streamlit.app/) (Breaks), [tore-remis-demo](https://sebastianhanisch-tore-remis-demo.streamlit.app/) (Heimvorteil schätzen) |
| **Keine Ruhetage, keine Ermüdung** | Nicht modelliert. Der Suchausschnitt einer Studie (PubMed 23851296, nicht im Volltext gelesen) nennt kaum Wirkung auf die Leistung der Mannschaften, eher auf Verletzungen. | – |
| **Feste Stärken auf einer gleichmäßigen Elo-Leiter** | Keine Stärkeänderung im Lauf, keine ungleichen Abstände; die Spannen hängen vom Elo-Abstand ab (Regler). | – |
| **Zufällige Zuordnungen** (Stichprobe aller n! Zuordnungen) | Die Spanne über eine Stichprobe ist höchstens die Spanne über alle Zuordnungen, nicht größer, und sie kann bei wenigen Zuordnungen deutlich kleiner sein (Standardfall, Stärkster: 0,43 bei 100 Zuordnungen gegen 0,59 bei allen 40.320). | – |
| **Rauschen gegen Verzerrung** | Ob eine Spanne im Vergleich zu den Zufallsschwankungen einer Saison ins Gewicht fällt, zeigt nur der Vergleich mit der Standardabweichung der Saisonpunkte im Kern-Verdikt. | Zusammenfluss „Rauschen oder Verzerrung?“ |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Mannschaften mit Grundstärken $R_i$ (Elo), eine Zuordnung $\pi$ der Mannschaften zu den Plätzen der Zirkeltafel und ein Zustandsmodell. Gesucht ist die Spanne
der erwarteten Punkte einer Mannschaft über die Zuordnungen.

| Symbol | Bedeutung |
|---|---|
| $R_i$ | Grundstärke der Mannschaft $i$ in Elo (M1 = Mannschaft 0 ist die stärkste) |
| $s_i(t) \in \{-1, 0, +1\}$ | Zustand vor dem Spieltag $t$: kalt, normal, heiß |
| $m$ | Wirkung der Form in Elo je Zustandsstufe |
| $\rho \in [0, 1)$ | Gedächtnis: Wahrscheinlichkeit, dass der Zustand bleibt |
| $\delta$ | Höchstwert des Remisanteils |
| $e_{ij}$, $d_{ij}$ | Erwartung und Remisanteil der Partie $i$ gegen $j$ |
| $E_i(\pi)$ | erwartete Punkte der Mannschaft $i$ bei Zuordnung $\pi$ |
| $\Delta_i$ | Spanne von $E_i$ über die Zuordnungen |
| $\pi_i^\ast$ | stationäre Zustandsverteilung der Mannschaft $i$ |

**Partie.** Mit den Stärken $r_i = R_i + m\,s_i$ gilt

$$e_{ij} = \frac{1}{1 + 10^{(r_j - r_i)/400}}$$

$$d_{ij} = \min\bigl(\delta,\; 2\min(e_{ij},\, 1 - e_{ij})\bigr)$$

Sieg von $i$ mit Wahrscheinlichkeit $e_{ij} - d_{ij}/2$, Remis mit $d_{ij}$, Niederlage mit $1 - e_{ij} - d_{ij}/2$. Punkte 3 / 1 / 0; der erwartete Punktanteil bleibt $e_{ij}$.

**Zustand mit Gedächtnis.** Nach der Partie mit Ergebnis $x_i \in \{-1, 0, +1\}$ (Sieg, Remis, Niederlage) gilt, mit einer eigenen Münze je Mannschaft,

$$s_i(t+1) = \begin{cases} s_i(t) & \text{mit Wahrscheinlichkeit } \rho \\ x_i(t) & \text{mit Wahrscheinlichkeit } 1 - \rho \end{cases}$$

**Stationäre Verteilung.** Mit $Q_i[s, x]$, der Wahrscheinlichkeit des Ergebnisses $x$ bei Zustand $s$ gegen einen gleichverteilt gezogenen Gegner mit eigener stationärer Verteilung, ist die Kette $P_i = \rho I + (1 - \rho) Q_i$.

$$\pi_i^\ast P_i = \pi_i^\ast \;\Longleftrightarrow\; \pi_i^\ast Q_i = \pi_i^\ast$$

Die stationäre Verteilung hängt also nicht von $\rho$ ab (Fixpunkt über alle Mannschaften). Der Start zieht $s_i(1) \sim \pi_i^\ast$; das Gedächtnis ändert dann nur, wann der Zustand wechselt.

**Messgröße.** Für die Zuordnung $\pi$ sind $E_i(\pi)$ die erwarteten Punkte der Mannschaft $i$ über alle Partien der Saison, und

$$\Delta_i = \max_\pi E_i(\pi) - \min_\pi E_i(\pi)$$

**Der Haken.** Bei $m = 0$ hängt die Partie $i$ gegen $j$ nur von $R_i$ und $R_j$ ab, nicht vom Zustand. Dann ist $E_i(\pi) = \sum_{j \ne i} (3\,p^{\text{Sieg}}_{ij} + p^{\text{Remis}}_{ij})$ (bei Doppelrunde doppelt) für jede Zuordnung gleich, also

$$\Delta_i = 0 \quad \text{für alle } i.$$

**Exakte Erwartung.** Die gemeinsame Verteilung aller Zustände $(s_1, \dots, s_n)$ hat $3^n$ Einträge. Jede Partie wirkt als Übergang auf zwei Achsen; die erwarteten Punkte einer Partie folgen aus der Randverteilung
der beiden Mannschaften vor der Partie. Die Simulation benutzt dieselben Regeln mit gemeinsamen Zufallszahlen (SplitMix64).

**Literatur.** Nicht im Volltext gelesen, nur als Hinweis: zu Form und Hot Hand Miller und Sanjurjo (2018); Studie zu Ruhetagen und Verletzungen (PubMed 23851296, nach Suchausschnitt).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Wettbewerbsverzerrung: Reihenfolge und Anreize](https://sebastianhanisch.net/konzepte-wettbewerbsverzerrung.html)."
)
