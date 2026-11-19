"""Plotly-Abbildungen: alle Achsen fest (fixedrange), damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import spf_constants as C
from spf_evaluation import Analysis, team_name, team_spans


def _lock(fig, height=380):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=40, b=50), legend=dict(orientation="h", y=-0.25))
    return fig


def build_bands(a: Analysis):
    """Je Mannschaft der Bereich der erwarteten Punkte über alle Zuordnungen, als Abweichung vom Mittel der Mannschaft (Punkte)."""
    P = a.main_points
    mean = P.mean(axis=0)
    lo = P.min(axis=0) - mean
    hi = P.max(axis=0) - mean
    names = [team_name(i) for i in range(P.shape[1])]
    height = team_spans(P)                                  # Rundungsreste der exakten Rechnung zählen als 0
    lo = np.where(height == 0.0, 0.0, lo)
    fig = go.Figure(go.Bar(x=names, y=height, base=lo, marker_color=C.COLOR_MAIN, name="Bereich über die Zuordnungen",
                           hovertemplate="%{x}: von %{base:+.3f} bis %{customdata:+.3f}<extra></extra>", customdata=hi))
    fig.add_hline(y=0, line_color=C.COLOR_GREY)
    lim = max(0.25, float(np.ceil(max(abs(lo).max(), abs(hi).max()) * 10) / 10) + 0.1)
    fig.update_yaxes(title="Erwartete Punkte minus Mittel der Mannschaft", range=[-lim, lim])
    fig.update_xaxes(title="Mannschaft (M1 stärkste)")
    fig.update_layout(title="Wie weit verschiebt die Zuordnung die erwarteten Punkte?")
    return _lock(fig)


def build_hist(a: Analysis):
    """Verteilung der erwarteten Punkte des Stärksten über die Zuordnungen."""
    col = a.main_points[:, 0]
    fig = go.Figure(go.Histogram(x=col, nbinsx=20, marker_color=C.COLOR_MAIN, name="Zuordnungen"))
    fig.add_vline(x=float(col.mean()), line_color=C.COLOR_ALT, line_dash="dash", annotation_text="Mittel", annotation_position="top")
    centre = float(col.mean())
    half = max(0.05, float(np.ceil(max(abs(col - centre).max(), 0.02) * 20) / 20) + 0.05)
    fig.update_xaxes(title="Erwartete Punkte des Stärksten (M1)", range=[centre - half, centre + half])
    fig.update_yaxes(title="Zahl der Zuordnungen", rangemode="tozero")
    fig.update_layout(title="Verteilung für den Stärksten", showlegend=False)
    return _lock(fig, 320)


def build_m_sweep(rows):
    """Experiment 1: Spanne des Stärksten gegen die Wirkung m."""
    ms = [r["m"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ms, y=[r["exakt"] for r in rows], mode="lines+markers", name="Erwartung (exakt bzw. Simulation bei n = 10)", line=dict(color=C.COLOR_MAIN)))
    fig.add_trace(go.Scatter(x=ms, y=[r["gemeinsam"] for r in rows], mode="lines+markers", name="Simulation, gemeinsame Zufallszahlen", line=dict(color=C.COLOR_LINE, dash="dot")))
    fig.add_trace(go.Scatter(x=ms, y=[r["naiv"] for r in rows], mode="lines+markers", name="Simulation, je Zuordnung eigene Zufallszahlen", line=dict(color=C.COLOR_ALT, dash="dash")))
    top = max(max(r["exakt"], r["gemeinsam"], r["naiv"]) for r in rows)
    fig.update_yaxes(title="Spanne des Stärksten (Punkte)", range=[0, max(0.5, top * 1.15)])
    fig.update_xaxes(title="Wirkung m der Form (Elo je Zustandsstufe)", tickvals=ms, range=[min(ms) - 5, max(ms) + 5])
    fig.update_layout(title="Spanne gegen Formwirkung")
    return _lock(fig, 400)


def build_rho_sweep(rows):
    """Experiment 2: Spanne und Zustandsstreuung gegen das Gedächtnis, stationärer gegen naiven Start."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Spanne des Stärksten (Punkte)", "Mittlere Zustandsstreuung"))
    for start, color, dash in (("stationär", C.COLOR_MAIN, "solid"), ("Nullzustand", C.COLOR_ALT, "dash")):
        sub = [r for r in rows if r["start"] == start]
        rhos = [r["rho"] for r in sub]
        fig.add_trace(go.Scatter(x=rhos, y=[r["top"] for r in sub], mode="lines+markers", name=f"Start {start}", line=dict(color=color, dash=dash),
                                 legendgroup=start), row=1, col=1)
        fig.add_trace(go.Scatter(x=rhos, y=[r["var"] for r in sub], mode="lines+markers", name=f"Start {start}", line=dict(color=color, dash=dash),
                                 legendgroup=start, showlegend=False), row=1, col=2)
    top = max(r["top"] for r in rows)
    fig.update_yaxes(range=[0, max(0.5, top * 1.15)], row=1, col=1)
    fig.update_yaxes(range=[0, 1.0], row=1, col=2)
    fig.update_xaxes(title="Gedächtnis ρ", range=[-0.05, 1.0])
    return _lock(fig, 400)
