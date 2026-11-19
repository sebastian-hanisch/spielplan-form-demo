"""End-to-end Smoke-Tests mit Streamlits AppTest: Standardlauf, jeder Preset-Knopf, jeder Regler an beiden Grenzen (Auswahlregler: jeder Eintrag),
beide Experimente, und dass die Anzeige wirklich die Regler der Seitenleiste benutzt."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import spf_constants as C
import spf_evaluation as ev
import spf_presets as P

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _widget(at, key):
    for art in ("slider", "select_slider", "radio", "number_input"):
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def _texts(at):
    return " ".join(x.value for x in list(at.success) + list(at.info) + list(at.warning))


def test_app_loads_without_exception():
    at = _fresh_app()
    assert any("Spanne des Stärksten" in m.label for m in at.metric)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_gives_a_verdict(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, (name, [str(e) for e in at.exception])
    assert "Punkte" in _texts(at)


def test_seed_button_changes_the_seed_and_does_not_raise():
    at = _fresh_app()
    old = at.sidebar.number_input(key="seed_input").value
    at.sidebar.button[-1].click().run(timeout=TIMEOUT)
    assert not at.exception
    new = at.sidebar.number_input(key="seed_input").value
    assert 0 <= new <= C.SEED_MAX and new != old        # Zufallsziehung: Gleichheit hat die Wahrscheinlichkeit 1 zu 1 Million


@pytest.mark.parametrize("state_key", [k for k, s in P.SETTING_SPECS.items() if s.choices is None and s.hi is not None])
def test_every_slider_at_its_min_and_max_does_not_raise(state_key):
    spec = P.SETTING_SPECS[state_key]
    at = _fresh_app()                                   # eine App, beide Grenzen nacheinander (spart je einen Neustart)
    for edge in (spec.lo, spec.hi):
        _widget(at, state_key).set_value(edge).run(timeout=TIMEOUT)
        assert not at.exception, (state_key, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("state_key", [k for k, s in P.SETTING_SPECS.items() if s.choices is not None])
def test_every_choice_runs(state_key):
    at = _fresh_app()
    for choice in P.SETTING_SPECS[state_key].choices:
        _widget(at, state_key).set_value(choice).run(timeout=TIMEOUT)
        assert not at.exception, (state_key, choice, [str(e) for e in at.exception])


def test_display_uses_the_values_of_the_sidebar():
    """Schlüsselwörter statt Reihenfolge (Settings nie positional): die angezeigte Spanne gleich der Rechnung mit denselben Reglerwerten."""
    at = _fresh_app()
    _widget(at, "n_select").set_value(4)
    _widget(at, "m_slider").set_value(100)
    _widget(at, "rho_slider").set_value(0.2)
    at.run(timeout=TIMEOUT)
    assert not at.exception
    a = ev.analyse(ev.Settings(n=4, m=100, rho=0.2))
    shown = next(m.value for m in at.metric if m.label.startswith("Spanne des Stärksten"))
    assert shown == f"{a.top_span:.2f} Punkte"
    assert a.top_span > 0


def test_without_form_the_app_shows_zero_and_the_haken_verdict():
    at = _fresh_app()
    at.sidebar.button(key="preset_Ohne Form (m = 0): der Haken").click().run(timeout=TIMEOUT)
    assert next(m.value for m in at.metric if m.label.startswith("Spanne des Stärksten")) == "0.00 Punkte"
    assert "Ohne Form (m = 0) ist die erwartete Punktzahl jeder Mannschaft in **jedem** Spielplan gleich" in _texts(at)


def test_experiments_ask_for_a_click_and_show_a_verdict_afterwards_for_four_teams():
    at = _fresh_app()
    assert "Noch nicht gerechnet" in _texts(at)
    _widget(at, "n_select").set_value(4)
    at.run(timeout=TIMEOUT)
    at.button(key="run_exp1").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "reines Rauschen" in _texts(at)
    at.button(key="run_exp2").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "Zustandsstreuung" in _texts(at)
    _widget(at, "m_slider").set_value(0).run(timeout=TIMEOUT)                    # geänderte Einstellung: Ergebnis von Experiment 2 gilt nicht mehr
    assert "Noch nicht gerechnet" in _texts(at)


def test_experiment_two_runs_for_ten_teams_without_the_exact_branch():
    at = _fresh_app()
    _widget(at, "n_select").set_value(10)
    at.run(timeout=TIMEOUT)
    at.button(key="run_exp2").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "Zustandsstreuung" in _texts(at)


def test_experiment_one_runs_for_the_default_league():
    at = _fresh_app()
    at.button(key="run_exp1").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "reines Rauschen" in _texts(at)
