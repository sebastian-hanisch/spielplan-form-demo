"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Begrenzung."""

import spf_constants as C
import spf_evaluation as ev
import spf_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            assert P.clamp(spec, preset[key]) == preset[key], (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert ev.Settings(**C.PRESETS[first]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_clamp_limits_numbers_and_replaces_invalid_choices():
    assert P.clamp(P.SETTING_SPECS["m_slider"], 999) == C.M_MAX
    assert P.clamp(P.SETTING_SPECS["m_slider"], -5) == C.M_MIN
    assert P.clamp(P.SETTING_SPECS["rho_slider"], 1.0) == C.RHO_MAX          # rho = 1 ist nie erreichbar
    assert P.clamp(P.SETTING_SPECS["n_select"], 7) == C.DEFAULT_N            # ungerade n gibt es nicht
    assert P.clamp(P.SETTING_SPECS["n_select"], 10) == 10
    assert P.clamp(P.SETTING_SPECS["start_radio"], "Unsinn") == C.DEFAULT_START


def test_preset_names_are_distinct_in_effect():
    """Kein Preset darf ein Doppelgänger eines anderen sein (jedes unterscheidet sich in mindestens einem Regler)."""
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_small_league_preset_enumerates_all_assignments():
    p = C.PRESETS["Kleine Liga (n = 4, alle 24 Zuordnungen)"]
    import math
    assert math.factorial(p["n"]) <= p["l"] == 100
