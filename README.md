# Spielplanreihenfolge und Form: was die Reihenfolge der Gegner verschiebt – Streamlit-Demo

<!-- Live-Link nach dem Deploy hier einsetzen (Leerzeile davor und danach): sebastianhanisch-spielplan-form-demo.streamlit.app -->

Konzepte-Demo der Linie **„Wettbewerbsverzerrung erklärt“** (Bereich Wertung und Wahl). Das Vehikel ist eine **simulierte Liga** mit bekannter Stärke je Mannschaft (Elo-Leiter), gespielt nach der
**Zirkelmethode** (Berger-Tafel). Die [Doppelrunde mit minimalen Breaks](https://sebastianhanisch-double-round-robin-demo.streamlit.app/) konstruiert solche Pläne, ohne ein Ergebnis zu bewerten;
diese Demo bewertet die **Reihenfolge der Gegner**. Alles ist simuliert, es gibt keine Felddaten und keine realen Vereine.

## Ergebnis in Kürze

Messung: 8 Mannschaften, Elo-Abstand 300 (Stärkster gegen Schwächsten), Remisanteil höchstens 0,25, einfache Runde, 3/1/0 Punkte, **100 zufällige von 40 320 Zuordnungen** der Mannschaften zur Zirkeltafel
(Zufalls-Seed 2026), erwartete Punkte **exakt** gerechnet; „Spanne“ heißt größter minus kleinster Wert der erwarteten Punkte einer Mannschaft über diese Zuordnungen.

- **Ohne Form (m = 0) ist die Spanne in diesem Modell 0** (Gleitkomma-Reste unter 1e-9 zählen als 0; gemessen höchstens rund 1e-14), für jede Mannschaft, geprüft für n = 4, 6, 8 (einfache Runde) und n = 4, 8 (Doppelrunde). Die Reihenfolge der Gegner verschiebt dann in diesem Modell keine erwartete Punktzahl.
- **Mit Form** (Wirkung m = 50 Elo, Gedächtnis ρ = 0,5) verschiebt sie die erwarteten Punkte des Stärksten über die 100 zufälligen Zuordnungen um **0,43 Punkte** (2,1 % der höchstens 21; 13 % der Standardabweichung 3,24 seiner Saisonpunkte).
  Das ist ein Stichprobenwert (Zufalls-Seed 2026) und eine Untergrenze: über alle 40.320 Zuordnungen sind es **0,59 Punkte** (2,8 % der höchstens 21). Die größte Spanne haben die mittleren Mannschaften (M4 und M5 mit je 0,89 über alle Zuordnungen;
  in der Stichprobe M4 mit 0,80), nicht der Stärkste.
- **Die Spanne wächst mit m und fällt mit dem Gedächtnis ρ** (Spannen aus den 100 Zuordnungen, also Untergrenzen): Stärkster, ρ = 0: 0,36 / 0,76 / 1,69 Punkte bei m = 25 / 50 / 100; ρ = 0,8: 0,09 / 0,19 / 0,40; ρ = 0,95: 0,03 / 0,05 / 0,11.
- **Das Gedächtnis ändert den Zeitpunkt der Zustandswechsel, kaum die Streuung des Zustands:** mit stationärem Start liegt die Zustandsstreuung für ρ = 0 bis 0,95 zwischen 0,650 und 0,653 (m = 50), mit Start in 0 zwischen 0,10 und 0,57.
  Die Spanne fällt trotzdem mit ρ (0,76 auf 0,05): die Korrektur gegenüber der Vormessung steht unten.
- **Der Haken der Messung:** eine Simulation mit je Spielplan eigenen Zufallszahlen zeigt bei m = 0 eine „Spanne“ des Stärksten von rund 0,47 Punkten (Median über drei Messreihen mit je 40 Zuordnungen und 1000 Läufen; Rauschen, die Zahl hängt an Läufen und Zuordnungen); mit gemeinsamen Zufallszahlen ist sie 0.

## Kernfrage

Wie viele erwartete Punkte kann allein die **Reihenfolge der Gegner** (die Zuordnung der Mannschaften zu den Plätzen der Zirkeltafel) verschieben? Die Antwort hat zwei Teile. Ohne einen Einfluss der
bisherigen Ergebnisse auf die nächste Partie: keine; in diesem Modell lässt sich das beweisen (jede Partie hängt nur von den Grundstärken ab, die Summe über die feste Gegnermenge nicht von der Reihenfolge). Mit einem solchen Einfluss („Form“): eine kleine, aber von null verschiedene Zahl, die vom Modell der Form abhängt.
Die Demo beantwortet damit auch, wozu das **Planen der Reihenfolge** (statt eines beliebigen Plans) überhaupt etwas ändern kann: in diesem Modell nur über Form. Heim und Auswärts und damit die **Breaks** selbst sind **nicht**
modelliert (siehe Abgrenzung und Grenzen).

## Modell

- **Liga:** n Mannschaften (n = 4, 6, 8, 10), Stärken gleichmäßig auf einer Elo-Leiter von +Abstand/2 bis −Abstand/2 (Abstand 0 bis 600, Standard 300). M1 ist die stärkste Mannschaft.
- **Partie:** Elo-Erwartung e = 1/(1 + 10^((r_b − r_a)/400)); Remisanteil d = min(Höchstwert, 2·min(e, 1 − e)); Sieg e − d/2, Niederlage 1 − e − d/2. Der erwartete Punktanteil bleibt e. Punkte 3 / 1 / 0.
- **Form (Markow-Zustand mit Gedächtnis):** Zustand s ∈ {−1, 0, +1} (kalt, normal, heiß), Stärke = Grundstärke + m·s. Nach jeder Partie bleibt der Zustand mit Wahrscheinlichkeit ρ, sonst wird er auf das
  Ergebnis gesetzt (Sieg +1, Remis 0, Niederlage −1); jede Mannschaft wirft ihre eigene Münze. ρ = 0 mit Start in 0 ist das einfache ±m-Modell. ρ ist höchstens 0,95 (bei 1 wechselt der Zustand nie).
- **Konstruktion gegen Äpfel mit Birnen:** Gedächtnis soll den Zeitpunkt der Wechsel ändern, nicht die Streuung des Zustands (die Spanne ändert es trotzdem, siehe Experiment 2). Darum startet der Zustand jeder Mannschaft in der **stationären Verteilung** ihrer
  Zustandskette. Die Kette ist P = ρ·I + (1 − ρ)·Q, mit Q der Ergebnisverteilung je Zustand gegen einen gleichverteilt gezogenen Gegner mit eigener stationärer Verteilung (Fixpunkt über alle Mannschaften).
  Es gilt πP = π genau dann, wenn πQ = π: **die stationäre Verteilung hängt nicht an ρ**. Eine Nachführung von m ist deshalb nicht nötig. Bei m = 0 ist die Konstruktion exakt (über alle Zuordnungen
  gemittelt hat jede Mannschaft in jedem Spieltag genau die Startverteilung; nachgerechnet für n = 4 mit allen 24 Zuordnungen und ρ = 0, 0,5, 0,9), bei m > 0 gilt sie im Mittelfeld-Sinn näherungsweise (Gegner nicht unabhängig): der Abstand der Zustandsstreuung über ρ ist gemessen.
  Wahlweise startet die App im Nullzustand (naiv), um den Unterschied zu zeigen.
- **Spielplan:** Zirkelmethode für gerades n (n − 1 Spieltage); Platz 0 bleibt, die übrigen rücken. Eine **Zuordnung** setzt Mannschaften auf Plätze; jede Zuordnung ist derselbe Satz Paarungen in anderer
  Reihenfolge. Doppelrunde: die Rückrunde wiederholt die Hinrunde in gleicher Reihenfolge, Hin- und Rückspiel mit eigenen Zufallszahlen.
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Zwei Rechenwege.** (1) **Simulation** mit **gemeinsamen Zufallszahlen**: eine Zahl je Paar (und Hin-/Rückspiel), je Mannschaft und Spieltag (Münze) und je Mannschaft für den Start, dieselben für alle
  Zuordnungen; die Ausrichtung je Paar ist fest (kleinere Mannschaftsnummer zuerst), sonst entstünde ein Artefakt (dieselbe Zahl, andere Auswertungsrichtung). Dadurch ist die Spanne bei m = 0 gleich 0 (bis auf Gleitkomma-Reste), nicht nur klein.
  (2) **Exakte Erwartung** bis n = 8: die gemeinsame Verteilung aller n Zustände (3^n Einträge) wird Partie für Partie weitergerechnet; die erwarteten Punkte folgen aus der Randverteilung der beiden Mannschaften vor jeder Partie.
- **Zufall:** SplitMix64 (`spf_rng.py`, wörtliche Kopie der geprüften Vorlage, plattformfest), vektorisiert zu Blöcken (`uniform_block`, bitgleich zur Schleife). Gleiche Reglerwerte und gleicher Zufalls-Seed ergeben dasselbe Ergebnis.
- **Zuordnungen:** alle n! (n = 4: 24), sonst die gewählte Zahl zufälliger (mit Zurücklegen). Die Spanne über eine Stichprobe ist höchstens die Spanne über alle Zuordnungen.
- **Experimente** auf Knopfdruck: je drei feste Messreihen (Zufalls-Seeds 2026, 2027, 2028) mit je 40 Zuordnungen, Median; Experiment 1 über m, Experiment 2 über ρ (stationär gegen Nullzustand).

## Befunde (gemessen, keine Behauptungen)

Voraussetzungen wie im Abschnitt „Ergebnis in Kürze“, falls nicht anders genannt. Exakt heißt: Tensor-Rechnung ohne Zufall; Zahlen mit Messreihen sind Mediane über drei Messreihen mit je 40 Zuordnungen.

| Frage | Befund | Test |
|---|---|---|
| Spanne ohne Form (m = 0)? | **0** für jede Mannschaft (Gleitkomma-Reste unter 1e-9 zählen als 0): n = 4 (alle 24 Zuordnungen, einfache Runde und Doppelrunde), n = 6 (einfache Runde) und n = 8 (einfache Runde und Doppelrunde) mit je 60 Zuordnungen, Gedächtnis ρ = 0,5; in der Simulation mit gemeinsamen Zufallszahlen ebenfalls 0 (n = 8 und n = 10). | `test_without_form_every_team_has_zero_spread_exactly`, `test_without_form_simulation_with_common_numbers_has_zero_spread_for_ten_teams_too`, `test_without_form_the_spread_is_zero_in_every_random_league` |
| Was zeigt eine naive Messung bei m = 0? | Mit je Zuordnung eigenen Zufallszahlen (1000 Läufe) eine „Spanne“ des Stärksten von **0,47 Punkten** (Median, 40 Zuordnungen): Rauschen, kein Plan-Glück; die Zahl hängt an der Zahl der Läufe und der Zuordnungen. | `test_naive_independent_numbers_show_spread_without_form` |
| Wie groß ist die Spanne im Standardfall (m = 50, ρ = 0,5)? | **100 zufällige Zuordnungen** (Untergrenze): Stärkster **0,43**, größte Spanne einer Mannschaft **0,80** (M4); M1 0,43, M8 0,42. 2,1 % der höchstens 21 Punkte, 13 % der Standardabweichung der Saisonpunkte des Stärksten (3,24). Mittel der erwarteten Punkte des Stärksten 14,63. | `test_default_spans` |
| Und über alle 40.320 Zuordnungen (Standardfall)? | Vollständige Aufzählung, exakte Erwartung: Stärkster **0,59** (2,8 % der höchstens 21); M2 0,74, M3 und M6 0,84, **M4 und M5 0,89**, M7 0,74, M8 0,58. Zwischen M4 und M5 liegt nur ein Gleitkomma-Rest (unter 1e-14). Die Spanne des Stärksten aus 100 zufälligen Zuordnungen schwankt mit dem Zufalls-Seed: über die 30 Zufalls-Seeds 2026 bis 2055 von 0,38 bis 0,57 (Mittel 0,47); die größte Spanne hatte dabei M4 in 18, M5 in 6, M3 und M6 in je 3 Fällen. | `test_full_enumeration_of_all_assignments_default_case` (Lauf von rund vier Minuten, nur mit `SPF_FULL=1`) |
| Wächst sie mit m und fällt sie mit ρ? | Stärkster, exakt gerechnet, Spannen aus 100 zufälligen Zuordnungen (Untergrenzen): ρ = 0: 0,36 / 0,76 / 1,69 (m = 25 / 50 / 100); ρ = 0,5: 0,20 / 0,43 / 0,97; ρ = 0,8: 0,09 / 0,19 / 0,40; ρ = 0,95: 0,03 / 0,05 / 0,11. In jeder der vier ρ-Zeilen wächst sie mit m. | `test_span_of_the_strongest_by_form_effect_and_memory`, `test_spread_grows_with_the_effect_of_form` |
| Bleibt die Zustandsstreuung über ρ konstant? | Mit **stationärem Start**: m = 50 zwischen 0,650 und 0,653 (Abstand 0,003), m = 100 zwischen 0,644 und 0,652 (Abstand 0,007); bei m = 0 für n = 4 über alle 24 Zuordnungen gleich (0,610 für jedes ρ, Abstand unter 1e-9), für n = 8 mit 100 Zuordnungen mit Abstand 0,0006. Mit **Start in 0** (m = 50): 0,574 / 0,491 / 0,310 / 0,102 für ρ = 0 / 0,5 / 0,8 / 0,95. | `test_state_variance_is_nearly_constant_in_rho_with_stationary_start`, `test_without_form_the_state_variance_does_not_depend_on_rho`, `test_state_variance_is_not_constant_with_start_in_zero`, `test_without_form_the_average_state_distribution_over_all_assignments_is_stationary_in_every_round` |
| Fällt die Spanne mit ρ auch bei konstanter Streuung? | **Ja.** Stärkster, m = 50, Messreihen: stationär 0,72 / 0,41 / 0,19 / 0,05 für ρ = 0 / 0,5 / 0,8 / 0,95; Start in 0: 0,72 / 0,37 / 0,14 / 0,03. Der naive Start unterschätzt die Spanne bei ρ = 0,5, 0,8 und 0,95. | `test_spread_still_falls_with_memory_even_with_constant_state_spread` |
| Experiment 1 (m)? | Stärkster, ρ = 0,5: m = 0 / 25 / 50 / 100: 0 / 0,19 / 0,41 / 0,93 (exakt). | `test_m_sweep_values` |
| Doppelrunde, Ligagröße, Elo-Abstand? | Doppelrunde (n = 8): 0,83 von höchstens 42 Punkten (2,0 %). n = 4, m = 100, alle 24 Zuordnungen: Stärkster 0,14, größte Spanne 0,41. n = 6: 0,28. n = 10 (nur Simulation, Rauschen inbegriffen): 0,73. Elo-Abstand 0 (Standardfall, Remisanteil höchstens 0,25): 0 (Gleitkomma-Rest unter 1e-9); Abstand 600: größte Spanne 1,04, Stärkster 0,39. Die n = 8-Zahlen sind Spannen aus 100 zufälligen Zuordnungen. | `test_double_round_and_league_size`, `test_elo_range_zero_and_six_hundred` |
| Hängt die Spanne an der Zahl der Zuordnungen? | Mit 300 statt 100 Zuordnungen: größte Spanne 0,82 statt 0,80, die des Stärksten bleibt 0,43. Das Gleichbleiben ist Zufall: die 300 Zuordnungen enthalten dieselben ersten 100, und die Spanne des Stärksten liegt über alle 40.320 bei 0,59. Über die 30 Zufalls-Seeds 2026 bis 2055 liegt sie bei 100 Zuordnungen im Mittel bei 0,47, bei 300 bei 0,51. | `test_more_assignments_contain_the_first_100_and_do_not_settle_the_span`, `test_full_enumeration_of_all_assignments_default_case` (nur mit `SPF_FULL=1`) |
| Wie genau ist die Simulation (2000 Läufe)? | Mittlere Abweichung von der exakten Erwartung 0,047 Punkte, größte 0,17 (100 Zuordnungen, 8 Mannschaften); simulierte Spanne des Stärksten 0,38 gegen exakt 0,43. | `test_simulation_against_exact_at_default` |
| Stimmt der Kern mit einem unabhängigen Rechenweg überein? | Ja: Aufzählung aller Ergebnisbäume für n = 4 (reines Python, 3^6 Ergebnisfolgen mit Münzwurf-Verzweigung) gegen Tensor-Rechnung (1e-9) und gegen die Simulation (80 000 Läufe, Band 0,055 Punkte); einfaches ±m-Modell als Sonderfall; stationäre Verteilung als Grenzwert der Matrixpotenz. | `tests/test_oracle_spielplan_form.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **„Die Spannen schrumpfen mit dem Gedächtnis nur, weil der Zustand bei Start in 0 langsamer in Gang kommt.“** Das stimmt größtenteils nicht. Mit stationärem Start bleibt die Zustandsstreuung für alle ρ
  praktisch gleich (0,650 bis 0,653), und die Spanne des Stärksten fällt trotzdem von 0,72 auf 0,05; der Start in 0 verstärkt den Abfall (0,72 auf 0,03) und verändert die Streuung stark (0,57 auf 0,10).
  Eine Deutung, nicht gemessen: bei großem Gedächtnis hängt der Zustand kaum davon ab, was im Plan passiert, deshalb unterscheiden sich die Pläne wenig.
- **Zahlen der Vormessung** (Stärkster, m = 50: 0,90 / 0,51 / 0,20 für ρ = 0 / 0,5 / 0,8, Simulation, Start in 0, 300 Zuordnungen) sind hier nicht reproduziert und nicht die README-Zahlen: der Bau rechnet exakt, mit 40 bzw. 100
  Zuordnungen und stationärem Start. Zahl der Zuordnungen und Simulationsrauschen wurden nicht getrennt untersucht.
- **„Der Stärkste hat die größte Spanne“** (Vormessung nannte die des Stärksten): im Standardfall hat in der Stichprobe von 100 Zuordnungen **M4** die größte (0,80), der Stärkste 0,43; über alle 40.320 Zuordnungen haben M4 und M5 die größte (0,89), der Stärkste 0,59. Dass nicht der Stärkste die größte hat, gilt in beiden; dass es M4 ist, nur in der Stichprobe.
- **Gleich starke Mannschaften (Elo-Abstand 0) haben im Standardfall auch mit Form die Spanne 0** (Gleitkomma-Rest unter 1e-9, gemessen rund 1e-14; n = 8, m = 50, ρ = 0,5, Remisanteil höchstens 0,25): sie sind austauschbar. Ein früherer Entwurf hielt die Spanne für klein, aber nicht null, mit der Begründung ungleichwertiger Plätze der Zirkeltafel; diese Begründung ist nicht belegt.
  Eine winzige Spanne bleibt nur bei hohem Remisanteil und großem m: bei Remisanteil höchstens 0,5, ρ = 0,5, n = 6 und allen 720 Zuordnungen 0,0002 bei m = 100 und 0,003 bei m = 150; bei Remisanteil höchstens 0,3 und m = 150 ist sie 0. Eine Deutung, nicht gemessen: dort bindet die Obergrenze des Remisanteils, und der erwartete Punktanteil ist im Zustand nicht mehr linear.
- **Nachführen von m ist nicht nötig:** die stationäre Verteilung hängt nicht an ρ; bei m > 0 bleibt ein kleiner Abstand der Zustandsstreuung (0,003 bei m = 50, 0,007 bei m = 100), weil Gegner nicht unabhängig sind.
- **Titel:** der Plan nannte als Titel „wozu die Breaks-Minimierung gut ist“. Die Demo modelliert Heim und Auswärts nicht; was sie belegt, ist die Wirkung der **Reihenfolge der Gegner**. Der Titel wurde entsprechend geändert.

## Abgrenzung

- [Doppelrunde mit minimalen Breaks](https://sebastianhanisch-double-round-robin-demo.streamlit.app/) konstruiert den Plan und minimiert Breaks, bewertet kein Ergebnis. Hier wird ein Ergebnismodell auf viele Pläne gelegt; Breaks
  (Heim/Auswärts-Folgen) fehlen.
- [Rundenturnier](https://sebastianhanisch-round-robin-demo.streamlit.app/) baut die Berger-Tafel; diese Demo hat eine eigene kleine Umsetzung (`circle_rounds`) und importiert nichts von dort.
- [Tore und Remis](https://sebastianhanisch-tore-remis-demo.streamlit.app/) schätzt den Heimvorteil aus Ergebnissen; hier gibt es keinen Heimvorteil und keinen Zustand je Mannschaft.
- [Tabelle nach Siegen](https://sebastianhanisch-siegtabelle-demo.streamlit.app/) misst, wie gut die Tabelle die Stärke trifft (Rauschen eines festen Plans); hier wird derselbe Plan in vielen Reihenfolgen verglichen.

## Ehrliche Grenzen

- **Form ist ein Modell, keine Messung.** Zustand mit drei Stufen, feste Wirkung m, Gedächtnis ρ: die Spannen gelten für diese Annahmen; ob echte Ligen Form dieser Art haben, ist hier nicht untersucht.
- **Stationär im Mittelfeld-Sinn:** exakt nur bei m = 0 (gemittelt über alle Zuordnungen); bei m > 0 näherungsweise (Abstand der Zustandsstreuung 0,003 bis 0,007 gegen 0,47 beim naiven Start).
- **Keine Heimvorteile, keine Breaks, keine Ruhetage oder Ermüdung.** Zu Ruhetagen nennt der Suchausschnitt einer Studie (PubMed 23851296, nicht im Volltext gelesen) kaum Wirkung auf die Leistung der Mannschaften, eher auf Verletzungen; die Demo
  modelliert sie nicht.
- **Feste Stärken auf gleichmäßiger Leiter, ein Remismodell** (Elo-Erwartung bleibt erhalten): andere Stärkeverteilungen sind nicht gemessen.
- **Stichprobe der Zuordnungen:** die Spanne über 100 von 40 320 Zuordnungen ist eine Untergrenze der Spanne über alle: im Standardfall 0,43 gegen 0,59 für den Stärksten. Beim Wechsel von 100 auf 300 Zuordnungen blieb die des Stärksten bei 0,43 (Zufall, die 300 enthalten die ersten 100),
  die größte Spanne stieg von 0,80 auf 0,82; über alle Zuordnungen sind es 0,89.
- **Simulation enthält Rauschen** (2000 Läufe: mittlere Abweichung 0,047 Punkte); bis n = 8 zeigt die App die exakte Zahl, bei n = 10 nur die Simulation.
- **Die Gegnerfolge des besten und des schlechtesten Plans** ist eine Beobachtung; warum eine Folge besser ist, wurde nicht untersucht.
- **Nicht mit echtem Browser gezeigt:** die Abbildungen und das Layout wurden mit AppTest geprüft, nicht im Browser (siehe Verifikation).

## Verifikation

- **Kern:** Zirkeltafel von Hand (n = 4), Partiewahrscheinlichkeiten von Hand (gleich stark: 0,375 / 0,25 / 0,375; 400 Elo Unterschied: 9/11, Remis 2/11), Übergangskern von Hand, stationäre Verteilung für zwei Mannschaften
  gegen die Ergebnisverteilung, `uniform_block` bitgleich zu SplitMix64, Erwartung gegen die Summe der Paar-Erwartungen (`tests/test_algorithm.py`).
- **Orakel:** `tests/test_oracle_spielplan_form.py`, reines Python ohne NumPy und ohne Aufruf des Prüflings (Aufzählung aller Ergebnisbäume, Baum des ±m-Modells, Matrixpotenz).
- **Eigenschaften:** Spanne null ohne Form in 14 zufälligen Ligen (exakt) und 8 (Simulation), Verschiebung aller Elo-Werte ändert nichts, jede Zuordnung ist ein vollständiges Rundenturnier, Reihenfolge der
  Simulationen ohne Einfluss, Spanne wächst mit m (`tests/test_eigenschaften_spielplan_form.py`).
- **Claims und Verdikte:** jede Zahl dieser README in `tests/test_claims.py`, jeder Verdikt-Zweig an Handwerten in `tests/test_verdicts.py`.
- **Oberfläche:** AppTest für jeden Preset-Knopf, jeden Regler an beiden Grenzen, jeden Eintrag der Auswahlregler und beide Experimente; Layout und Footer.
- **Nicht geprüft:** Browser-Ansicht (Kopf, Abbildungen, KaTeX im Expander), Linux/Docker-Lauf (`demo_linux_check.py`), erneute Prüfung nach den Korrekturen der unabhängigen Aussagenprüfung (diese ist gelaufen, ihre Korrekturen sind eingearbeitet).

**177 Tests**, davon einer nur mit `SPF_FULL=1` (vollständige Aufzählung aller 40.320 Zuordnungen, rund vier Minuten; sonst übersprungen). Laufzeit der übrigen 176 seriell 117 Sekunden (ein Lauf am 2026-11-18 im Test-venv; die Experimente und die Claims-Tests rechnen die Messreihen neu).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Messung, Ergebnis, Kernfrage, zwei Experimente, Grenzen, 📐-Expander |
| `spf_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `spf_scenario.py` | Elo-Leiter, Zirkeltafel, Zuordnungen, Partiewahrscheinlichkeiten, `uniform_block` |
| `spf_algorithm.py` | Form (Markow-Zustand), stationäre Verteilung, Simulation mit gemeinsamen Zufallszahlen, exakte Erwartung |
| `spf_evaluation.py` | Messläufe über Zuordnungen, Spannen, Experimente, Verdikte |
| `spf_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `spf_presets.py`, `spf_layout.py` | Permalink, Presets, Zufalls-Seed-Knopf, fester Kopf |
| `spf_rng.py` | wörtliche Kopie des SplitMix64-Generators |
| `tests/` | Einheiten, Orakel, Eigenschaften, Claims, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Kopie-Test |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit-Bibliotheken (ungepinnt), dazu pytest |
| `.github/workflows/tests.yml`, `.streamlit/config.toml`, `.gitignore` | CI, Streamlit-Einstellungen, Ausschlüsse |

## Bewusst nicht umgesetzt

- **Heimvorteil und Breaks:** eigenes Modell; Vermutung, nicht gemessen: bei Heimvorteil wäre die Spanne ohne Form in der einfachen Runde nicht mehr null, weil dann auch die Verteilung des Heimrechts an der Zuordnung hängt.
- **Ruhetage und Ermüdung:** nur der Suchausschnitt einer Studie gesehen; kein Modell.
- **PDF-Export:** die Demo hat keinen.
- **Andere Rückrunden-Ordnungen** (umgekehrte oder zufällige Reihenfolge der Rückrunde), ungerade Mannschaftszahlen (Freilos) und n > 10.
- **Anpassung der Stärke an die Zeit** (Elo-Fortschreibung): Form ist hier ein Zustand, keine Stärkeänderung.

## Literatur (nicht im Volltext gelesen)

- Miller, J. B., Sanjurjo, A. (2018): Surprised by the Hot Hand Fallacy? A Truth in the Law of Small Numbers. Econometrica 86(6). Nur als Hinweis auf die Hot-Hand-Debatte; die Demo prüft sie nicht.
- Studie zu Ruhetagen und Verletzungen (PubMed 23851296), nach Suchausschnitt.
- Elo-Erwartung und Remisanteil: übliches Modell, die Wahl des Remisanteils (Höchstwert, durch die Erwartung begrenzt) ist eine Annahme dieser Demo.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Wettbewerbsverzerrung: Reihenfolge und Anreize](https://sebastianhanisch.net/konzepte-wettbewerbsverzerrung.html).
