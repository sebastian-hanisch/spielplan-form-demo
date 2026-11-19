"""Gemeinsame Einstellungen der Tests: der Wurzelordner steht im Importpfad (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
