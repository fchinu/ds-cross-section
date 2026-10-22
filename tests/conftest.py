"""
pytest configuration: make the analysis code importable and run ROOT in batch mode
"""
import sys
from pathlib import Path

import ROOT

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "preprocess"))  # preprocess/preprocess.py, imported as `preprocess`

ROOT.gROOT.SetBatch(True)
