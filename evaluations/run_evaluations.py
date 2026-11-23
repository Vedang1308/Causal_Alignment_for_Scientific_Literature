#!/usr/bin/env python3
"""
Evaluations Runner Helper.
Runs phase 6 evaluation on the benchmark and outputs summary metrics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase6_evaluation_and_experiments.run_phase6_verification import run_phase6_verification as run_eval

if __name__ == "__main__":
    run_eval()
