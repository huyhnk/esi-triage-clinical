"""Run the reproducible pipeline after raw data have been placed in data/raw/."""

import subprocess
import sys

COMMANDS = [
    [sys.executable, "-m", "src.data_preparation"],
    [sys.executable, "-m", "src.train_numeric"],
    [sys.executable, "-m", "src.train_multimodal"],
    [sys.executable, "-m", "src.analysis_figures"],
]

for command in COMMANDS:
    print("\n$", " ".join(command))
    subprocess.run(command, check=True)
