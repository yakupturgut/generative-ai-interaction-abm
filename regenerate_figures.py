"""Regenerate all repository figures from existing CSV result files."""
from pathlib import Path
from figures import make_all_figures

ROOT = Path(__file__).resolve().parent
make_all_figures(ROOT / 'results', ROOT / 'figures')
print(f'Figures written to {ROOT / "figures"}')
