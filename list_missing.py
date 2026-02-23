import yaml
import glob
from pathlib import Path

# Load quarto yaml
with open('_quarto.yml', 'r') as f:
    quarto_cfg = yaml.safe_load(f)

documented = set()
for section in quarto_cfg.get('quartodoc', {}).get('sections', []):
    for item in section.get('contents', []):
        documented.add(item)

# Find all py files
all_py_files = set()
for p in Path('src/spotforecast2_safe').rglob('*.py'):
    if p.name == '__init__.py':
        continue
    # Convert path to module format
    mod_path = str(p.relative_to('src/spotforecast2_safe')).replace('/', '.').replace('.py', '')
    all_py_files.add(mod_path)

missing = sorted(list(all_py_files - documented))
for m in missing:
    print(m)
