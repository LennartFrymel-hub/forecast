# Version Management for spotforecast2-safe

## Overview

As a **safety-critical system**, spotforecast2-safe implements **strict version control** to ensure auditability and reproducibility across releases. This document outlines the version management strategy for MLOps engineers.

## Design Principle: Single Source of Truth

**Version is defined once, in `pyproject.toml`**

```toml
[project]
name = "spotforecast2-safe"
version = "0.0.5"  # ← Single source of truth
```

All other files (e.g., `MODEL_CARD.md`, documentation) reference this version programmatically or are automatically synchronized.

## Architecture

### 1. **Version Definition Layer** (`pyproject.toml`)
- Contains the canonical version
- Used by `pip`, `uv`, and build systems
- Updated during release workflows

### 2. **Version Access Layer** (`__init__.py`)
- Exposes version via `__version__` attribute
- Uses `importlib.metadata.version()` (runtime dynamic)
- Fallback for development environments
- Accessible programmatically:
  ```python
  import spotforecast2_safe
  print(spotforecast2_safe.__version__)  # → "0.0.5"
  ```

### 3. **Version Synchronization Layer** (`scripts/update_version.py`)
- Automatically updates all documentation
- Guarantees consistency across files
- Runs in CI/CD pipelines
- Can be manually invoked

### 4. **Version Verification Layer** (CI/CD, Pre-commit hooks)
- Validates version consistency
- Prevents accidental mismatches
- Supports audit compliance

## Workflow

### For MLOps/Release Engineers

#### 1. **During Development**
Version stays at current release version in `pyproject.toml`. No manual updates needed.

#### 2. **Before Release** (e.g., from 0.0.5 → 0.0.6)

**Step 1: Update pyproject.toml**
```toml
[project]
version = "0.0.6"
```

**Step 2: Run version synchronization script**
```bash
# Automatically update all documentation
python scripts/update_version.py

# Or dry-run first (recommended for safety-critical systems)
python scripts/update_version.py --dry-run
```

**Step 3: Verify consistency**
```bash
# Verify all files are in sync
python scripts/update_version.py --verify
```

**Step 4: Commit changes**
```bash
git add pyproject.toml docs/safe/MODEL_CARD.md docs/index.md
git commit -m "chore: release version 0.0.6"
```

#### 3. **Using Pre-commit Hooks** (Recommended)

Pre-commit hooks automatically run version synchronization before each commit:

```bash
# Setup (one-time)
pip install pre-commit
pre-commit install

# Now every commit will automatically synchronize versions
# (Only if pyproject.toml changed)
```

### For Developers

#### I want to check the current version:

**Option A: From Python**
```python
import spotforecast2_safe
print(spotforecast2_safe.__version__)
```

**Option B: From CLI**
```bash
python scripts/update_version.py --verify
```

**Option C: From pyproject.toml**
```bash
grep 'version = ' pyproject.toml
```

#### I want to verify everything is in sync:

```bash
python scripts/update_version.py --verify
```

Exit code: `0` = version consistent, `1` = inconsistency detected

## Script Reference

### `scripts/update_version.py`

**Purpose**: Synchronize version information across documentation files.

**Usage**:
```bash
# Normal: Update all files to match pyproject.toml
python scripts/update_version.py

# Dry-run: Show changes without modifying files
python scripts/update_version.py --dry-run

# Verify: Only check consistency (no modifications)
python scripts/update_version.py --verify
```

**Output**:
```
======================================================================
spotforecast2-safe: Version Synchronization Script
======================================================================

📋 Version Consistency Check:
  pyproject.toml:  0.0.5
  MODEL_CARD.md:   0.0.5
  docs/index.md:   0.0.5
✓ Versions are in sync!
```

## Files Synchronized

1. **`pyproject.toml`** - Primary source
2. **`docs/safe/MODEL_CARD.md`** - Model/Method card (for EU AI Act compliance)
3. **`docs/index.md`** - Documentation landing page (version badge and banner)
4. **`src/spotforecast2_safe/__init__.py`** - Package metadata (dynamic)

## Safety & Compliance Features

### ✅ Audit Trail
```bash
# Git history preserves all version changes
git log --oneline -- pyproject.toml
```

### ✅ Consistency Guarantees
- Version synchronization script prevents mismatches
- Pre-commit hooks enforce synchronization
- CI/CD pipeline verifies before release

### ✅ EU AI Act Compliance
- Version is documented in MODEL_CARD.md (Article 13 - Transparency)
- Version history is traceable in git
- Releases are tagged with version information

### ✅ Reproducibility
- Each version pins exact dependency versions
- `uv lock` creates lockfile for determinism
- Users can reproduce results with `spotforecast2_safe==0.0.5`

## CI/CD Integration

### GitHub Actions Example

<!-- markdownlint-disable MD046 -->
{% raw %}
```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      # Verify version consistency before release
      - name: Verify version consistency
        run: python scripts/update_version.py --verify
      
      # Extract version for tagging
      - name: Get version
        id: version
        run: |
          VERSION=$(grep 'version = ' pyproject.toml | cut -d'"' -f2)
          echo "version=${VERSION}" >> $GITHUB_OUTPUT
      
      # Build and publish to PyPI
      - name: Build and publish
        run: |
          uv build
          uv publish
        env:
          UV_PUBLISH_TOKEN: ${{ secrets.PYPI_TOKEN }}
```
{% endraw %}
<!-- markdownlint-enable MD046 -->

## Troubleshooting

### Issue: Version mismatch detected

```
⚠ Version mismatch detected!
  pyproject.toml:  0.0.6
  MODEL_CARD.md:   0.0.5
```

**Solution**: Run synchronization script
```bash
python scripts/update_version.py
```

### Issue: Pre-commit hook prevents commit

This is intentional! The hook detected a version mismatch.

**Solution**:
```bash
# Let the hook automatically fix it
pre-commit run --all-files

# Then commit normally
git add .
git commit -m "chore: synchronize versions"
```

### Issue: `importlib.metadata` not found

This should not happen in Python 3.8+. If it does:

1. Check Python version: `python --version`
2. Ensure package is installed: `uv pip install -e .`
3. Fallback version (0.0.5) will be used automatically

## Best Practices

### ✅ Do

- Update version **only in `pyproject.toml`**
- Run `update_version.py` after changing version
- Use pre-commit hooks for automation
- Verify consistency before releases (`--verify` flag)
- Tag releases with version: `git tag v0.0.6`

### ❌ Don't

- Manually edit version in MODEL_CARD.md
- Forget to synchronize after bumping version
- Skip the verify step before releases
- Commit version mismatches to main branch

## Versioning Scheme

spotforecast2-safe uses **Semantic Versioning** (MAJOR.MINOR.PATCH):

- **Major** (0.X.X): Breaking API changes (rare for a data transformation library)
- **Minor** (X.Y.X): New features, improvements (backward compatible)
- **Patch** (X.Y.Z): Bug fixes, documentation updates

Example:
- `0.0.1` → `0.0.2`: Patch - bug fix
- `0.0.5` → `0.1.0`: Minor - new feature
- `0.X.X` → `1.0.0`: Major - breaking change

## References

- [Semantic Versioning](https://semver.org/)
- [EU AI Act - Transparency (Article 13)](https://eur-lex.europa.eu/eli/reg/2023/1689/oj)
- [PEP 440 - Version Identification](https://www.python.org/dev/peps/pep-0440/)
- [importlib.metadata Documentation](https://docs.python.org/3/library/importlib.metadata.html)
