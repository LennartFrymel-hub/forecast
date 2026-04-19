# Contributing to spotforecast2-safe

Thank you for your interest in contributing to spotforecast2-safe! This document provides guidelines and requirements for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please treat all community members with respect.

## Getting Started

### Prerequisites

- Python 3.13 or later
- uv package manager (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Development Setup

1. Clone the repository:

```bash
git clone https://github.com/sequential-parameter-optimization/spotforecast2-safe.git
cd spotforecast2-safe
```

2. Create and activate the virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the project with development dependencies:

```bash
uv sync
```

4. Run the test suite to verify setup:

```bash
uv run pytest tests/ -v
```

## Building the Package

To build the source and binary distributions (wheels):

```bash
# Using the standard build tool
uv run python -m build

# The artifacts will be in the dist/ directory
ls -lah dist/
```

## Documentation

Update or create documentation:

- API documentation goes in `docs/api/`
- Processing guides go in `docs/processing/`
- Preprocessing guides go in `docs/preprocessing/`
- Use Markdown with Google-style docstring conventions

Build documentation locally:

```bash
uv run python docs/quartodoc_build.py
uv run quartodoc interlinks
uv run quarto render --no-cache
```

or

```bash
uv run python docs/quartodoc_build.py; uv run quartodoc interlinks; uv run quarto render --no-cache
```


Open the generated documentation in a local browser:
```bash
open _site/index.html 
```



## Coding Standards

All contributions must adhere to the following standards:

### Code Formatting

- Code style: Black (enforced)
- Import sorting: isort
- Linting: flake8

Run formatting tools before committing:

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/ --max-line-length=180
uv run ruff check src/ tests/
```

### Documentation Style

- Docstrings: Google style format
- All public functions, classes, and modules must have comprehensive docstrings
- Include type hints in function signatures
- Include usage examples in docstrings where applicable

Example:

```python
def get_cpe_identifier(version: str = "*") -> str:
    """Generates the CPE 2.3 identifier for the spotforecast2-safe project.

    This function constructs a Common Platform Enumeration (CPE) 2.3 formatted
    string that uniquely identifies the spotforecast2-safe software.

    Args:
        version: The specific version of the software. Use wildcard "*" to match
            all versions, or provide a semantic version string. Defaults to "*".

    Returns:
        str: The formatted Common Platform Enumeration 2.3 string.

    Raises:
        TypeError: If version is not a string.

    Examples:
        Generate a CPE identifier for all versions:

        >>> get_cpe_identifier()
        'cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:*:*:*:*:*:*:*:*'

    See Also:
        https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-188.pdf
    """
```

### SPDX License Headers

All source files (Python, YAML, etc.) must include SPDX headers at the top:

```python
# SPDX-FileCopyrightText: <year> <your name>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Module docstring."""
```

The project uses REUSE compliance for license tracking. Run the REUSE check:

```bash
uv run reuse lint
```

## Testing Requirements

All contributions must include tests covering new functionality:

- Write tests in `tests/` directory following the existing naming convention: `test_*.py`
- Use pytest as the testing framework
- Aim for high code coverage (minimum 80% for new code)
- Run tests before submitting a pull request:

```bash
uv run pytest tests/ -v --cov=src/spotforecast2_safe
```

Test files should also include SPDX headers and follow the same style guidelines.

## Type Hints

Python 3.13+ features optional type hints are encouraged for all new code. Use precise types to improve IDE support and catch errors early:

```python
from typing import Optional

def process_data(values: list[float], threshold: Optional[float] = None) -> dict[str, int]:
    """Process a list of values."""
    pass
```

## Commit Messages

This project uses Semantic Versioning and Conventional Commits for automatic changelog generation.

Commit message format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types (required):
- feat: A new feature
- fix: A bug fix
- docs: Documentation changes only
- style: Changes that do not affect code meaning (formatting, SPDX headers)
- refactor: Code change that neither fixes a bug nor adds a feature
- perf: Performance improvements
- test: Test additions or changes
- chore: Changes to build system, dependencies, or other non-code changes

Example:

```
feat(cpe): add CPE identifier generation for compliance tracking

Implement get_cpe_identifier() function to generate NIST CPE 2.3
formatted strings for vulnerability tracking and SBOM management.

Closes #42
```

## Pull Request Process

1. Create a feature branch from `develop`:

```bash
git checkout -b feat/your-feature-name
```

2. Make your changes and commit with conventional commit messages
3. Ensure all tests pass and code standards are met
4. Create a Pull Request against the `develop` branch
5. PR description should clearly explain:
   - What problem it solves
   - How the solution works
   - Any breaking changes

Pull Request title should follow the conventional commit format.


## Safety-Critical Standards

This is a safety-critical library. Contributions must maintain:

- Deterministic behavior (same input = same output, bit-level reproducible)
- Fail-safe operation (explicit errors, no silent failures)
- Auditability (white-box code, clear logic, comprehensive tests)
- Minimal dependencies (no unnecessary external packages)

Any changes that affect these properties must be clearly documented and justified in the PR.

## Reporting Issues

Before reporting a bug:

1. Check existing issues to avoid duplicates
2. Run the latest version to confirm the bug persists
3. Include reproduction steps, expected behavior, and actual behavior

Security issues should not be reported in public issues. Email security concerns directly to bartzbeielstein.

## License

By contributing to spotforecast2-safe, you agree that your contributions are licensed under AGPL-3.0-or-later. Include the SPDX header in all new files you create.

## Questions?

- Check existing issues and discussions on GitHub
- Read the [Model/Method Card](safe/MODEL_CARD.qmd) for system design details
- Review the [Safety Documentation](safe/spotforecast2-safe.qmd)

Thank you for contributing to spotforecast2-safe!