<div align="left">
  <img src="https://raw.githubusercontent.com/sequential-parameter-optimization/spotforecast2-safe/main/logo/spotlogo.png" alt="spotforecast2-safe Logo" width="300">
</div>

# spotforecast2-safe (Core)

## Features

### Version & License

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/downloads/)
[![GitHub Release](https://img.shields.io/github/v/release/LennartFrymel-hub/forecast)](https://github.com/LennartFrymel-hub/forecast/releases)
[![PyPI Version](https://img.shields.io/pypi/v/spotforecast2-safe)](https://pypi.org/project/spotforecast2-safe/)

[![License](https://img.shields.io/github/license/LennartFrymel-hub/forecast)](LICENSE)

### Downloads

[![PyPI Downloads](https://img.shields.io/pypi/dm/spotforecast2-safe)](https://pypi.org/project/spotforecast2-safe/)
[![Total Downloads](https://static.pepy.tech/badge/spotforecast2-safe)](https://pepy.tech/project/spotforecast2-safe)

### Quality

[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Ready-success)](MODEL_CARD.md)
[![Dependencies](https://img.shields.io/badge/dependencies-minimal-blue)](pyproject.toml)
[![Audit](https://img.shields.io/badge/audit-whitebox-brightgreen)](MODEL_CARD.md)
[![Reliability](https://img.shields.io/badge/robustness-fail--safe-orange)](MODEL_CARD.md)
[![Security](https://img.shields.io/badge/security-policy-blue)](https://sequential-parameter-optimization.github.io/spotforecast2-safe/security/)

### Testing

[![Build Status](https://img.shields.io/github/actions/workflow/status/LennartFrymel-hub/forecast/ci.yml?branch=main&label=Tests)](https://github.com/LennartFrymel-hub/forecast/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen)](https://sequential-parameter-optimization.github.io/spotforecast2-safe/)
[![codecov](https://codecov.io/gh/sequential-parameter-optimization/spotforecast2-safe/branch/develop/graph/badge.svg)](https://codecov.io/gh/sequential-parameter-optimization/spotforecast2-safe)
[![REUSE status](https://api.reuse.software/badge/github.com/sequential-parameter-optimization/spotforecast2-safe)](https://api.reuse.software/info/github.com/sequential-parameter-optimization/spotforecast2-safe)

### Scores

[![OpenSSF Scorecard](https://img.shields.io/ossf-scorecard/github.com/LennartFrymel-hub/forecast)](https://scorecard.dev/viewer/?uri=github.com/LennartFrymel-hub/forecast)

OpenSSF Best Practices badge: register this fork at
[bestpractices.dev](https://www.bestpractices.dev/) and replace this note with
the generated badge for `github.com/LennartFrymel-hub/forecast`.



### Status

[![Maintenance](https://img.shields.io/badge/maintenance-active-green)](https://github.com/LennartFrymel-hub/forecast)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



## Safety-Critical Design Goals

`spotforecast2-safe` is a specialized Python library designed to facilitate time series forecasting in safety-critical production environments and embedded systems.

Unlike standard machine and deep learning libraries, it follows a strict Safety-First architecture by design. However, users must independently verify that these features meet their specific regulatory requirements:

Zero Dead Code: We aim to minimize the attack surface by excluding visualization and training logic.
Deterministic Logic: The algorithms are designed to be purely mathematical and deterministic.
Fail-Safe Operation: The system is designed to favor explicit errors over silent failures when encountering invalid data.
EU AI Act Support: The architecture supports transparency and data governance, helping users build compliant high-risk AI components.
Compliance & Inventory Management: The package includes Common Platform Enumeration (CPE) identifiers for vulnerability tracking, SBOM generation, and supply chain disclosure. See [MODEL_CARD.md](MODEL_CARD.md) for the current CPE identifier.

For a detailed technical overview of our safety mechanisms, see [MODEL_CARD.md](MODEL_CARD.md).

An extended version of this library with visualization and additional features is available at: [https://sequential-parameter-optimization.github.io/spotforecast2/](https://sequential-parameter-optimization.github.io/spotforecast2/)

## Disclaimer & Liability

IMPORTANT: This software is provided "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.

In no event shall the authors, copyright holders, or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

The use of this software in safety-critical systems is at the sole risk of the user.

## Attributions

Parts of the code are ported from `skforecast` to reduce external dependencies.
Many thanks to the [skforecast team](https://skforecast.org/0.20.0/more/about-skforecast.html) for their great work!

## Documentation

 Documentation (API) is available at: [https://sequential-parameter-optimization.github.io/spotforecast2-safe/](https://sequential-parameter-optimization.github.io/spotforecast2-safe/)

## Contributing

We welcome contributions to spotforecast2-safe! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide for details on:

- Development setup and coding standards
- Testing and documentation requirements
- Commit message conventions
- Pull request process
- SPDX license header requirements

## License

`spotforecast2-safe` software: [AGPL-3.0-or-later License](LICENSE)


# References

## spotforecast2

The "full" version of `spotforecast2-safe`, which is named `spotforecast`, is available at: [https://sequential-parameter-optimization.github.io/spotforecast2/](https://sequential-parameter-optimization.github.io/spotforecast2/)

## skforecast 

* Amat Rodrigo, J., & Escobar Ortiz, J. (2026). skforecast (Version 0.20.0) [Computer software]. https://doi.org/10.5281/zenodo.8382788 

## spotoptim

* [spotoptim documentation](https://sequential-parameter-optimization.github.io/spotoptim/)

## Quality

* [OpenSSF Best Practices Badge Program](https://www.bestpractices.dev/en)
* [National Vulnarability Database: Common Platform Enumeration (CPE)](https://nvd.nist.gov/products/cpe)
* [Code Coverage (codecov)](https://about.codecov.io)
