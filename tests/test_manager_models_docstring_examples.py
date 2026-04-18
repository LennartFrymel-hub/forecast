"""Tests for docstring examples in ForecasterRecursiveModel."""

import doctest

import spotforecast2_safe.manager.models.forecaster_recursive_model as model_module


def test_manager_models_docstring_examples():
    """Validate ForecasterRecursiveModel docstring examples using doctest."""
    # Using testmod for the whole module is safer to ensure consistency with other docstring tests in the repo.
    results = doctest.testmod(
        model_module,
        verbose=True,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )

    assert (
        results.failed == 0
    ), f"Docstring examples failed with {results.failed} failures"
