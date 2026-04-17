from unittest.mock import patch

from spotforecast2_safe.forecaster.utils import select_n_jobs_fit_forecaster


def test_select_n_jobs_fit_forecaster_returns_int():
    """
    Test that select_n_jobs_fit_forecaster returns an integer.
    """
    n_jobs = select_n_jobs_fit_forecaster("test_forecaster", None)
    assert isinstance(n_jobs, int)
    assert n_jobs >= 1


@patch("os.cpu_count")
def test_select_n_jobs_fit_forecaster_with_mocked_cpu_count(mock_cpu_count):
    """
    Test that select_n_jobs_fit_forecaster uses os.cpu_count() correctly.
    """
    # Case: cpu_count returns 4
    mock_cpu_count.return_value = 4
    assert select_n_jobs_fit_forecaster("test_forecaster", None) == 4

    # Case: cpu_count returns None (fallback to 1)
    mock_cpu_count.return_value = None
    assert select_n_jobs_fit_forecaster("test_forecaster", None) == 1

    # Case: cpu_count returns 0 (edge case, fallback to 1)
    mock_cpu_count.return_value = 0
    assert select_n_jobs_fit_forecaster("test_forecaster", None) == 1


def test_select_n_jobs_fit_forecaster_arguments_ignored():
    """
    Test that arguments are currently ignored but don't cause errors.
    """
    # Just verify it doesn't crash with different types of arguments
    assert select_n_jobs_fit_forecaster("ForecasterRecursive", object()) >= 1
    assert select_n_jobs_fit_forecaster("", None) >= 1
