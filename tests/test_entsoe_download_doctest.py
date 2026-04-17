"""
Pytest tests for download_new_data examples (doctest validation).
"""

import os

from spotforecast2_safe.downloader import entsoe


def test_download_new_data_doctest():
    """Run doctest on download_new_data docstring examples."""
    import doctest

    os.environ["ENTSOE_API_KEY"] = "dummy_key"
    failures, _ = doctest.testmod(
        entsoe, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    assert failures == 0
