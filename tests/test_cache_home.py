"""Tests for get_cache_home function."""

import os
import tempfile
from pathlib import Path
import pytest

from spotforecast2_safe.data.fetch_data import get_cache_home


class TestGetCacheHome:
    """Test suite for get_cache_home function."""

    def test_default_cache_home(self):
        """Test get_cache_home with default parameters."""
        cache_home = get_cache_home()

        # Should return a Path object
        assert isinstance(cache_home, Path)

        # Should be an absolute path
        assert cache_home.is_absolute()

        # Should exist after calling
        assert cache_home.exists()

        # Should be in home directory with expected name
        assert cache_home.name == "spotforecast2_cache"
        assert cache_home.parent == Path.home()

    def test_cache_home_with_string_path(self):
        """Test get_cache_home with string path parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "test_cache")
            cache_home = get_cache_home(test_path)

            # Should return a Path object
            assert isinstance(cache_home, Path)

            # Should be absolute
            assert cache_home.is_absolute()

            # Should exist
            assert cache_home.exists()

            # Should have the correct path
            assert str(cache_home) == str(Path(test_path).expanduser().absolute())

    def test_cache_home_with_path_object(self):
        """Test get_cache_home with Path object parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test_cache"
            cache_home = get_cache_home(test_path)

            # Should return a Path object
            assert isinstance(cache_home, Path)

            # Should be absolute
            assert cache_home.is_absolute()

            # Should exist
            assert cache_home.exists()

            # Should match the input path
            assert cache_home == test_path.expanduser().absolute()

    def test_cache_home_creates_directory(self):
        """Test that get_cache_home creates the directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "new", "nested", "cache")

            # Ensure it doesn't exist yet
            assert not os.path.exists(test_path)

            # Call get_cache_home
            cache_home = get_cache_home(test_path)

            # Should now exist
            assert cache_home.exists()
            assert cache_home.is_dir()

    def test_cache_home_with_tilde_expansion(self):
        """Test that ~ is expanded to home directory."""
        cache_home = get_cache_home("~/test_spotforecast_cache")

        # Should be in home directory
        assert str(cache_home).startswith(str(Path.home()))

        # Should not contain ~ in final path
        assert "~" not in str(cache_home)

        # Should exist
        assert cache_home.exists()

    def test_cache_home_idempotent(self):
        """Test that calling get_cache_home multiple times returns the same path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "test_cache")

            cache_home1 = get_cache_home(test_path)
            cache_home2 = get_cache_home(test_path)

            # Should return the same path
            assert cache_home1 == cache_home2

    def test_cache_home_is_directory(self):
        """Test that returned path is a directory, not a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "test_cache")
            cache_home = get_cache_home(test_path)

            # Should be a directory
            assert cache_home.is_dir()

            # Should not be a file
            assert not cache_home.is_file()

    def test_cache_home_with_existing_directory(self):
        """Test get_cache_home with an already existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create the directory
            os.makedirs(tmpdir, exist_ok=True)

            cache_home = get_cache_home(tmpdir)

            # Should work fine
            assert cache_home.exists()
            assert cache_home.is_dir()
            assert str(cache_home) == str(Path(tmpdir).expanduser().absolute())

    def test_cache_home_with_relative_path(self):
        """Test get_cache_home with relative path (should be converted to absolute)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Use relative path
                cache_home = get_cache_home("./test_cache")

                # Should be converted to absolute path
                assert cache_home.is_absolute()
                assert cache_home.exists()

            finally:
                os.chdir(original_cwd)

    def test_cache_home_consistent_with_data_home(self):
        """Test that get_cache_home has similar signature and behavior to get_data_home."""
        from spotforecast2_safe.data.fetch_data import get_data_home

        # Both should work without arguments
        cache_home = get_cache_home()
        data_home = get_data_home()

        # Both should return Path objects
        assert isinstance(cache_home, Path)
        assert isinstance(data_home, Path)

        # Both should be absolute
        assert cache_home.is_absolute()
        assert data_home.is_absolute()

        # Both should exist
        assert cache_home.exists()
        assert data_home.exists()

        # Both should have different names
        assert cache_home.name != data_home.name


class TestGetCacheHomeEnvironment:
    """Test suite for environment variable handling in get_cache_home."""

    def test_cache_home_with_environment_variable(self):
        """Test that SPOTFORECAST2_CACHE environment variable is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_var_path = os.path.join(tmpdir, "env_cache")

            # Set environment variable
            original_env = os.environ.get("SPOTFORECAST2_CACHE")
            try:
                os.environ["SPOTFORECAST2_CACHE"] = env_var_path

                # Call without arguments should use env var
                cache_home = get_cache_home()

                # Should use the environment variable path
                assert str(cache_home) == str(
                    Path(env_var_path).expanduser().absolute()
                )
                assert cache_home.exists()

            finally:
                # Restore original environment
                if original_env is not None:
                    os.environ["SPOTFORECAST2_CACHE"] = original_env
                else:
                    os.environ.pop("SPOTFORECAST2_CACHE", None)

    def test_explicit_path_overrides_environment_variable(self):
        """Test that explicit path parameter overrides SPOTFORECAST2_CACHE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, "env_cache")
            explicit_path = os.path.join(tmpdir, "explicit_cache")

            # Set environment variable
            original_env = os.environ.get("SPOTFORECAST2_CACHE")
            try:
                os.environ["SPOTFORECAST2_CACHE"] = env_path

                # Call with explicit argument should override env var
                cache_home = get_cache_home(explicit_path)

                # Should use the explicit path, not the environment variable
                assert str(cache_home) == str(
                    Path(explicit_path).expanduser().absolute()
                )
                assert cache_home.exists()

            finally:
                # Restore original environment
                if original_env is not None:
                    os.environ["SPOTFORECAST2_CACHE"] = original_env
                else:
                    os.environ.pop("SPOTFORECAST2_CACHE", None)


class TestGetCacheHomeIntegration:
    """Integration tests for get_cache_home."""

    def test_cache_home_can_be_used_for_file_operations(self):
        """Test that cache_home path can be used for actual file operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_home = get_cache_home(tmpdir)

            # Test creating a file in cache_home
            test_file = cache_home / "test_model.joblib"
            test_file.write_text("test content")

            # Should be able to read it back
            assert test_file.exists()
            assert test_file.read_text() == "test content"

            # Cleanup
            test_file.unlink()

    def test_cache_home_suitable_for_persistence(self):
        """Test that cache_home is suitable for model persistence."""
        import pickle

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_home = get_cache_home(tmpdir)

            # Test saving and loading a Python object
            test_obj = {"model": "test", "params": [1, 2, 3]}
            model_file = cache_home / "test_model.pkl"

            # Save
            with open(model_file, "wb") as f:
                pickle.dump(test_obj, f)

            # Load
            with open(model_file, "rb") as f:
                loaded_obj = pickle.load(f)

            # Should be identical
            assert loaded_obj == test_obj

            # Cleanup
            model_file.unlink()


class TestGetCacheHomeCreateDir:
    """Tests for the create_dir parameter of get_cache_home."""

    def test_create_dir_true_creates_directory(self, tmp_path):
        target = tmp_path / "new_cache"
        assert not target.exists()
        result = get_cache_home(target, create_dir=True)
        assert result.exists()
        assert result.is_dir()

    def test_create_dir_false_does_not_create_directory(self, tmp_path):
        target = tmp_path / "no_cache"
        assert not target.exists()
        result = get_cache_home(target, create_dir=False)
        assert not result.exists()

    def test_create_dir_false_returns_correct_path(self, tmp_path):
        target = tmp_path / "resolved_cache"
        result = get_cache_home(target, create_dir=False)
        assert result == target.expanduser().absolute()

    def test_create_dir_false_returns_path_object(self, tmp_path):
        target = tmp_path / "some_cache"
        result = get_cache_home(target, create_dir=False)
        assert isinstance(result, Path)

    def test_create_dir_false_is_absolute(self, tmp_path):
        target = tmp_path / "abs_cache"
        result = get_cache_home(target, create_dir=False)
        assert result.is_absolute()

    def test_create_dir_false_with_existing_directory(self, tmp_path):
        target = tmp_path / "existing_cache"
        target.mkdir()
        result = get_cache_home(target, create_dir=False)
        assert result.exists()
        assert result == target.expanduser().absolute()

    def test_create_dir_false_nested_path_not_created(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        assert not target.exists()
        result = get_cache_home(target, create_dir=False)
        assert not result.exists()
        assert not (tmp_path / "a").exists()

    def test_create_dir_default_is_true(self, tmp_path):
        target = tmp_path / "default_cache"
        assert not target.exists()
        result = get_cache_home(target)
        assert result.exists()

    def test_create_dir_false_with_none_cache_home(self, monkeypatch, tmp_path):
        env_path = tmp_path / "env_cache"
        monkeypatch.setenv("SPOTFORECAST2_CACHE", str(env_path))
        result = get_cache_home(None, create_dir=False)
        assert not result.exists()
        assert result == env_path.expanduser().absolute()

    def test_create_dir_false_with_tilde_path(self):
        result = get_cache_home("~/spotforecast2_cache", create_dir=False)
        assert result.is_absolute()
        assert "~" not in str(result)

    def test_create_dir_false_with_string_path(self, tmp_path):
        target = str(tmp_path / "str_cache")
        result = get_cache_home(target, create_dir=False)
        assert isinstance(result, Path)
        assert not result.exists()

    def test_create_dir_true_idempotent_on_existing_dir(self, tmp_path):
        target = tmp_path / "idempotent_cache"
        target.mkdir()
        result1 = get_cache_home(target, create_dir=True)
        result2 = get_cache_home(target, create_dir=True)
        assert result1 == result2
        assert result1.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
