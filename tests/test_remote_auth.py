"""Unit tests for Kaggle authentication and credential resolution."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.tqc.remote.kaggle_auth import (
    KaggleAuthError,
    ensure_kaggle_cli_installed,
    resolve_kaggle_credentials,
    validate_kaggle_credentials,
)


class TestRemoteAuth(unittest.TestCase):
    """Test suite for Kaggle credential detection, validation, and prompt fallback."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.kaggle_dir = Path(self.temp_dir) / ".kaggle"
        self.env_file = Path(self.temp_dir) / ".env"

        # Preserve and clear any active env vars
        self.orig_user = os.environ.get("KAGGLE_USERNAME")
        self.orig_key = os.environ.get("KAGGLE_KEY")
        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if self.orig_user is not None:
            os.environ["KAGGLE_USERNAME"] = self.orig_user
        else:
            os.environ.pop("KAGGLE_USERNAME", None)
        if self.orig_key is not None:
            os.environ["KAGGLE_KEY"] = self.orig_key
        else:
            os.environ.pop("KAGGLE_KEY", None)

    def test_resolve_from_kaggle_json(self):
        """Test resolving credentials from a valid ~/.kaggle/kaggle.json."""
        self.kaggle_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.kaggle_dir / "kaggle.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"username": "test_user", "key": "test_api_key_123"}, f)

        user, key = resolve_kaggle_credentials(kaggle_dir=self.kaggle_dir)
        self.assertEqual(user, "test_user")
        self.assertEqual(key, "test_api_key_123")

    def test_resolve_from_env_vars(self):
        """Test resolving credentials from environment variables."""
        os.environ["KAGGLE_USERNAME"] = "env_user"
        os.environ["KAGGLE_KEY"] = "env_key_456"

        user, key = resolve_kaggle_credentials(kaggle_dir=self.kaggle_dir)
        self.assertEqual(user, "env_user")
        self.assertEqual(key, "env_key_456")

    def test_resolve_from_env_file(self):
        """Test resolving credentials from .env file fallback."""
        with open(self.env_file, "w", encoding="utf-8") as f:
            f.write("# Kaggle Credentials\nKAGGLE_USERNAME=dotenv_user\nKAGGLE_KEY=dotenv_key_789\n")

        user, key = resolve_kaggle_credentials(
            kaggle_dir=self.kaggle_dir,
            env_path=self.env_file,
        )
        self.assertEqual(user, "dotenv_user")
        self.assertEqual(key, "dotenv_key_789")

    def test_missing_credentials_raises_error(self):
        """Test that KaggleAuthError is raised when credentials are missing and prompt is False."""
        with self.assertRaises(KaggleAuthError) as ctx:
            resolve_kaggle_credentials(
                prompt_if_missing=False,
                kaggle_dir=self.kaggle_dir,
                env_path=self.env_file,
            )
        self.assertIn("Kaggle credentials not found", str(ctx.exception))

    def test_interactive_prompt_writes_file(self):
        """Test interactive prompting writes ~/.kaggle/kaggle.json and returns credentials."""
        inputs = iter(["prompted_user", "prompted_key_abc"])

        user, key = resolve_kaggle_credentials(
            prompt_if_missing=True,
            kaggle_dir=self.kaggle_dir,
            env_path=self.env_file,
            input_fn=lambda prompt: next(inputs),
        )

        self.assertEqual(user, "prompted_user")
        self.assertEqual(key, "prompted_key_abc")

        # Verify kaggle.json was created with correct data
        saved_file = self.kaggle_dir / "kaggle.json"
        self.assertTrue(saved_file.is_file())
        with open(saved_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data["username"], "prompted_user")
        self.assertEqual(saved_data["key"], "prompted_key_abc")

    @patch("src.tqc.remote.kaggle_auth.requests.get")
    def test_validate_credentials_success(self, mock_get):
        """Test credential validation when API returns HTTP 200."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        result = validate_kaggle_credentials("valid_user", "valid_key")
        self.assertTrue(result)
        mock_get.assert_called_once()

    @patch("src.tqc.remote.kaggle_auth.requests.get")
    def test_validate_credentials_failure(self, mock_get):
        """Test credential validation when API returns HTTP 401."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        result = validate_kaggle_credentials("bad_user", "bad_key")
        self.assertFalse(result)

    def test_validate_credentials_empty(self):
        """Test credential validation returns False immediately for empty inputs."""
        self.assertFalse(validate_kaggle_credentials("", ""))
        self.assertFalse(validate_kaggle_credentials("user", ""))
        self.assertFalse(validate_kaggle_credentials("", "key"))

    def test_ensure_kaggle_cli_installed(self):
        """Test that ensure_kaggle_cli_installed detects python kaggle module or binary."""
        self.assertTrue(ensure_kaggle_cli_installed())


if __name__ == "__main__":
    unittest.main()
