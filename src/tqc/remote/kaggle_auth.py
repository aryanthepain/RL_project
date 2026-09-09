"""Kaggle credentials resolver, validation, and CLI availability checker."""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import requests


class KaggleAuthError(Exception):
    """Raised when Kaggle credentials cannot be resolved or validated."""
    pass


def _load_env_file(env_path: Path) -> dict:
    """Load key-value pairs from a .env file without external dependencies."""
    env_vars = {}
    if not env_path.is_file():
        return env_vars

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                env_vars[key] = val
    except Exception:
        pass
    return env_vars


def resolve_kaggle_credentials(
    prompt_if_missing: bool = False,
    kaggle_dir: Optional[Union[str, Path]] = None,
    env_path: Optional[Union[str, Path]] = None,
    input_fn: Callable[[str], str] = input,
) -> Tuple[str, str]:
    """Resolve Kaggle credentials from kaggle.json, environment variables, or interactive prompt.

    Priority:
        1. ~/.kaggle/kaggle.json (or specified kaggle_dir)
        2. Environment variables KAGGLE_USERNAME and KAGGLE_KEY (plus .env file)
        3. Interactive prompt (if prompt_if_missing=True)

    Args:
        prompt_if_missing: If True, prompt user interactively when credentials are absent.
        kaggle_dir: Optional custom directory to search for kaggle.json.
        env_path: Optional custom path to .env file.
        input_fn: Injectable input callable for testing prompt interaction.

    Returns:
        Tuple of (username, api_key).

    Raises:
        KaggleAuthError: If credentials cannot be found or resolved.
    """
    # 1. Check kaggle.json
    target_dir = Path(kaggle_dir) if kaggle_dir else Path.home() / ".kaggle"
    kaggle_json = target_dir / "kaggle.json"

    if kaggle_json.is_file():
        try:
            with open(kaggle_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            username = data.get("username", "").strip()
            key = data.get("key", "").strip()
            if username and key:
                return username, key
        except (json.JSONDecodeError, OSError) as e:
            # File exists but is malformed or unreadable
            pass

    # 2. Check environment variables & .env
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()

    if not (username and key):
        project_env = Path(env_path) if env_path else Path(".env")
        env_vars = _load_env_file(project_env)
        if not username:
            username = env_vars.get("KAGGLE_USERNAME", "").strip()
        if not key:
            key = env_vars.get("KAGGLE_KEY", "").strip()

    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        if key.startswith("KGAT_"):
            os.environ["KAGGLE_API_TOKEN"] = key
            access_token_file = target_dir / "access_token"
            if not access_token_file.is_file():
                try:
                    access_token_file.write_text(key, encoding="utf-8")
                except Exception:
                    pass
        return username, key

    # 3. Interactive prompt fallback
    if prompt_if_missing:
        print("\n[Kaggle Authentication] No credentials found in ~/.kaggle/kaggle.json or environment.")
        try:
            prompt_user = input_fn("Enter Kaggle Username: ").strip()
            prompt_key = input_fn("Enter Kaggle API Key: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise KaggleAuthError("Interactive credential prompt interrupted by user.")

        if prompt_user and prompt_key:
            target_dir.mkdir(parents=True, exist_ok=True)
            credential_data = {"username": prompt_user, "key": prompt_key}
            try:
                with open(kaggle_json, "w", encoding="utf-8") as f:
                    json.dump(credential_data, f, indent=2)
                if prompt_key.startswith("KGAT_"):
                    (target_dir / "access_token").write_text(prompt_key, encoding="utf-8")
                    os.environ["KAGGLE_API_TOKEN"] = prompt_key
                # Set permissions to owner-only on POSIX platforms
                if os.name != "nt":
                    os.chmod(kaggle_json, 0o600)
            except OSError as e:
                raise KaggleAuthError(f"Failed to write credentials to {kaggle_json}: {e}")

            os.environ["KAGGLE_USERNAME"] = prompt_user
            os.environ["KAGGLE_KEY"] = prompt_key
            return prompt_user, prompt_key

    raise KaggleAuthError(
        "Kaggle credentials not found. Please provide credentials in ~/.kaggle/kaggle.json, "
        "set KAGGLE_USERNAME and KAGGLE_KEY environment variables, or run with interactive prompt enabled."
    )


def validate_kaggle_credentials(
    username: str,
    api_key: str,
    timeout: float = 10.0,
) -> bool:
    """Validate Kaggle credentials against the official Kaggle API endpoint.

    Uses basic HTTP authentication against Kaggle's datasets endpoint.

    Args:
        username: Kaggle username.
        api_key: Kaggle API token.
        timeout: Request timeout in seconds.

    Returns:
        True if authentication succeeds (HTTP 200), False otherwise.
    """
    if not username or not api_key:
        return False

    url = "https://www.kaggle.com/api/v1/datasets/list"
    try:
        response = requests.get(
            url,
            auth=(username, api_key),
            params={"pageSize": 1},
            timeout=timeout,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def ensure_kaggle_cli_installed() -> bool:
    """Check whether the Kaggle CLI is installed and available in PATH or Python environment.

    Returns:
        True if kaggle binary is on PATH or importable as Python module.
    """
    if shutil.which("kaggle") is not None:
        return True

    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        return False
