import os


def check_env() -> None:
    """Throws if any of the required env vars is not set."""
    from const import REQUIRED_ENV_VARS

    for env_var in REQUIRED_ENV_VARS:
        try:
            assert os.environ.get(env_var, None)
        except Exception:
            raise KeyError(f"Required environment variable {env_var} not found.")
