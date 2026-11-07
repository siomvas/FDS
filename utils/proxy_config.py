"""Proxy helpers for model downloads."""

import os
from contextlib import contextmanager
from typing import Generator, Optional

_PROXY_UNSET = object()
_PROXY_ENV_KEYS = ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY")


def get_proxy_url(proxy_url: Optional[str] = None) -> Optional[str]:
    """Resolve the proxy URL for a run.

    Semantics:
    - explicit non-empty string: use that proxy
    - explicit empty string: disable proxy overrides for this run
    - ``None``: fall back to ``FDS_HPC_PROXY`` when configured
    """
    if proxy_url is not None:
        return proxy_url.strip()

    env_proxy = os.environ.get("FDS_HPC_PROXY")
    if env_proxy is None:
        return None
    env_proxy = env_proxy.strip()
    return env_proxy or None


def _snapshot_proxy_env() -> dict[str, object]:
    return {key: os.environ.get(key, _PROXY_UNSET) for key in _PROXY_ENV_KEYS}


def _restore_proxy_env(snapshot: dict[str, object]) -> None:
    for key, value in snapshot.items():
        if value is _PROXY_UNSET:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _set_proxy_env(proxy: Optional[str]) -> None:
    for key in _PROXY_ENV_KEYS:
        if proxy:
            os.environ[key] = proxy
        else:
            os.environ.pop(key, None)


@contextmanager
def hpc_proxy(proxy_url: Optional[str] = None) -> Generator[None, None, None]:
    """Temporarily configure download proxy environment variables.

    Passing ``proxy_url=""`` explicitly disables ambient proxy vars for the
    duration of the context. Passing ``None`` uses ``FDS_HPC_PROXY`` when set.
    """
    resolved_proxy = get_proxy_url(proxy_url)
    if proxy_url is None and resolved_proxy is None:
        yield
        return

    snapshot = _snapshot_proxy_env()
    try:
        _set_proxy_env(resolved_proxy)
        yield
    finally:
        _restore_proxy_env(snapshot)
