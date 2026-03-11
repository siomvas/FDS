"""Proxy helpers for model downloads."""

import os
from contextlib import contextmanager
from typing import Generator, Optional


def get_proxy_url(proxy_url: Optional[str] = None) -> Optional[str]:
    """Return the configured proxy URL, if one was explicitly provided."""
    if proxy_url is not None:
        return proxy_url
    return os.environ.get("FDS_HPC_PROXY")


@contextmanager
def hpc_proxy(proxy_url: Optional[str] = None) -> Generator[None, None, None]:
    """Temporarily set `https_proxy` when an explicit proxy is configured."""
    proxy = get_proxy_url(proxy_url)
    if not proxy:
        yield
        return

    original_proxy = os.environ.get("https_proxy")

    try:
        os.environ["https_proxy"] = proxy
        yield
    finally:
        if original_proxy is not None:
            os.environ["https_proxy"] = original_proxy
        else:
            os.environ.pop("https_proxy", None)
