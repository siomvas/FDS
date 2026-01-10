"""Centralized HPC proxy configuration for model downloads.

This module provides a context manager for safely setting and restoring proxy
configuration when downloading pretrained models on City University HPC cluster.

Environment Variable:
    FDS_HPC_PROXY: Override default proxy URL (optional)
    Default: http://proxy.example.org:3128

Examples:
    >>> from utils.proxy_config import hpc_proxy
    >>>
    >>> # Use default proxy from environment or hardcoded fallback
    >>> with hpc_proxy():
    ...     model = timm.create_model('resnet18', pretrained=True)
    >>>
    >>> # Override with custom proxy
    >>> with hpc_proxy('http://custom-proxy:8080'):
    ...     model = AutoModel.from_pretrained('bert-base')
    >>>
    >>> # Get proxy URL without context manager
    >>> from utils.proxy_config import get_proxy_url
    >>> url = get_proxy_url()
"""

import os
from contextlib import contextmanager
from typing import Generator, Optional


DEFAULT_HPC_PROXY = 'http://proxy.example.org:3128'


def get_proxy_url(proxy_url: Optional[str] = None) -> str:
    """Get the proxy URL to use for model downloads.

    Args:
        proxy_url: Optional override proxy URL. If None, reads from
            FDS_HPC_PROXY environment variable, then falls back to default.

    Returns:
        str: Proxy URL to use

    Examples:
        >>> get_proxy_url()
        'http://proxy.example.org:3128'
        >>>
        >>> os.environ['FDS_HPC_PROXY'] = 'http://custom:3128'
        >>> get_proxy_url()
        'http://custom:3128'
        >>>
        >>> get_proxy_url('http://override:8080')
        'http://override:8080'
    """
    if proxy_url is not None:
        return proxy_url

    # Check environment variable
    env_proxy = os.environ.get('FDS_HPC_PROXY')
    if env_proxy:
        return env_proxy

    # Fallback to default
    return DEFAULT_HPC_PROXY


@contextmanager
def hpc_proxy(proxy_url: Optional[str] = None) -> Generator[None, None, None]:
    """Context manager for HPC proxy configuration.

    Sets https_proxy environment variable for model downloads on City University HPC
    cluster. Automatically restores original proxy setting on exit, including on
    exceptions. This ensures proxy settings don't leak between different operations.

    Thread-safe: Stores and restores the original proxy value rather than deleting it,
    which prevents breaking nested proxy contexts.

    Args:
        proxy_url: Optional proxy URL. If None, reads from FDS_HPC_PROXY env var,
            then defaults to 'http://proxy.example.org:3128'

    Yields:
        None: Context where https_proxy is set

    Examples:
        >>> # Basic usage with default proxy
        >>> with hpc_proxy():
        ...     model = timm.create_model('resnet18', pretrained=True)
        >>>
        >>> # Custom proxy override
        >>> with hpc_proxy('http://custom-proxy:8080'):
        ...     weights = torch.hub.load_state_dict_from_url(url)
        >>>
        >>> # Nested contexts work correctly
        >>> with hpc_proxy('http://outer:3128'):
        ...     with hpc_proxy('http://inner:8080'):
        ...         # Uses inner proxy
        ...         pass
        ...     # Back to outer proxy
        ...     pass
        >>> # Original proxy (or no proxy) restored

    Notes:
        - Always restores original https_proxy value on exit
        - If no original value existed, removes the environment variable
        - Exception-safe: cleanup happens even if code inside context raises
        - HPC cluster requirement: Model downloads fail without proxy on HPC network
    """
    # Determine which proxy URL to use
    proxy = get_proxy_url(proxy_url)

    # Store original proxy value (may be None if not set)
    original_proxy = os.environ.get('https_proxy')

    try:
        # Set the proxy
        os.environ['https_proxy'] = proxy
        yield
    finally:
        # Restore original state
        if original_proxy is not None:
            # Restore previous value (supports nested contexts)
            os.environ['https_proxy'] = original_proxy
        else:
            # Remove if didn't exist before
            os.environ.pop('https_proxy', None)
