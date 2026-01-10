"""Utility modules for FDS project.

This package provides utility functions and context managers for common tasks
across the FDS federated learning codebase.
"""

from .proxy_config import hpc_proxy, get_proxy_url

__all__ = ['hpc_proxy', 'get_proxy_url']
