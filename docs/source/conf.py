# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath('../..'))

# Mock imports BEFORE importing modules (for autosummary)
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
    'matplotlib',
    'matplotlib.pyplot',
    'scipy',
    'scipy.stats',
    'scipy.spatial',
    'scipy.spatial.distance',
    'scipy.linalg',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.utils',
    'torch.utils.data',
    'torchvision',
    'torchvision.transforms',
    'torchvision.models',
    'PIL',
    'PIL.Image',
    'numpy',
    'alibi_detect',
    'hdbscan',
    'sklearn',
    'sklearn.decomposition',
    'sklearn.cluster',
    'sklearn.random_projection',
    'transformers',
    'timm',
    'open_clip',
    'kymatio',
    'kymatio.numpy',
]

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FDS'
copyright = '2025, FDS Team'
author = 'FDS Team'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',        # Auto-generate from docstrings
    'sphinx.ext.autosummary',    # Summary tables
    'sphinx.ext.napoleon',       # Google/NumPy docstrings
    'sphinx.ext.viewcode',       # Source code links
    'sphinx.ext.intersphinx',    # Link to other docs
]

# Optional: Add sphinx_autodoc_typehints if available
try:
    import sphinx_autodoc_typehints
    extensions.append('sphinx_autodoc_typehints')
except ImportError:
    pass  # Type hints will be shown from docstrings instead

templates_path = ['_templates']
exclude_patterns = []

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autosummary settings
autosummary_generate = False  # Disable auto-generation to avoid import issues
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Suppress warnings for missing references
suppress_warnings = ['ref.any', 'ref.python']

# Mock imports for packages that may not be installed
autodoc_mock_imports = [
    'matplotlib',
    'scipy',
    'torch',
    'torchvision',
    'PIL',
    'numpy',
    'alibi_detect',
    'hdbscan',
    'sklearn',
    'transformers',
    'timm',
    'open_clip',
]

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Try to use Read the Docs theme if available, otherwise use default
try:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_options = {
        'navigation_depth': 4,
        'collapse_navigation': False,
        'sticky_navigation': True,
        'includehidden': True,
        'titles_only': False
    }
except ImportError:
    html_theme = 'alabaster'  # Built-in default theme
    html_theme_options = {
        'sidebar_width': '250px',
        'page_width': '1200px',
    }

html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
