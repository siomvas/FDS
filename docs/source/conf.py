# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

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
]

templates_path = ['_templates']
exclude_patterns = [
    'api/generated/embedding*.rst',
]

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
autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Suppress warnings for missing references
suppress_warnings = ['ref.any', 'ref.python', 'autosummary.import_cycle']

# Mock imports for packages that may not be installed
autodoc_mock_imports = [
    'datasets',
    'alibi_detect',
    'open_clip',
    'matplotlib',
    'matplotlib.cm',
    'matplotlib.patches',
    'kymatio',
    'kymatio.numpy',
]

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
