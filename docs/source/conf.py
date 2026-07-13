# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'SpaBEAT'
copyright = '2026, Shandong University'
author = 'Qingzhen Hou'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# Add your custom CSS
html_css_files = [
    'css/custom.css',
]
html_title = 'SpaBEAT (Spatial Batch Effect Assessment and Testing)'
html_short_title = 'SpaBEAT'

html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'titles_only': False,
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
