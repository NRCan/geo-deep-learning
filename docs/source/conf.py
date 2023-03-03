# -*- coding: utf-8 -*-

# Configuration file for the Sphinx documentation builder.

import os
import sys

# path to repo-head
sys.path.insert(0, os.path.abspath('../..'))

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# Since we aren't installing package here, we mock imports of the dependencies.
sys.path.insert(0, os.path.abspath('../../losses'))
sys.path.insert(0, os.path.abspath('../../models'))
sys.path.insert(0, os.path.abspath('../../config/tiling'))
sys.path.insert(0, os.path.abspath('../../config/inference'))
sys.path.insert(0, os.path.abspath('../../config/training'))

# -- Project information

project = 'Geo-Deep-Learning'
copyright = '2099, Sherbrooke, Quebec, Canada'
author = 'Charles Authier at NRCAN'

release = '0.1'
version = '0.1.0'

# The master toctree document.
master_doc = 'index'

autodoc_mock_imports = [
    'torch', 'numpy', 'ruamel_yaml', 'mlflow', 'hydra',
    'pandas', 'pytorch_lightning', 'rich', 'omegaconf',
    'scipy', 'torchvision', 'rasterio', 'affine', 'segmentation_models_pytorch',
    
]

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

html_static_path = ['_static']

templates_path = ['_templates']

source_suffix = '.rst'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

html_logo = "../img/logo.png"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'GDLdoc'

