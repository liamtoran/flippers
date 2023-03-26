# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath(".."))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# Import the theme and set it as the default

project = "Flippers"
copyright = f"2013 - {datetime.now().year}, Liam Toran"
author = "Liam Toran"
release = "alpha"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ["_templates"]
exclude_patterns = ["_build", "templates", "includes", "themes"]

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

extensions = [
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.autodoc",
    "nbsphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
]

# # see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False


autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "exclude-members": "__weakref__",
}

# Do not execute the notebooks when building the docs
nbsphinx_execute = "never"
