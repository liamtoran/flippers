# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# Import the theme and set it as the default

project = "Flippers"
copyright = f"2023 - {datetime.now().year}, Liam Toran"
author = "Liam Toran"
release = "alpha"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx_argparse_cli",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosectionlabel_prefix_document = True
autosummary_imported_members = True
autosummary_generate = True
autosummary_recursive = True
autodoc_typehints = "description"
autoclass_content = "both"
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    # "special-members": "__init__",
    "exclude-members": "__weakref__",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
