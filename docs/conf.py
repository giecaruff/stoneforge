# -*- coding: utf-8 -*-
"""conf.py — Configuração do Sphinx para a documentação do projeto."""

import os
import sys

# para localizar os pacotes no diretório pai
sys.path.insert(0, os.path.abspath("../stoneforge"))

# -- Project information -----------------------------------------------------

project = "Stoneforge"
author = "APPy Team"
copyright = "2024 GIECAR Laboratory, Universidade Federal Fluminense (UFF)"

# -- General configuration ---------------------------------------------------

# Note: The variable 'project_copyright' is used to avoid conflict with the built-in 'copyright'.

extensions = [
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"

# -- Customização de HTML ----------------------------------------------------

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

# -- Napoleon settings (para docstrings estilo Google/Numpy) -----------------

napoleon_numpy_docstring = True

# -- Autodoc settings --------------------------------------------------------

bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'plain'  # or 'unsrt', 'alpha', etc.