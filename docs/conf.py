"""conf.py — Configuração do Sphinx para a documentação do projeto."""

import os
import sys

# para localizar os pacotes no diretório pai
sys.path.insert(0, os.path.abspath("../stoneforge"))

# -- Project information -----------------------------------------------------

project = "Stoneforge"
author = "APPy Team"
copyright = "2024 GIEACAR Laboratory, Universidade Federal Fluminense (UFF)"

# -- General configuration ---------------------------------------------------

# Note: The variable 'project_copyright' is used to avoid conflict with the built-in 'copyright'.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Customização de HTML ----------------------------------------------------

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}


# -- Napoleon settings (para docstrings estilo Google/Numpy) -----------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
