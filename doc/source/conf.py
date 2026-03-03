import os
import sys
from datetime import datetime

# 1. Path setup: Points to the 'src' directory relative to this file
sys.path.insert(0, os.path.abspath("../../src"))

# 2. Project information
project = "PEARL"
copyright = f"{datetime.now().year}, PearlLab Team"
author = "PearlLab Team"
release = "1.0.0"

# 3. General configuration
extensions = [
    "sphinx.ext.autodoc",  # Core library to generate docs from docstrings
    "sphinx.ext.autosummary",  # For automated API tables
    "sphinx.ext.napoleon",  # To understand NumPy/Google docstrings
    "sphinx.ext.viewcode",  # To add links to highlighted source code
    "sphinx.ext.intersphinx",  # To link to numpy/pandas external docs
]

# 4. Napoleon settings to match your NumPy style
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# 5. Automation settings
autosummary_generate = True  # Turn on automated API page generation
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# 6. HTML Output
html_theme = "sphinx_rtd_theme"
html_static_path = []  # Use [] to avoid the _static warning if folder is empty

autodoc_mock_imports = ["numba", "numpy", "pandas", "scipy"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
