# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../src/slax'))

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

project = 'Slax'
copyright = '2025, Thomas Summe'
author = 'Thomas Summe'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'autoapi.extension',
]
autoapi_dirs = ['../src/slax']
autoapi_python_class_content = 'both'
autoapi_options = [ 'members', 'private-members', 'show-inheritance', 'show-module-summary', 'special-members', 'imported-members',]
#autodoc_mock_imports = ["slax"]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
