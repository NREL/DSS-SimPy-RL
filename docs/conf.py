# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.path.abspath('../envs'))
sys.path.insert(2, os.path.abspath('../envs/simpy_env'))
sys.path.insert(3, os.path.abspath('../envs/simpy_dss'))
sys.path.insert(4, os.path.abspath('../agents/imitation/behavioral_cloning/experts'))
sys.path.insert(5, os.path.abspath('../visuals'))
sys.path.insert(6, os.path.abspath('../agents'))
sys.path.insert(7, os.path.abspath('../agents/imitation/behavioral_cloning/tests'))
sys.path.insert(8, os.path.abspath('../agents/imitation/behavioral_cloning/'))
sys.path.insert(9, os.path.abspath('../mininet'))

# -- Project information -----------------------------------------------------

project = 'Adaptive Resilience Metric IRL'
copyright = '2022, NREL, Golden, CO'
author = 'Abhijeet Sahu'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']