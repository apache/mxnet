from setuptools import setup
from mxtheme import __version__

setup(
    name = 'mxtheme',
    version = __version__,
    author = 'Mu Li',
    author_email= '',
    url="https://github.com/mli/mx-theme",
    description='A Sphinx theme based on Material Design, adapted from sphinx_materialdesign_theme',
    packages = ['mxtheme'],
    include_package_data=True,
    license= 'MIT License',
    entry_points = {
        'sphinx.html_themes': [
            'mxtheme = mxtheme',
        ]
    },
)
