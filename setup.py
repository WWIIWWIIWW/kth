"""
Installation file for python kth module
"""
import os
from io import open as io_open
from setuptools import setup, find_packages

package_name = 'kth'
source_name = 'src'

__version__ = None
filepath = os.path.dirname(__file__)
version_file = os.path.join(filepath, source_name, '_version.py')

with io_open(version_file, mode='r') as fd:
    exec(fd.read())

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()


def license():
    with open('LICENSE') as readme_file:
        return readme_file.read()


setup(
    name = package_name,
    package_dir = {'': 'src'},
    packages = {'canteraKTH',
                'keyfi',
                'NN'},
    version = __version__,
    description = 'kth package containing collections',
    long_description = readme(),
    author = 'Kai Zhang',
    author_email='kaizhang@kth.se',
    url='https://github.com/WWIIWWIIWW/kth',
    license=license()
)
