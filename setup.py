"""
Setup script to install suncet as a Python package.
Reads the requirements.txt file to get dependencies.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from setuptools import find_packages, setup
import configparser


# -----------------------------------------------------------------------------
# RUN setup() FUNCTION
# -----------------------------------------------------------------------------

# Read in dependencies
with open('requirements.txt', 'r') as txt_file:
    requirements = [line.strip() for line in txt_file]

config = configparser.ConfigParser()
config.read('suncet_processing_pipeline/config_files/config_default.ini')
try:
    version_string = config['structure']['version']
except KeyError:
    version_string = '0.0.0'

# Run setup()
setup(
    name='suncet',
    version=version_string, 
    description='Simulate the Sun Coronal Ejection Tracker observations',
    url='https://github.com/suncet/suncet_instrument_simulator',
    install_requires=requirements,
    packages=find_packages(),
    zip_safe=False,
)