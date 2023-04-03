"""
Setup script to install suncet_instrument_simulator as a Python package.
Reads the requirements.txt file to get dependencies.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from setuptools import find_packages, setup


# -----------------------------------------------------------------------------
# RUN setup() FUNCTION
# -----------------------------------------------------------------------------

# Read in dependencies
with open('requirements.txt', 'r') as txt_file:
    requirements = [line.strip() for line in txt_file]

with open('version', 'r') as txt_file:
    version_string = [line.strip() for line in txt_file][0]

# Run setup()
setup(
    name='suncet_instrument_simulator',
    version=version_string, 
    description='Simulate the Sun Coronal Ejection Tracker observations',
    url='https://github.com/suncet/suncet_instrument_simulator',
    install_requires=requirements,
    packages=find_packages(),
    zip_safe=False,
)