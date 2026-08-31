"""
Setup script to install suncet as a Python package.
Reads the requirements.txt file to get dependencies.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import configparser

from setuptools import find_packages, setup


# -----------------------------------------------------------------------------
# RUN setup() FUNCTION
# -----------------------------------------------------------------------------

# Read in dependencies
with open('requirements.txt', 'r') as txt_file:
    requirements = [
        line.strip()
        for line in txt_file
        if line.strip() and not line.lstrip().startswith('#')
    ]

config = configparser.ConfigParser()
config.read('suncet_processing_pipeline/config_files/config_default.ini')
try:
    version_string = config["structure"]["version_pipeline"]
except KeyError:
    version_string = '0.0.0'

# Run setup()
setup(
    name='suncet',
    version=version_string, 
    description='Process Sun Coronal Ejection Tracker mission data',
    url='https://github.com/suncet/suncet_processing_pipeline',
    install_requires=requirements,
    python_requires='>=3.12',
    packages=find_packages(),
    package_data={
        'suncet_processing_pipeline': [
            'config_files/*.ini',
            'config_files/*.json',
        ],
        'suncet_processing_pipeline.satnogs': [
            'public_beacon_schema.csv',
            'suncet_apid1.ksy',
            'test_data/*.hex',
            'test_data/*.json',
        ],
    },
    zip_safe=False,
)
