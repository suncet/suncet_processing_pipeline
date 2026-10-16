"""
Setup script to install suncet as a Python package.
Reads the requirements.txt file to get dependencies.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

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

repository_root = Path(__file__).resolve().parent
version_string = (repository_root / 'version').read_text(encoding='utf-8').strip()

# Run setup()
setup(
    name='suncet',
    version=version_string, 
    description='Process Sun Coronal Ejection Tracker mission data',
    url='https://github.com/suncet/suncet_processing_pipeline',
    install_requires=requirements,
    python_requires='>=3.11,<3.15',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
    ],
    packages=find_packages(
        exclude=(
            'suncet_processing_pipeline.tests',
            'suncet_processing_pipeline.tests.*',
        )
    ),
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
    entry_points={
        'console_scripts': [
            'suncet-process-new-data=suncet_processing_pipeline.incremental_level0:operational_main',
        ],
    },
    zip_safe=False,
)
