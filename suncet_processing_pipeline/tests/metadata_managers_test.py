"""Tests for the metadata_managers module"""

import glob
import os
import pytest
import shutil

from astropy.io import fits
import numpy as np

from .. import metadata_managers

# Path to example run directory in tests/data/
EXAMPLE_RUN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data',
    'example_run'
)


@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmpdir):
    """This function is used to provide setup/teardown for other tests
    in this module.
    """
    for in_file in glob.glob(f'{EXAMPLE_RUN_DIR}/suncet_metadata*'):
        out_file = os.path.join(tmpdir, os.path.basename(in_file))
        shutil.copy(in_file, out_file)

    yield  # do the test

    shutil.rmtree(tmpdir)


def test_fits_constructor(tmpdir):
    """Tests FitsMetadataManager constructors with example test data"""
    metadata_managers.FitsMetadataManager(tmpdir)
    

def test_generate_fits_header(tmpdir):
    """Tests the FitsMetadataManager.generate_fits_header() method"""
    # Create instance of class
    metadata = metadata_managers.FitsMetadataManager(tmpdir)
    metadata.load_from_dict({
        "project_name": "SunCET",
        "data_title": "SunCET Level 1 Image",
        "solr_radius_app": 959.63
    })
    
    # Write fits file and call generate_fits_header()
    fits_path = tmpdir / 'output.fits'
    fits_file = fits.open(fits_path, "append")
        
    image_data = np.zeros((1024, 1024), dtype=np.uint16)
    hdu = fits.ImageHDU(image_data)
    fits_file.append(hdu)
    
    metadata.generate_fits_header(fits_file)

    fits_file.close()
    
    # Test loading the fits file
    fits_file_new = fits.open(fits_path)
    
    assert fits_file_new[0].header['PROJECT'] == 'SunCET'
    assert fits_file_new[0].header['TITLE'] == 'SunCET Level 1 Image'
    assert fits_file_new[0].header['RSUN'] == 959.63

    #raise RuntimeError(repr(fits_file_new[0].header)) # debug
    
    # Test comments are added (doesn't check exact text as metadata may
    # change)
    assert fits_file_new[0].header.comments['PROJECT'] 
    assert fits_file_new[0].header.comments['TITLE']

    # Test units are added in comments
    assert 'nan' not in fits_file_new[0].header.comments['PROJECT'].lower()
    assert 'arcsec' in fits_file_new[0].header.comments['RSUN'].lower()
    
    
def test_load_from_fits(tmpdir):
    """Tests the FitsMetadataManager.generate_fits_header() method.
    
    - Makes first instance and write metadata to output.fits
    - Creates second instance and loads metadata from output.fits, writing
      to output_new.fits
    - Checks output_new.fits has metadata in it
    """
    # Create instance of class
    metadata = metadata_managers.FitsMetadataManager(tmpdir)
    metadata.load_from_dict({
        "project_name": "SunCET",
        "data_title": "SunCET Level 1 Image",
    })
    
    # Write fits file and call generate_fits_header()
    fits_path = tmpdir / 'output.fits'
    fits_file = fits.open(fits_path, "append")
        
    image_data = np.zeros((1024, 1024), dtype=np.uint16)
    fits_file.append(fits.ImageHDU(image_data))
    
    metadata.generate_fits_header(fits_file)

    fits_file.close()
    
    # Test inherting from output.fits to output_new.fits
    fits_path_new = tmpdir / 'output_new.fits'
    fits_file_new = fits.open(fits_path_new, "append")
    fits_file_new.append(fits.ImageHDU(image_data))

    metadata_new = metadata_managers.FitsMetadataManager(tmpdir)
    metadata_new.load_from_fits(fits_path)
    metadata_new.generate_fits_header(fits_file_new)

    fits_file_new.close()

    # Test the metadata is in output_new.fits
    fits_file_new = fits.open(fits_path_new)
    
    assert fits_file_new[0].header['PROJECT'] == 'SunCET'
    assert fits_file_new[0].header['TITLE'] == 'SunCET Level 1 Image'

    # Test comments are added (doesn't check exact text as metadata may
    # change)
    assert fits_file[0].header.comments['PROJECT'] 
    assert fits_file[0].header.comments['TITLE'] 

    
def test_fits_validate_faulty_case(tmpdir):
    """Tests the FitsMetadataManager.validate() method.
    
    - Makes FITS metadata manager instance and uses load_from_dict()
    - tests that exception is thrown for faulty case
    """
    # Create instance of class
    metadata = metadata_managers.FitsMetadataManager(tmpdir)
    metadata.load_from_dict({
        "project_name": "SunCET",
        "data_title": "SunCET Level 1 Image",
    })

    # Check that exception is thrown and that it includes one of
    # the missing keys
    with pytest.raises(metadata_managers.IncompleteMetadataError) as exc:
        metadata.validate(1)        
        assert 'data_ref_doi' in repr(exc)


def test_fits_validate_valid_case(tmpdir):        
    """Tests the FitsMetadataManager.validate() method.
    
    - Makes FITS metadata manager instance and uses load_from_dict()
    - tests that not exception is thrown for valid case
    """
    # Create instance of class
    metadata = metadata_managers.FitsMetadataManager(tmpdir)
    metadata.load_from_dict({
        "project_name": "SunCET",
        "data_title": "SunCET Level 1 Image",
    })

    # Check that not exception is thrown. Do level 0 to force no
    # requirements.
    metadata.validate(0)        

