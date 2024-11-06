"""Download auxilary data for a run from online resources"""
import argparse
import os
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
import ssl
import sys
import re
from termcolor import cprint

# URL to synthetic level 1 fits file on dropbox
SYNTHETIC_LEVEL1_URL = 'https://www.dropbox.com/scl/fi/udcemchdku67mjhawfa7b/config_default_OBS_2023-02-14T17-00-00.000.fits?rlkey=iaa9fnjl4imjcs3rlnjceas12&dl=1'

# URL to frozen SunCET metadata file
SUNCET_METADATA_FROZEN_URL = 'https://www.dropbox.com/scl/fi/rbe7vm3sha9mbloek1iio/suncet_metadata_definition_v1.0.0.csv?rlkey=mswa2lvdrvbb9o1rer1z60p2x&dl=1'

# URL to live SunCET metadata file, or development
SUNCET_METADATA_LIVE_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSBYimlrLlhl04mQ-mYZnu6j9aK6sbEzEEMqPuFhK_Qavy3skMpmv9mmzGyGf-msVxARAmIjI-tc8Mh/pub?output=csv'


def run():
    "Main routine of the program"    
    # Parse command line arguments and check that run exists
    args = get_parser().parse_args()
    run_dir = os.path.join('processing_runs', args.run_name)

    if not os.path.exists(run_dir):
        cprint(f"Cannot find run \"{args.run_name}\"; no such directory {run_dir}", "red")
        sys.exit(1)

    # Setup downloading capability for dropbox
    ssl._create_default_https_context = ssl._create_unverified_context
        
    # Save Synthetic Level 1 Data
    synthetic_path = Path(run_dir) / 'level1' / get_filename_from_url(SYNTHETIC_LEVEL1_URL)
    synthetic_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(SYNTHETIC_LEVEL1_URL, synthetic_path)

    # Save SunCET Metatadata
    if args.live_metadata:
        url = SUNCET_METADATA_LIVE_URL
        metadata_version = 'live'
        print("Using 'live' version of metadata from Google Drive")
    else:
        url = SUNCET_METADATA_FROZEN_URL
        metadata_version = find_metadata_version(url)
        print("Using 'frozen' version of metadata from Dropbox")


    # Setup output paths
    metadata_path = Path(run_dir) / 'suncet_metadata_definition.csv'
    metadata_ver_path = Path(run_dir) / 'suncet_metadata_definition_version.csv'
    
    metadata_path.parent.mkdir(parents=True, exist_ok=True)    
        
    download_file(url, metadata_path)
    with open(metadata_ver_path, 'w') as fh:
        cprint('Writing metadata version: ', 'green', end='')
        print(metadata_version)
        fh.write(metadata_version)
    
    # Print message tht all completed successfully
    print("All Downloads completed successfully")


def download_file(url, out_path):
    """
    Download a file to an output path and write a message to console.
    
    Args
       url: URL to download
       out_path: Path to write output to
    """    
    thisurl = urllib.request.urlopen(url)  
    data = thisurl.read()
    thisurl.close()

    with open(out_path, "wb") as fh:
        cprint(f'downloading file:', 'green')
        print(f'  url = {url}')
        print(f'  output path = {out_path}')
        fh.write(data)

        
def get_filename_from_url(url):
    """Parse a filename from a URL
    
    Args
      url: source of filename
    Returns
      string filename
    """    
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)


def get_parser():
    """Get command line ArgumentParser object with options defined.
        
    Returns
       argparse.ArgumentParser object which can be used to parse command
       line objects
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--run-name', type=str, required=True,
                        help='String name of the run')
    parser.add_argument('--live-metadata', action='store_true',
                        help='Use live version of metadata from Google Drive (for devel)')
    return parser


def find_metadata_version(url):
    """Determine the Metadata Version from the filename in a URL.

    If cannot determine, prints an error message and returns "unknown".    

    Args
      url: URL to file, e.g. on dropbox
    Returns
      string version such as "v1.1.0" or "unknown" if could not determine.
    """
    # Expects URLs like https://www.dropbox.com/scl/fi/rbe7vm3sha9mbloek1iio/suncet_metadata_
    # definition_v1.0.0.csv?rlkey=mswa2lvdrvbb9o1rer1z60p2x&dl=1
    match = re.search(r'_v(.*)\.csv', url)

    if match:
        # e.g. 1.1.0 
        version = (
            match
            .group(0)
            .replace('_v', '')
            .replace('.csv', '')
        )
    else:
        cprint('Could not determine version of metadata from URL; setting as "unknown"', 'red')
        version = 'unknown'

    return version


if __name__ == "__main__":
    run()

