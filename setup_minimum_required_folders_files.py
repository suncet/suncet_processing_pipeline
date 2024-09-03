
import argparse
import os
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
import ssl
import sys
from termcolor import cprint


DOWNLOAD_FILE_URLS = [
    # Note: dl=1 is important for dropbox links
    
    # Synthetic Level 1 FITS file
    'https://www.dropbox.com/scl/fi/udcemchdku67mjhawfa7b/config_default_OBS_2023-02-14T17-00-00.000.fits?rlkey=iaa9fnjl4imjcs3rlnjceas12&dl=1',

    # Metadata Definitions CSV file    
    "https://www.dropbox.com/scl/fi/rbe7vm3sha9mbloek1iio/suncet_metadata_definition_v1.0.0.csv?rlkey=mswa2lvdrvbb9o1rer1z60p2x&dl=1"
]


def run():
    # Parse command line arguments and check that run exists
    args = get_parser().parse_args()
    run_dir = os.path.join('processing_runs', args.run_name)

    if not os.path.exists(run_dir):
        cprint(f"Cannot find run \"{args.run_name}\"; no such directory {run_dir}", "red")
        sys.exit(1)
    
    # Set paths to write synthetic Level 1 data and Metadata CSV
    synthetic_path = Path(run_dir) / 'level1'
    synthetic_path.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(run_dir) 

    ssl._create_default_https_context = ssl._create_unverified_context

    for url in DOWNLOAD_FILE_URLS:
        thisurl = urllib.request.urlopen(url)  
        data = thisurl.read()
        thisurl.close()
        filename = get_filename_from_url(url)
        if filename.endswith('.fits'):
            filename = synthetic_path / filename
        elif filename.startswith('suncet_metadata_'):
            filename = metadata_path / filename
        with open(filename, "wb") as f:
            cprint('downloading file: {}'.format(filename), 'green')
            f.write(data)

    print("All Downloads completed successfully")

    
def get_filename_from_url(url):
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)


def get_parser():
    """Get command line ArgumentParser object with options defined.
        
    Returns
    -------
    parser : argparse.ArgumentParser
       object which can be used to parse command line objects
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--run-name', type=str, required=True,
                        help='String name of the run')

    return parser


if __name__ == "__main__":
    run()
