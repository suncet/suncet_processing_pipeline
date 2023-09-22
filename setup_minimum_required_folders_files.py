import os
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
import ssl


tmp_file_urls = ['https://www.dropbox.com/scl/fi/udcemchdku67mjhawfa7b/config_default_OBS_2023-02-14T17-00-00.000.fits?rlkey=iaa9fnjl4imjcs3rlnjceas12&dl=1',
                 "https://www.dropbox.com/scl/fi/rbe7vm3sha9mbloek1iio/suncet_metadata_definition_v1.0.0.csv?rlkey=mswa2lvdrvbb9o1rer1z60p2x&dl=1"] # dl=1 is important for dropbox links

def run():
    if os.getenv('suncet_data') == None:
        os.environ['suncet_data'] = './'
    synthetic_path = Path(os.getenv('suncet_data') + '/synthetic/level0_raw/fits/')
    synthetic_path.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(os.getenv('suncet_data') + '/metadata')
    metadata_path.mkdir(parents=True, exist_ok=True)

    ssl._create_default_https_context = ssl._create_unverified_context

    for url in tmp_file_urls:
        thisurl = urllib.request.urlopen(url)  
        data = thisurl.read()
        thisurl.close()
        filename = get_filename_from_url(url)
        if filename.endswith('.fits'):
            filename = synthetic_path / filename
        elif filename.startswith('suncet_metadata_'):
            filename = metadata_path / filename
        with open(filename, "wb") as f:
            print('downloading file: {}'.format(filename))
            f.write(data)

def get_filename_from_url(url):
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)


if __name__ == "__main__":
    run()
