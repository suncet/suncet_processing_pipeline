import os
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
import ssl


tmp_file_urls = [] # dl=1 is important for dropbox links

def run():
    if os.getenv('suncet_data') == None:
        os.environ['suncet_data'] = './'
    
    example_path = Path('./ancillary/')
    example_path.mkdir(parents=True, exist_ok=True)

    ssl._create_default_https_context = ssl._create_unverified_context

    for url in tmp_file_urls:
        thisurl = urllib.request.urlopen(url)  
        data = thisurl.read()
        thisurl.close()
        filename = get_filename_from_url(url)
        if filename.endswith('fullemiss.nc'):
            filename = example_path / filename
        with open(filename, "wb") as f:
            print('downloading file: {}'.format(filename))
            f.write(data)

def get_filename_from_url(url):
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)


if __name__ == "__main__":
    run()