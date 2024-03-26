
"""
Our downloaded dataset initially consisted of 272376 total spectra, 
which after automatic removal of corrupted files, undefined values 
(NaN: Not a Number) and noise-like ones, was reduced to 267361 “stable” 
ones. 
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import requests
import cgi
import pandas as pd


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--csv", required=True,
                help="path to csv input file")
ap.add_argument("-o", "--output", required=True,
                help="path to output files")
args = vars(ap.parse_args())

# Load berv parameters
bervs = pd.read_csv(args['csv'])

fnames = bervs['dp_id'].values

# check if path exists and files already downloaded
if not Path(args["output"]).exists():
    # create output path
    Path(args["output"]).mkdir(parents=True, exist_ok=True)

# distribute data over cpus

def getDispositionFilename( response ):
    """Get the filename from the Content-Disposition in the response's http header"""
    contentdisposition = response.headers.get('Content-Disposition')
    if contentdisposition == None:
        return None
    value, params = cgi.parse_header(contentdisposition)
    filename = params["filename"]
    return filename

def writeFile( response ):
    """Write on disk the retrieved file"""
    if response.status_code == 200:
        # The ESO filename can be found in the response header
        filename = getDispositionFilename( response )
        # Let's write on disk the downloaded FITS spectrum using the ESO filename:
        with open(os.path.join(args['output'], filename), 'wb') as f:
            f.write(response.content)
        return filename 
    
def download(fname):
    # download file
    file_url = 'https://dataportal.eso.org/dataportal_new/file/' + fname
    response = requests.get(file_url)
    filename = writeFile( response )
    if filename:
        print("Saved file: %s" % (filename))
    else:
        print("Could not get file (status: %d)" % (response.status_code))
    

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:

        print(f"started at {time.strftime('%X')}")
        
        # launch downloads over cpus
        futures = {executor.submit(download, fname) for fname in fnames}
    
    print(f"finished at {time.strftime('%X')}")