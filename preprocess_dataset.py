
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
from harps_spec_info import harps_spec_info as spec_info
from util import load_harps_spectrum, save_fits
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory")
args = vars(ap.parse_args())

# get filenames
fnames = [fname[:-5] for fname in os.listdir(args['input']) if fname.endswith('.fits')]

# check if path exists and files already downloaded
if not Path(args["output"]).exists():
    # create output path
    Path(args["output"]).mkdir(parents=True, exist_ok=True)

# distribute data over cpus
    
def preprocess(fname):

    #-- Load the spectrum
    w, flux = load_harps_spectrum(specID=fname, data_path=args['input'])

    #-- Apply the pre-processing steps (trimming, uniform wavelength grids) 
    WAVE, flux = spec_info.preprocess(w, flux)
    
    # do not save noisy spectra
    if not np.any(flux):
        return
    
    # do not save spectra with NaN values
    if not np.isfinite(flux).all():
        return
    
    #-- Save the pre-processed spectrum
    save_fits({fname: flux[np.newaxis, :]}, args['output'])


if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:

        print(f"started at {time.strftime('%X')}")
        
        # launch downloads over cpus
        futures = {executor.submit(preprocess, fname) for fname in fnames}
    
    print(f"finished at {time.strftime('%X')}")