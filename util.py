from astropy.io import fits
import os
from typing import Dict
import torch
from pathlib import Path

def load_harps_spectrum(specID, data_path='fits/'):
    """
    Load a HARPS spectrum from the FITS file.
    """
    hdu = fits.open(os.path.join(data_path, specID+'.fits'))
    data = hdu[1].data
    wave = hdu[1].data.field('WAVE').T
    flux = hdu[1].data.field('FLUX').T

    return wave, flux



def save_fits(tensors: Dict[str, torch.Tensor],
              output_folder: str,
              overwrite: bool = True,
              verbose: bool = True):
    
    extension = ['fits']
    
    # creating directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for filename, data in tensors.items():
        if verbose: print(f"\n------------- START SAVE {filename.upper()} -------------")
        # check extension
        if filename.split(".")[-1] not in extension:
            filename += '.fits'

            if verbose:
                print(f"Extension \'.fits\' added to {filename}")
        
        if verbose: 
            print(f"Storing data as {filename} in folder: ")
            print(output_folder)

        # if is_writable(output_folder) == False:
        #    print(f"Storing {filename} failed.")
        #    return
        
        # open PrimaryHDU, HDUList object
        hdu = fits.PrimaryHDU(data)
        hdul = fits.HDUList([hdu])
        
        # get complete output filepath
        output_fp = os.path.join(output_folder, filename)

        # write fits file
        hdul.writeto(output_fp, overwrite=overwrite)

        if verbose: 
            print(f"File {filename} successfully saved!")
            print(f"------------- END SAVE {filename.upper()} -------------\n")
