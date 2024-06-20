import numpy as np
from astropy.io import fits
import os
import glob

def missing_wavelength_files(filelist):
    missing_files = []
    for f in filelist:
        path = f[0:str.rfind(f,'/')+1]
        sp = fits.open(f)
        header = sp[0].header
        wave_file = header['HIERARCH ESO DRS CAL TH FILE']
        if os.path.isfile(path+wave_file):
            continue
        else:
            missing_files = np.append(missing_files, wave_file)
            
    return np.unique(missing_files)
    
    
if __name__ == '__main__':
    path = '/home/m.gorsini/stellar-karaoke-with-steroids/fits' # data directory
    fnames = [os.path.join(path, fname) for fname in os.listdir(path)[:10] if fname.endswith('.fits')]
    print(fnames)
    missing_files = missing_wavelength_files(fnames)
    print(missing_files)
    np.savetxt('missing_files.txt', missing_files, fmt='%s')
               