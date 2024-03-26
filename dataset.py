
import os
from torch.utils.data import Dataset

import numpy as np
import torch
from astropy.io import fits


class HARPSDataset(Dataset):
    def __init__(self, spectra_dir):
        
        # store fits dir
        self.spectra_dir = spectra_dir

        # get filenames 
        self.fnames = [fname for fname in os.listdir(spectra_dir) if fname.endswith('.fits')]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):

        flux = fits.getdata(os.path.join(self.spectra_dir, self.fnames[idx]), ext=0)
        return torch.from_numpy(flux.astype(np.float32)), self.fnames[idx]
    

    
