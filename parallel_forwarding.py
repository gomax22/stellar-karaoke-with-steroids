
"""
Our downloaded dataset initially consisted of 272376 total spectra, 
which after automatic removal of corrupted files, undefined values 
(NaN: Not a Number) and noise-like ones, was reduced to 267361 “stable” 
ones. 
"""

import argparse
import torch
from concurrent.futures import ProcessPoolExecutor
import time
import sys
sys.path.append('./models')
from models import models
from dataset import HARPSDataset
from torch.utils.data import DataLoader
import os
import numpy as np
from astropy.io import fits
from pathlib import Path

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input files")
ap.add_argument("-o", "--output", required=True,
                help="path to output files")

args = vars(ap.parse_args())

# check if path exists and files already downloaded
if not Path(args["output"]).exists():
    # create output path
    Path(args["output"]).mkdir(parents=True, exist_ok=True)

device = torch.device('cpu')
model = models.ae1d().to(device)
model.load_state_dict(torch.load('models/model_14v3_128d_e116_i954k.pth.tar', map_location=device)['state_dict'],strict=True)
fnames = [fname for fname in os.listdir(args['input']) if fname.endswith('.fits')]
model.eval()



def forward(fname, idx):

    if idx % 1000 == 0:
        print(f"{idx} files processed.")
    flux = torch.from_numpy(fits.getdata(os.path.join(args['input'], fname), ext=0).astype(np.float32))

    flux = flux.unsqueeze(0).to(device)
    with torch.inference_mode():

        for i, layer in enumerate(list(model.children())):
            flux = layer(flux)
            
            if flux.isnan().any():
                # print(i, layer)
                # if i not in layers:
                #    layers[i] = 0
                # layers[i] += 1
                print(f"{idx}: {fname} NaN output.")
                os.rename(os.path.join(args['input'], fname), os.path.join(args['output'], fname))
                return
        

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:

        print(f"started at {time.strftime('%X')}")
        
        # launch downloads over cpus
        futures = {executor.submit(forward, fname, i) for i, fname in enumerate(fnames)}
    
    print(f"finished at {time.strftime('%X')}")