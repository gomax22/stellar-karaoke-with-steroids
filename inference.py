import sys
sys.path.append('./models')
from models import models, bvae
import os
import torch
import numpy as np
from dataset import HARPSDataset
from util import load_harps_spectrum
from harps_spec_info import harps_spec_info as spec_info
from torch.utils.data import DataLoader
# dirs = ['preprocessed/split/1', 'preprocessed/split/2', 'preprocessed/split/3']
dirs = ['test']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_ae1d =  models.ae1d().to(device)
model_sae = models.StellarAutoencoder(enc_dim=128).to(device)
model_bvae = bvae.BetaVAE_H(latent_dim=128, beta=0.3).to(device)

pretrained_ae1d = 'models/model_14v3_128d_e116_i954k.pth.tar'
pretrained_sae = 'models/StellarAutoencoder.pth'
pretrained_bvae = 'models/BetaVAE_H.pth'

model_ae1d.load_state_dict(torch.load(pretrained_ae1d, map_location=device)['state_dict'], strict=True)
model_sae.load_state_dict(torch.load(pretrained_sae, map_location=device)['model_state_dict'], strict=True)
model_bvae.load_state_dict(torch.load(pretrained_bvae, map_location=device)['model_state_dict'], strict=True)
model_ae1d.eval()
model_sae.eval()
model_bvae.eval()



loss_fn = torch.nn.L1Loss(reduction='mean')
loss_ae1d = []
loss_sae = []

loss_bvae = []
recon_loss = []
kld_loss = []

fnames = [fname[:-5] for fname in os.listdir('test') if fname.endswith('.fits')]
    
#for folder in dirs:
for i, fname in enumerate(fnames):   
    """
    loss_ae1d = []
    loss_sae = []

    loss_bvae = []
    recon_loss = []
    kld_loss = []
    """
    # dataset = HARPSDataset(folder)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    w, data = load_harps_spectrum(specID=fname, data_path='test')
        
    #-- Apply the pre-processing steps (trimming, uniform wavelength grids) 
    _, flux = spec_info.preprocess(w, data)
    
    #-- Pass it through the network
    flux = torch.from_numpy(flux.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    
    #-- Pass it through the network
    with torch.inference_mode():
        # for i, (flux, _) in enumerate(dataloader):
            # if i % 2500 == 0 or i == 0 or i == len(dataloader) - 1:
            #     print(f"Processing {i} of {len(dataloader)}")
            
        #    print(f"Processing {i} of {len(dataloader)}")
        # print(f"Processing {i} of {len(fnames)}")
        flux = flux.to(device)
        reconst = model_ae1d.forward(flux)
        loss = loss_fn(flux, reconst)
        loss_ae1d.append(loss.detach().cpu().numpy())

        reconst = model_sae.forward(flux)
        loss = loss_fn(flux, reconst)
        loss_sae.append(loss.detach().cpu().numpy())
        
        reconst, mu, logvar = model_bvae.forward(flux)
        loss = model_bvae.loss_function(reconst, flux, mu, logvar, kld_weight=1.0)
        loss_bvae.append(loss['loss'].detach().cpu().numpy())
        recon_loss.append(loss['recon_loss'].detach().cpu().numpy())
        kld_loss.append(loss['kld_loss'].detach().cpu().numpy())        
        
        del flux, reconst, loss, mu, logvar


    print(f"np.mean(loss_ae1d): {np.mean(loss_ae1d)}")
    print(f"np.mean(loss_sae): {np.mean(loss_sae)}")
    print(f"np.mean(loss_bvae): {np.mean(loss_bvae)}")
    print(f"np.mean(recon_loss): {np.mean(recon_loss)}")
    print(f"np.mean(kld_loss): {np.mean(kld_loss)}\n")

    print(f"np.std(loss_ae1d): {np.std(loss_ae1d)}")
    print(f"np.std(loss_sae): {np.std(loss_sae)}")
    print(f"np.std(loss_bvae): {np.std(loss_bvae)}")
    print(f"np.std(recon_loss): {np.std(recon_loss)}")
    print(f"np.std(kld_loss): {np.std(kld_loss)}\n")

    print(f"np.median(loss_ae1d): {np.median(loss_ae1d)}")
    print(f"np.median(loss_sae): {np.median(loss_sae)}")
    print(f"np.median(loss_bvae): {np.median(loss_bvae)}")
    print(f"np.median(recon_loss): {np.median(recon_loss)}")
    print(f"np.median(kld_loss): {np.median(kld_loss)}\n")

    print(f"np.min(loss_ae1d): {np.min(loss_ae1d)}")
    print(f"np.min(loss_sae): {np.min(loss_sae)}")
    print(f"np.min(loss_bvae): {np.min(loss_bvae)}")
    print(f"np.min(recon_loss): {np.min(recon_loss)}")
    print(f"np.min(kld_loss): {np.min(kld_loss)}\n")

    print(f"np.max(loss_ae1d): {np.max(loss_ae1d)}")
    print(f"np.max(loss_sae): {np.max(loss_sae)}")
    print(f"np.max(loss_bvae): {np.max(loss_bvae)}")
    print(f"np.max(recon_loss): {np.max(recon_loss)}")
    print(f"np.max(kld_loss): {np.max(kld_loss)}\n")

    del loss_ae1d, loss_sae, loss_bvae, recon_loss, kld_loss #, dataset, dataloader