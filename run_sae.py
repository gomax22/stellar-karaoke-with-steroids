from models import models
import os
import torch
from torch.utils.data import DataLoader, random_split
from training import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from util import save_fits
from dataset import HARPSDataset
from torchinfo import summary
from training import train, eval_model
import argparse

def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to dataset directory")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output directory")
    args = vars(ap.parse_args())


    #--- Setup the network
    dataset = HARPSDataset(spectra_dir=args['dataset'])
    generator = torch.Generator()

    n_train_samples = int(0.80 * len(dataset))
    n_val_samples = int(0.10 * len(dataset))
    n_test_samples =  len(dataset) - (n_train_samples + n_val_samples)

    # creating training and test split
    X_train, X_val, X_test = random_split(dataset, [n_train_samples, n_val_samples, n_test_samples], generator)

    train_dataloader = DataLoader(X_train, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(X_val, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(X_test, batch_size=64, shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = models.StellarAutoencoder(enc_dim=128).to(device)
    loss_fn = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    print(f"\nNumber of batches in train_dataloader: {len(train_dataloader)}")
    print(f"Number of batches in val_dataloader: {len(val_dataloader)}")
    print(f"Number of batches in test_dataloader: {len(test_dataloader)}")
    print(f"Device: {device}\n")

    try:
        checkpoint = torch.load(f"models/StellarAutoencoder.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']
        print(f"Model loaded successfully frome epoch {epoch} - loss: {loss}.")
    except FileNotFoundError:
        print("No checkpoint found.")
        pass
        
    early_stopping = EarlyStopping(tolerance=5, min_delta=10)
    writer = SummaryWriter()

    summary(model, [128, 1, 327680])

    train(model, 
          train_dataloader, 
          val_dataloader, 
          optimizer, 
          device, 
          loss_fn, 
          early_stopping,
          writer, 
          epochs=15, 
          epoch_step=1)

    
    results, outputs = eval_model(model, test_dataloader, loss_fn, writer, device)

    print(results)
    print(f"Output shape: {outputs.shape}")
    
    # for spectra, (_, fname) in zip(outputs, test_dataloader):
    #     save_fits({f'{fname[0]}': spectra[0].cpu().numpy()}, args["output"])

if __name__ == "__main__":
    main()
