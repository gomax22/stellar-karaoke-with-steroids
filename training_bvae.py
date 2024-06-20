import torch
import torch.utils.data
from torch import nn
from loss import reconstruction_loss, kl_divergence, sparse_loss
from torch.autograd import Variable
from tqdm import tqdm
from models.bvae import BaseVAE



class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            self.counter = 0
                
                
def eval_model(model: BaseVAE,
               dataloader: torch.utils.data.DataLoader, 
               writer,
               device: torch.device):
    """
    Returns a dictionary containing the results of model predicting on data_loader.
    """
    
    batch_size = next(iter(dataloader))[0].shape[0]
    kld_weight = batch_size / (len(dataloader) * batch_size)

    outputs = []

    eval_loss = 0.
    eval_recon_loss = 0.
    eval_kld_loss = 0.

    model.eval()

    with torch.inference_mode():
        for i, (X, _) in enumerate(dataloader):

            X = X.to(device)

            # 1. Forward pass
            output, mu, logvar = model(X)
            
            # 2. Calculate loss
            losses = model.loss_function(output, X, mu, logvar, kld_weight=kld_weight)
            

            # stack outputs
            outputs.append(output.detach())

            # print(f"Train loss: {loss.item():.4f} (recon_loss: {recon_loss.item():.4f}, kld_loss: {kld_weight * total_kld.item():.4f}, total_kld: {total_kld.item():.4f})")
            writer.add_scalar("Total_loss/test (batch)", losses['loss'], i)
            writer.add_scalar("Recon_loss/test (batch)", losses['recon_loss'], i)
            writer.add_scalar("KLD_loss/test (batch)", losses['kld_loss'], i)
            writer.add_scalar("KLD_loss_weighted/test (batch)", losses['kld_loss_weighted'], i)
                

            # Calculate and accumulate loss
            eval_loss += losses['loss'].item()
            eval_recon_loss += losses['recon_loss'].item()
            eval_kld_loss += losses['kld_loss_weighted'].item()


            # print(f"i: {i}, {fnames[0]}: loss = {loss.item():.4f}, cumulative val_loss = {eval_loss:.4f}")

    eval_loss /= len(dataloader)
    eval_recon_loss /= len(dataloader)
    eval_kld_loss /= len(dataloader)

    return {"model_name": model.__class__.__name__,
            "model_loss": eval_loss,
            "model_recon_loss": eval_recon_loss,
            "model_kld_loss": eval_kld_loss}, torch.cat(outputs, dim=0)



def train_step(model: BaseVAE, 
               dataloader: torch.utils.data.DataLoader, 
               optimizer: torch.optim.Optimizer, 
               writer,
               device: torch.device):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss = 0.
    train_recon_loss = 0.
    train_kld_loss = 0.

    batch_size = next(iter(dataloader))[0].shape[0]
    kld_weight = batch_size / (len(dataloader) * batch_size)
    
    # Loop through data loader data batches
    for batch_i, (X, _) in enumerate(dataloader):

        X = X.to(device)

        # 1. Forward pass
        output, mu, logvar = model(X)
        
        # 2. Calculate loss
        losses = model.loss_function(output, X, mu, logvar, kld_weight=kld_weight)
        
        
        # print(f"Train loss: {loss.item():.4f} (recon_loss: {recon_loss.item():.4f}, kld_loss: {kld_weight * total_kld.item():.4f}, total_kld: {total_kld.item():.4f})")
        writer.add_scalar("Total_loss/train (batch)", losses['loss'], batch_i)
        writer.add_scalar("Recon_loss/train (batch)", losses['recon_loss'], batch_i)
        writer.add_scalar("KLD_loss/train (batch)", losses['kld_loss'], batch_i)
        writer.add_scalar("KLD_loss_weighted/train (batch)", losses['kld_loss_weighted'], batch_i)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        losses['loss'].backward()
    
        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        train_loss += losses['loss'].item()
        train_recon_loss += losses['recon_loss'].item()
        train_kld_loss += losses['kld_loss_weighted'].item()

        # print(f"i: {batch_i}, {fnames[0]}: loss = {loss.item():.4f}, cumulative train_loss = {train_loss:.4f}")

       

    # Adjust metrics to get average loss per epoch 
    train_loss = train_loss / len(dataloader)
    train_recon_loss = train_recon_loss / len(dataloader)
    train_kld_loss = train_kld_loss / len(dataloader)

    return train_loss, train_recon_loss, train_kld_loss

def validation_step(model: BaseVAE, 
              dataloader: torch.utils.data.DataLoader, 
              writer,
              device: torch.device):
    # Put model in eval mode
    model.eval() 
    
    batch_size = next(iter(dataloader))[0].shape[0]
    kld_weight = batch_size / (len(dataloader) * batch_size)

    # Setup val loss and val accuracy values
    val_loss = 0.
    val_recon_loss = 0.
    val_kld_loss = 0.

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, _) in enumerate(dataloader):

            X = X.to(device)

            # 1. Forward pass
            output, mu, logvar = model(X)
            
            # 2. Calculate loss
            losses = model.loss_function(output, X, mu, logvar, kld_weight=kld_weight)
            
            
            # print(f"Train loss: {loss.item():.4f} (recon_loss: {recon_loss.item():.4f}, kld_loss: {kld_weight * total_kld.item():.4f}, total_kld: {total_kld.item():.4f})")
            writer.add_scalar("Total_loss/val (batch)", losses['loss'], batch)
            writer.add_scalar("Recon_loss/val (batch)", losses['recon_loss'], batch)
            writer.add_scalar("KLD_loss/val (batch)", losses['kld_loss'], batch)
            writer.add_scalar("KLD_loss_weighted/val (batch)", losses['kld_loss_weighted'], batch)
                
            # Calculate and accumulate loss
            val_loss += losses['loss'].item()
            val_recon_loss += losses['recon_loss'].item()
            val_kld_loss += losses['kld_loss_weighted'].item()

            # print(f"{fnames[0]}: loss = {loss.item():.4f}, cumulative val_loss = {val_loss:.4f}")
            
    # Adjust metrics to get average loss per epoch 
    val_loss = val_loss / len(dataloader)
    val_recon_loss = val_recon_loss / len(dataloader)
    val_kld_loss = val_kld_loss / len(dataloader)

    return val_loss, val_recon_loss, val_kld_loss


# 1. Take in various parameters required for training and test steps
def train(model: BaseVAE, 
          train_dataloader: torch.utils.data.DataLoader,  
          val_dataloader: torch.utils.data.DataLoader,  
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          lr_scheduler: torch.optim.lr_scheduler._LRScheduler, 
          early_stopping: EarlyStopping,
          writer,
          epochs: int = 100,
          epoch_step: int = 10):
    
    # set model to device (necessary)
    model.to(device)

    # 2. Create empty results dictionary
    results = {"train_loss": [], "train_recon_loss": [], "train_kld_loss": [], "val_loss": [], "val_recon_loss": [], "val_kld_loss": []}
    best_val_loss = torch.inf
    
    # Process tqdm bar
    # batch_bar = tqdm(total=epochs, leave=False, position=0, desc="Train")
    

    # Uniform weights for random draw train/test
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_recon_loss, train_kld_loss = train_step(model=model,
                                                        dataloader=train_dataloader,
                                                        optimizer=optimizer,
                                                        writer=writer,
                                                        device=device)
        val_loss, val_recon_loss, val_kld_loss = validation_step(model=model, 
                                                    dataloader=val_dataloader,
                                                    writer=writer,
                                                    device=device)
        
        writer.add_scalar("Total_loss/train (epoch)", train_loss, epoch)
        writer.add_scalar("Total_loss/val (epoch)", val_loss, epoch)
        writer.add_scalar("Recon_loss/train (epoch)", train_recon_loss, epoch)
        writer.add_scalar("Recon_loss/val (epoch)", val_recon_loss, epoch)
        writer.add_scalar("KLD_loss/train (epoch)", train_kld_loss, epoch)
        writer.add_scalar("KLD_loss/val (epoch)", val_kld_loss, epoch)

        # 4. Print out what's happening
        if epoch % epoch_step == 0 or epoch == 0 or epoch_step == 1:
            print(
                f"Epoch: {epoch} \t| "
                f"train_loss: {train_loss:.4f} \t| "
                f"train_recon_loss: {train_recon_loss:.4f} \t| "
                f"train_kld_loss: {train_kld_loss:.4f} \t| "
                f"val_loss: {val_loss:.4f} | "
                f"val_recon_loss: {val_recon_loss:.4f} | "
                f"val_kld_loss: {val_kld_loss:.4f}"
            )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["train_kld_loss"].append(train_kld_loss)
        results["val_kld_loss"].append(val_kld_loss)
        results["train_recon_loss"].append(train_recon_loss)
        results["val_recon_loss"].append(val_recon_loss)

        # save checkpoint
        if results["val_loss"][-1] < best_val_loss:
            best_val_loss = results["val_loss"][-1]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'scheduler': lr_scheduler.state_dict()
            }, f"models/{model.__class__.__name__}.pth")

        # lr_scheduler.step()
        
        early_stopping(train_loss, val_loss)
        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            break
    
    # 6. Return the filled results at the end of the epochs
    return results



