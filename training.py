import torch


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
                
def eval_model(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               writer,
               device: torch.device):
    """
    Returns a dictionary containing the results of model predicting on data_loader.
    """
    
    outputs = []
    
    eval_loss = 0.
    model.eval()

    with torch.inference_mode():
        for i, (X, _) in enumerate(dataloader):

            X = X.to(device)

            # 1. Forward pass
            output = model(X)

            # stack outputs
            outputs.append(output.detach())

            # 2. Calculate and accumulate loss
            loss = loss_fn(X, output)
            writer.add_scalar("Loss/test (batch)", loss, i)

            # Calculate and accumulate loss
            eval_loss += loss.item()

    eval_loss /= len(dataloader)

    return {"model_name": model.__class__.__name__,
            "model_loss": eval_loss}, torch.cat(outputs, dim=0)



def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               writer,
               device: torch.device):
    global steps
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss = 0.
    
    batch_size = next(iter(dataloader))[0].shape[0]
    kld_weight = batch_size / (len(dataloader) * batch_size)

    # Loop through data loader data batches
    for batch_i, (X, _) in enumerate(dataloader):

        X = X.to(device)

        # 1. Forward pass
        out = model(X)
        
        # 2. Calculate  and accumulate loss
        loss = loss_fn(X, out)
        writer.add_scalar("Loss/train (batch)", loss, batch_i)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()
    
        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        train_loss += loss.item()

    # Adjust metrics to get average loss per epoch 
    train_loss = train_loss / len(dataloader)

    return train_loss

def validation_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              writer,
              device: torch.device):
    # Put model in eval mode
    model.eval() 
    
    # Setup val loss and val accuracy values
    val_loss = 0.

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, _) in enumerate(dataloader):

            X = X.to(device)

            # 1. Forward pass
            out = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(X, out)
            writer.add_scalar("Loss/val (batch)", loss, batch)
            
            # Calculate and accumulate loss
            val_loss += loss.item()

            
    # Adjust metrics to get average loss per epoch 
    val_loss = val_loss / len(dataloader)

    return val_loss


# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          loss_fn: torch.nn.Module,
          early_stopping: EarlyStopping,
          writer,
          epochs: int = 100,
          epoch_step: int = 10):
    
    # set model to device (necessary)
    model.to(device)

    # 2. Create empty results dictionary
    results = {"train_loss": [], "val_loss": []}
    best_val_loss = torch.inf
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                writer=writer,
                                device=device)
        val_loss = validation_step(model=model, 
                              dataloader=val_dataloader,
                              loss_fn=loss_fn,
                              writer=writer,
                              device=device)
        
        writer.add_scalar("Loss/train (epoch)", train_loss, epoch)
        writer.add_scalar("Loss/val (epoch)", val_loss, epoch)
        
        # 4. Print out what's happening
        if epoch % epoch_step == 0 or epoch == 0 or epoch_step == 1:
            print(
                f"Epoch: {epoch} \t| "
                f"train_loss: {train_loss:.4f} \t| "
                f"val_loss: {val_loss:.4f} | "
            )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        # save checkpoint
        if results["val_loss"][-1] < best_val_loss:
            best_val_loss = results["val_loss"][-1]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, f"models/{model.__class__.__name__}.pth")

        
        early_stopping(train_loss, val_loss)
        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            break
    
    # 6. Return the filled results at the end of the epochs
    return results



