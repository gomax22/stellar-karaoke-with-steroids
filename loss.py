import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return kld_loss

class bvae_loss(nn.Module):
    def __init__(self, beta: float):
        super(bvae_loss,self).__init__()
        self.beta = beta
        self.recon_loss = nn.L1Loss(reduction='mean')
        self.kl_loss = KLLoss()

    def forward(self, recons: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kld_weight: float = 1.0):
        recon_loss = self.recon_loss(recons, x)
        kl_loss = self.kl_loss(mu, logvar)

        loss = recon_loss + self.beta * kld_weight * kl_loss
        # print(f"total_loss: {loss:.4f}, recon_loss: {recon_loss:.4f}, kl_loss: {kl_loss:.4f}, kl_loss_weighted: {self.beta * kld_weight * kl_loss:.4f}")

        return loss
    


def reconstruction_loss(x, x_recon):
    batch_size = x.size(0)
    assert batch_size != 0

    recon_loss = F.l1_loss(x_recon, x, size_average=False).div(batch_size)

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld



def sparse_loss(model, spectrum):
    loss = 0
    values = spectrum
    children = list(model.children())
    for i in range(len(children)):
        values = F.relu((children[i](values)))
        loss += torch.mean(torch.abs(values))
    return loss
    