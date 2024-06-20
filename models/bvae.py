import torch
from models.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class BetaVAE_H(BaseVAE):
    def __init__(self,
                 latent_dim: int,
                 beta: float = 0.3,
                 **kwargs) -> None:
        super(BaseVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=(7-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=(5-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=(5-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True), 
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),   
            View((-1, 512 * 20)),
            nn.Linear(512 * 20, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 20),
            View((-1, 512, 20)),
            nn.ConvTranspose1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.weight_init(mode='kaiming')

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = result[:, :self.latent_dim]
        log_var = result[:, self.latent_dim:]

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder(z)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) ->  List[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['kld_weight']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        """
        mu2 = mu ** 2
        var = log_var.exp()
        elbo = torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
        half_elbo = -0.5 * elbo

        kld_loss = torch.mean(half_elbo, dim = 0)
        """
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        """
        print(f"mu (min, max): {mu.min().item():.4f}, {mu.max().item():.4f}")
        print(f"log_var (min, max): {log_var.min().item():.4f}, {log_var.max().item():.4f}")
        print(f"mu2 (min, max): {mu2.min().item():.4f}, {mu2.max().item():.4f}")
        print(f"var (min, max): {var.min().item():.4f}, {var.max().item():.4f}")
        print(f"elbo (min, max): {elbo.min().item():.4f}, {elbo.max().item():.4f}")
        print(f"half_elbo (min, max): {half_elbo.min().item():.4f}, {half_elbo.max().item():.4f}")
        print(f"kld_loss: {kld_loss:.4f}")
        """
        
        # https://openreview.net/forum?id=Sy2fzU9gl
        kld_loss_weighted = self.beta * kld_weight * kld_loss
        loss = recons_loss + kld_loss_weighted

        return {'loss': loss, 'recon_loss':recons_loss, 'kld_loss':kld_loss, 'kld_loss_weighted': kld_loss_weighted}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    
    def weight_init(self, mode: str = 'kaiming'):
        for block in self._modules:
            for m in self._modules[block]:
                if mode == 'kaiming': kaiming_init(m)
                if mode == 'uniform': uniform_init(m)
                if mode == 'normal': normal_init(m, 0, 0.02)


class BetaVAE_B(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 latent_dim: int,
                 gamma:float = 1e3,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 10000,
                 **kwargs) -> None:
        super(BetaVAE_B, self).__init__()

        self.latent_dim = latent_dim
        self.gamma = gamma
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter


        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=(7-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=(5-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=(5-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True), 
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=(3-1)//2, bias=True), nn.LeakyReLU(0.1, inplace=True),   
            View((-1, 512 * 20)),
            nn.Linear(512 * 20, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 20),
            View((-1, 512, 20)),
            nn.ConvTranspose1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.weight_init(mode='kaiming')

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = result[:, :self.latent_dim]
        log_var = result[:, self.latent_dim:]

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder(z)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) ->  List[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['kld_weight']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        """
        mu2 = mu ** 2
        var = log_var.exp()
        elbo = torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
        half_elbo = -0.5 * elbo

        kld_loss = torch.mean(half_elbo, dim = 0)
        """
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        """
        print(f"mu (min, max): {mu.min().item():.4f}, {mu.max().item():.4f}")
        print(f"log_var (min, max): {log_var.min().item():.4f}, {log_var.max().item():.4f}")
        print(f"mu2 (min, max): {mu2.min().item():.4f}, {mu2.max().item():.4f}")
        print(f"var (min, max): {var.min().item():.4f}, {var.max().item():.4f}")
        print(f"elbo (min, max): {elbo.min().item():.4f}, {elbo.max().item():.4f}")
        print(f"half_elbo (min, max): {half_elbo.min().item():.4f}, {half_elbo.max().item():.4f}")
        print(f"kld_loss: {kld_loss:.4f}")
        """
        # https://arxiv.org/pdf/1804.03599.pdf
        self.C_max = self.C_max.to(input.device)

        
        # print(f"num_iter: {(self.C_max/self.C_stop_iter * self.num_iter).detach().cpu().numpy()}")
        C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
        # print(f"C: {C.item():.4f}")
        kld_loss_weighted = self.gamma * kld_weight * (kld_loss - C).abs()
        loss = recons_loss + kld_loss_weighted

        return {'loss': loss, 'recon_loss':recons_loss, 'kld_loss':kld_loss, 'kld_loss_weighted': kld_loss_weighted}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    
    def weight_init(self, mode: str = 'kaiming'):
        for block in self._modules:
            for m in self._modules[block]:
                if mode == 'kaiming': kaiming_init(m)
                if mode == 'uniform': uniform_init(m)
                if mode == 'normal': normal_init(m, 0, 0.02)


def uniform_init(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.uniform_(m.weight, a=-0.008, b=0.008)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    pass
