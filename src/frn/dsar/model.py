r"""
DSAR model definition.
"""
from typing import List
import torch
import torch.nn as nn
from torch.optim.adam import Adam
import lightning as ltn
# from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MatthewsCorrCoef
# from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
import frn.constants as const


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        nn_modules = []
        for h_dim in hidden_dims:
            nn_modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    nn.LeakyReLU(),
                )
            )
            input_dim = h_dim
        self.encoder = nn.Sequential(*nn_modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        nn_modules = []
        for h_dim in hidden_dims:
            nn_modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    nn.LeakyReLU(),
                )
            )
            input_dim = h_dim
        nn_modules.append(nn.Linear(hidden_dims[-1], 1))
        self.discriminator = nn.Sequential(*nn_modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.discriminator(x)
        prob_real_or_fake = torch.sigmoid(result)
        return prob_real_or_fake


class Decoder(nn.Module):
    def __init__(self, output_dim: int, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        nn_modules = []
        for h_dim in hidden_dims:
            nn_modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim, h_dim),
                    nn.Sigmoid(),
                )
            )
            latent_dim = h_dim
        nn_modules.append(nn.Linear(hidden_dims[-1], output_dim))
        nn_modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*nn_modules)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder(z)
        return result


class Generator(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        nn_modules = []
        for h_dim in hidden_dims:
            nn_modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim, h_dim),
                    nn.Sigmoid(),
                )
            )
            latent_dim = h_dim
        nn_modules.append(nn.Linear(hidden_dims[-1], input_dim))
        nn_modules.append(nn.Sigmoid())
        self.generator = nn.Sequential(*nn_modules)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        result = self.generator(z)
        return result


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super().__init__()

        # Build encoder
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims)

        # Build decoder
        self.decoder = Decoder(input_dim, latent_dim, hidden_dims)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return [self.decoder(z), z, mu, log_var]


class DSAR(ltn.LightningModule):
    def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            hidden_dims_vae: List[int],
            hidden_dims_disc: List[int],
            lr: float,
        ):
        r"""DSAR.

        Args:
            input_dim: input dimension.

            latent_dim: latent dimension.
            
            hidden_dims_vae: hidden dimensions of VAE.
            
            hidden_dims_disc: hidden dimensions of Discriminator.
            
            lr: learning rate.
        
        """
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr

        self.vae = VAE(input_dim, latent_dim, hidden_dims_vae)
        self.discriminator = Discriminator(input_dim, hidden_dims_disc)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        recon, z, mu, log_var = self.vae(x)
        prob_real = self.discriminator(x)
        prob_fake = self.discriminator(recon)
        return [recon, z, mu, log_var, prob_real, prob_fake]
    
    def get_loss(self, x) -> List[torch.Tensor]:
        recon, z, mu, log_var, prob_real, prob_fake = self.forward(x)
        
        # VAE
        recon_loss = nn.functional.mse_loss(recon, x)
        kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        vae_loss = recon_loss + kl_loss

        # GAN
        loss_adv = nn.BCELoss()
        gan_loss_disc = loss_adv(prob_real, torch.ones_like(prob_real)) + loss_adv(prob_fake, torch.zeros_like(prob_fake))
        gan_loss_gen = loss_adv(prob_fake, torch.ones_like(prob_fake))

        # Total
        total_loss = vae_loss + gan_loss_gen + gan_loss_disc * 0.5
        return [total_loss, recon]
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": const.dkey.title_val_loss}
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x = batch
        loss, recon = self.get_loss(x)
        self.log(const.dkey.title_trn_loss, loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x = batch
        loss, recon = self.get_loss(x)
        self.log(const.dkey.title_val_loss, loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        x = batch
        loss, recon = self.get_loss(x)
        self.log(const.dkey.title_tst_loss, loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        x = batch
        loss, recon = self.get_loss(x)
        return recon
    
    def get_latent(self, batch, batch_idx) -> torch.Tensor:
        x = batch
        recon, z, mu, log_var, prob_real, prob_fake = self.forward(x)
        return z
