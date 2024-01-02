import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(1152, 512)
        self.fc21 = nn.Linear(512, 256)  # Mean of the latent space
        self.fc22 = nn.Linear(512, 256)  # Standard deviation of the latent space

        # Decoder
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1152)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 1152))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def extract_features(self, x):
        with torch.no_grad():
            mu, _ = self.encode(x.view(-1, 1152))
            return mu
    
    # Loss function with MSE in this case input data normalization with standard format is ok. 
    def loss_function(recon_x, x, mu, logvar, beta=1):
        MSE = F.mse_loss(recon_x, x.view(-1, 1152), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = MSE + beta * KLD
        return total_loss, MSE, KLD
    
