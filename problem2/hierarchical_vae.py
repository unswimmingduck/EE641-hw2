import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalDrumVAE(nn.Module):
    def __init__(self, z_high_dim=4, z_low_dim=12):
        super().__init__()
        self.z_high_dim = z_high_dim
        self.z_low_dim = z_low_dim


        self.enc_conv = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.Flatten()  
        )
        self.fc_mu_low = nn.Linear(512, self.z_low_dim)
        self.fc_logvar_low = nn.Linear(512, self.z_low_dim)

        self.enc_high = nn.Sequential(
            nn.Linear(self.z_low_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_mu_high = nn.Linear(32, self.z_high_dim)
        self.fc_logvar_high = nn.Linear(32, self.z_high_dim)

        self.prior_low_mu = nn.Sequential(
            nn.Linear(self.z_high_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.z_low_dim)
        )
        self.prior_low_logvar = nn.Sequential(
            nn.Linear(self.z_high_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.z_low_dim)
        )


        dec_in = self.z_high_dim + self.z_low_dim
        self.dec_fc = nn.Sequential(
            nn.Linear(dec_in, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4 * 128), 
            nn.ReLU(inplace=True),
        )
        self.dec_deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),   
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),    
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 9, kernel_size=3, padding=1)  
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_hierarchy(self, x):
        """
        x: [B,16,9] float
        returns:
          (mu_low, logvar_low, z_low),
          (mu_high, logvar_high, z_high)
        """
        x = x.transpose(1, 2).float() 
        h = self.enc_conv(x)          
        mu_low = self.fc_mu_low(h)
        logvar_low = self.fc_logvar_low(h)
        z_low = self.reparameterize(mu_low, logvar_low)

        h2 = self.enc_high(z_low)      
        mu_high = self.fc_mu_high(h2)
        logvar_high = self.fc_logvar_high(h2)
        z_high = self.reparameterize(mu_high, logvar_high)
        return (mu_low, logvar_low, z_low), (mu_high, logvar_high, z_high)

    def decode_hierarchy(self, z_high, z_low=None, temperature=1.0):
        """
        If z_low is None, sample from conditional prior p(z_low|z_high)
        temperature scales logits before Bernoulli (lower => sharper)
        """
        if z_low is None:
            mu_p = self.prior_low_mu(z_high)
            logvar_p = self.prior_low_logvar(z_high)
            z_low = self.reparameterize(mu_p, logvar_p)

        z = torch.cat([z_high, z_low], dim=-1) 
        h = self.dec_fc(z)                      
        h = h.view(h.size(0), 128, 4)           
        logits = self.dec_deconv(h)             
        logits = logits / max(1e-6, float(temperature))
        return logits.transpose(1, 2)

    def forward(self, x, beta=1.0, temperature=1.0):
        (mu_low, logvar_low, z_low), (mu_high, logvar_high, z_high) = self.encode_hierarchy(x)
        recon_logits = self.decode_hierarchy(z_high, z_low, temperature=temperature)
        return recon_logits, mu_low, logvar_low, mu_high, logvar_high
