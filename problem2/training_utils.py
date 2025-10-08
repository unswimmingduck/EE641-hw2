import math
import torch
import numpy as np
from collections import defaultdict


def kl_annealing_schedule(epoch, method='cyclical', total_epochs=100, n_cycles=4, ratio=0.5):
    """
    Return beta \in [0,1].
    - 'linear'   : linearly grows to 1 over total_epochs*ratio
    - 'sigmoid'  : smooth growth
    - 'cyclical' : cyclic growth 0->1 multiple times (good for hierarchical VAEs)
    """
    if method == 'linear':
        t = min(1.0, epoch / max(1, int(total_epochs * ratio)))
        return float(t)
    elif method == 'sigmoid':
        x = (epoch / max(1, total_epochs * ratio)) * 12 - 6
        return float(1 / (1 + math.exp(-x)))
    else:  # cyclical
        period = total_epochs / max(1, n_cycles)
        phase = (epoch % max(1, int(period))) / max(1, (period * ratio))
        beta = min(1.0, max(0.0, phase))
        return float(beta)

def temperature_annealing_schedule(epoch, t_start=2.0, t_end=0.8, total_epochs=100):
    """
    Anneal Bernoulli logits temperature from t_start -> t_end
    """
    if total_epochs <= 1:
        return t_end
    alpha = min(1.0, epoch / (total_epochs - 1))
    return float(t_start + (t_end - t_start) * alpha)


def apply_free_bits(kl_per_dim, free_bits=0.5):
    """
    kl_per_dim: tensor [B, D] KL of each latent dimension
    floor each dim by free_bits (nats) to avoid posterior collapse
    Returns scalar kl loss
    """
    kl_mean = kl_per_dim.mean(dim=0) 
    kl_clamped = torch.clamp(kl_mean, min=free_bits)
    return kl_clamped.sum()


def analyze_posterior_collapse(model, data_loader, device='cuda'):
    """
    Compute average KL per dimension for both levels (approx. using N(0,I) prior).
    """
    model.eval()
    with torch.no_grad():
        kls_low = []
        kls_high = []
        for x, _, _ in data_loader:
            x = x.to(device)
            recon, mu_l, logvar_l, mu_h, logvar_h = model(x)
            kl_low_per_dim = -0.5 * (1 + logvar_l - mu_l.pow(2) - logvar_l.exp())  
            kl_high_per_dim = -0.5 * (1 + logvar_h - mu_h.pow(2) - logvar_h.exp()) 
            kls_low.append(kl_low_per_dim)
            kls_high.append(kl_high_per_dim)
        kls_low = torch.cat(kls_low, dim=0).mean(dim=0).cpu().numpy()
        kls_high = torch.cat(kls_high, dim=0).mean(dim=0).cpu().numpy()
    return {'kl_low_per_dim': kls_low, 'kl_high_per_dim': kls_high}

def sample_diverse_patterns(model, n_styles=5, n_variations=10, temperature=0.9, device='cuda'):
    """
    Sample from prior to visualize style consistency across variations.
    Returns a tensor [n_styles*n_variations, 16, 9] in row-major by style.
    """
    model.eval()
    with torch.no_grad():
        z_high = torch.randn(n_styles, model.z_high_dim, device=device)
        all_samples = []
        for s in range(n_styles):
            z_h = z_high[s].unsqueeze(0).repeat(n_variations, 1)  
            logits = model.decode_hierarchy(z_h, z_low=None, temperature=temperature)
            probs = torch.sigmoid(logits)
            samples = (probs > 0.5).float()
            all_samples.append(samples.cpu())
        return torch.cat(all_samples, dim=0)  
