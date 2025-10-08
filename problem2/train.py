import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from pathlib import Path

from dataset import DrumPatternDataset
from hierarchical_vae import HierarchicalDrumVAE
from training_utils import kl_annealing_schedule, temperature_annealing_schedule, apply_free_bits

def compute_hierarchical_elbo(recon_x, x, mu_low, logvar_low, mu_high, logvar_high, beta=1.0, use_free_bits=True):
    """
    ELBO = E[log p(x|z)] - beta * ( KL_low + KL_high )
    """
    recon_loss = F.binary_cross_entropy_with_logits(
        recon_x.reshape(-1), x.reshape(-1), reduction='sum'
    )

    kl_high_per_dim = -0.5 * (1 + logvar_high - mu_high.pow(2) - logvar_high.exp())  
    kl_low_per_dim  = -0.5 * (1 + logvar_low  - mu_low.pow(2)  - logvar_low.exp())  

    if use_free_bits:
        kl_high = apply_free_bits(kl_high_per_dim)  
        kl_low  = apply_free_bits(kl_low_per_dim)
    else:
        kl_high = kl_high_per_dim.sum(dim=1).mean()
        kl_low  = kl_low_per_dim.sum(dim=1).mean()

    total = recon_loss + beta * (kl_low + kl_high)
    return total, recon_loss, kl_low, kl_high

def train_epoch(model, data_loader, optimizer, epoch, device, config):
    model.train()
    metrics = {'total_loss': 0, 'recon_loss': 0, 'kl_low': 0, 'kl_high': 0}

    beta = kl_annealing_schedule(epoch, method=config['kl_anneal_method'],
                                 total_epochs=config['num_epochs'])
    temperature = temperature_annealing_schedule(epoch,
                                                 total_epochs=config['num_epochs'])
    for batch_idx, (patterns, styles, densities) in enumerate(data_loader):
        patterns = patterns.to(device)
        optimizer.zero_grad()

        recon, mu_low, logvar_low, mu_high, logvar_high = model(patterns, beta=beta, temperature=temperature)
        loss, recon_loss, kl_low, kl_high = compute_hierarchical_elbo(
            recon, patterns, mu_low, logvar_low, mu_high, logvar_high, beta=beta, use_free_bits=True
        )
        loss.backward()
        optimizer.step()

        bs = patterns.size(0)
        metrics['total_loss'] += loss.item()
        metrics['recon_loss'] += recon_loss.item()
        metrics['kl_low']     += float(kl_low)
        metrics['kl_high']    += float(kl_high)

        if batch_idx % 10 == 0:
            print(f'Epoch {epoch:3d} [{batch_idx:3d}/{len(data_loader)}] '
                  f'Loss: {loss.item()/bs:.4f} Beta: {beta:.3f} Temp: {temperature:.2f}')

    n = len(data_loader.dataset)
    for k in metrics: metrics[k] /= n
    return metrics

def main():
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'z_high_dim': 4,
        'z_low_dim': 12,
        'kl_anneal_method': 'cyclical',  
        'data_dir': 'data/drums',
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results'
    }
    config['device'] = str(config['device'])
    
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    Path(f"{config['results_dir']}/generated_patterns").mkdir(parents=True, exist_ok=True)
    Path(f"{config['results_dir']}/latent_analysis").mkdir(parents=True, exist_ok=True)
    Path(f"{config['results_dir']}/audio_samples").mkdir(parents=True, exist_ok=True)

    train_dataset = DrumPatternDataset(config['data_dir'], split='train')
    val_dataset   = DrumPatternDataset(config['data_dir'], split='val')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'], shuffle=False, num_workers=2)

    model = HierarchicalDrumVAE(config['z_high_dim'], config['z_low_dim']).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    history = {'train': [], 'val': [], 'config': config}

    best_val = float('inf')
    for epoch in range(config['num_epochs']):
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, config['device'], config)
        history['train'].append(train_metrics)

        if epoch % 5 == 0:
            model.eval()
            val_metrics = {'total_loss': 0, 'recon_loss': 0, 'kl_low': 0, 'kl_high': 0}
            with torch.no_grad():
                for patterns, styles, densities in val_loader:
                    patterns = patterns.to(config['device'])
                    recon, mu_low, logvar_low, mu_high, logvar_high = model(patterns)
                    loss, recon_loss, kl_low, kl_high = compute_hierarchical_elbo(
                        recon, patterns, mu_low, logvar_low, mu_high, logvar_high, beta=1.0, use_free_bits=True
                    )
                    val_metrics['total_loss'] += loss.item()
                    val_metrics['recon_loss'] += recon_loss.item()
                    val_metrics['kl_low']     += float(kl_low)
                    val_metrics['kl_high']    += float(kl_high)
            n_val = len(val_dataset)
            for k in val_metrics: val_metrics[k] /= n_val
            history['val'].append(val_metrics)
            print(f"Epoch {epoch:3d} Validation - Loss: {val_metrics['total_loss']:.4f} "
                  f"KL_high: {val_metrics['kl_high']:.4f} KL_low: {val_metrics['kl_low']:.4f}")

            if val_metrics['total_loss'] < best_val:
                best_val = val_metrics['total_loss']
                torch.save(model.state_dict(), f"{config['results_dir']}/best_model.pth")

        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), f"{config['results_dir']}/best_model.pth")
    with open(f"{config['results_dir']}/training_log.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Results saved to {config['results_dir']}/")

if __name__ == '__main__':
    main()
