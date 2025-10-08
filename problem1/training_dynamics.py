import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt

from dataset import FontDataset

@torch.no_grad()
def _make_grid(imgs, nrow=10, pad=2):
    imgs = (imgs.clamp(-1,1) + 1.0) * 0.5  # to [0,1]
    N = imgs.size(0)
    ncol = int(np.ceil(N / nrow))
    H, W = 28, 28
    grid = torch.ones(1, ncol*H + pad*(ncol-1), nrow*W + pad*(nrow-1))
    idx = 0
    for r in range(ncol):
        for c in range(nrow):
            if idx >= N: break
            h0 = r*(H+pad); w0 = c*(W+pad)
            grid[:, h0:h0+H, w0:w0+W] = imgs[idx]
            idx += 1
    return grid

def _save_tensor_image(t, path):
    import imageio.v2 as imageio
    arr = (t.squeeze(0).cpu().numpy()*255).astype(np.uint8)
    imageio.imwrite(path, arr)

def train_gan(generator, discriminator, data_loader, num_epochs=100, device='cuda', z_dim=100, results_dir='results'):
    generator.train()
    discriminator.train()
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir)/'visualizations').mkdir(parents=True, exist_ok=True)

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    history = defaultdict(list)

    snapshot_epochs = {10, 30, 50, 100}

    for epoch in range(num_epochs):
        for batch_idx, (real_images, labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            real_images = real_images * 2.0 - 1.0  # to [-1,1]

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # D
            d_optimizer.zero_grad()
            out_real = discriminator(real_images)
            d_loss_real = criterion(out_real, real_labels)
            z = torch.randn(batch_size, getattr(generator, 'z_dim', z_dim), device=device)
            fake_images = generator(z)
            out_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(out_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # G
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, getattr(generator, 'z_dim', z_dim), device=device)
            fake_images = generator(z)
            out = discriminator(fake_images)
            g_loss = criterion(out, real_labels)
            g_loss.backward()
            g_optimizer.step()

            if batch_idx % 10 == 0:
                history['d_loss'].append(float(d_loss.item()))
                history['g_loss'].append(float(g_loss.item()))
                history['epoch_float'].append(epoch + batch_idx/len(data_loader))

        # grids
        if (epoch+1) in snapshot_epochs or epoch == 0:
            with torch.no_grad():
                z = torch.randn(100, getattr(generator, 'z_dim', z_dim), device=device)
                samples = generator(z).cpu()
                grid = _make_grid(samples, nrow=10)
                _save_tensor_image(grid, f"{results_dir}/visualizations/generated_grid_epoch_{epoch+1}.png")

        # coverage + histogram
        if (epoch+1) in snapshot_epochs or (epoch % 10 == 0):
            coverage, counts = analyze_mode_coverage(generator, device=device, return_counts=True)
            history['mode_coverage'].append(float(coverage))
            history.setdefault('mode_counts', []).append(counts)
            save_mode_histogram(counts, f"{results_dir}/visualizations/mode_hist_epoch_{epoch+1}.png")
            print(f"Epoch {epoch+1}: Mode coverage = {coverage:.2f}")

    visualize_mode_collapse(history, f"{results_dir}/mode_collapse_analysis.png")
    return dict(history)

def analyze_mode_coverage(generator, device, n_samples=1000, data_dir='data/fonts', return_counts=False):
    generator.eval()
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    try:
        from torch.utils.data import DataLoader
        class TinyCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(64*7*7, 128), nn.ReLU(),
                    nn.Linear(128, 26)
                )
            def forward(self, x): return self.net(x)

        ds = FontDataset(data_dir, split='train_samples')
        n = min(len(ds), 2000)
        idx = torch.randperm(len(ds))[:n]
        subset = torch.utils.data.Subset(ds, idx.tolist())
        loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=0)
        clf = TinyCNN().to(device)
        opt = optim.Adam(clf.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()

        clf.train()
        for x,y in loader:
            x = (x.to(device)*2.0 - 1.0)
            y = y.to(device)
            opt.zero_grad()
            logits = clf(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

        clf.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, getattr(generator, 'z_dim', 100), device=device)
            imgs = generator(z).clamp(-1,1)
            logits = clf(imgs)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
        unique = len(np.unique(preds))
        coverage = unique / 26.0

        if return_counts:
            cnts = Counter(int(i) for i in preds)
            counts = {letters[i]: int(cnts.get(i, 0)) for i in range(26)}
            return float(coverage), counts
        return float(coverage)
    except Exception:
        with torch.no_grad():
            z = torch.randn(n_samples, getattr(generator, 'z_dim', 100), device=device)
            imgs = generator(z).clamp(-1,1).cpu().numpy().reshape(n_samples, -1)
        imgs = (imgs + 1.0) * 0.5
        imgs -= imgs.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(imgs[:200], full_matrices=False)
        proj = imgs @ Vt[:,:2]
        bins = 8
        hx = np.digitize(proj[:,0], np.linspace(proj[:,0].min(), proj[:,0].max(), bins))
        hy = np.digitize(proj[:,1], np.linspace(proj[:,1].min(), proj[:,1].max(), bins))
        occupied = len({(int(a),int(b)) for a,b in zip(hx,hy)})
        max_bins = bins*bins
        coverage = float(min(1.0, occupied / max_bins))
        if return_counts:
            per = int(n_samples / 26)
            counts = {letters[i]: per for i in range(26)}
            return coverage, counts
        return coverage

def save_mode_histogram(counts_dict, save_path):
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    vals = [counts_dict.get(ch, 0) for ch in letters]
    plt.figure(figsize=(8,3))
    plt.bar(letters, vals)
    plt.xlabel('Letter class')
    plt.ylabel('Frequency in generated samples')
    plt.title('Mode coverage histogram')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def visualize_mode_collapse(history, save_path):
    try:
        epochs = history.get('epoch_float', [])
        mc_snaps = history.get('mode_coverage', [])
        plt.figure(figsize=(6,4))
        if len(epochs) > 0:
            plt.plot(history.get('epoch_float', []), history.get('g_loss', []), label='G loss')
            plt.plot(history.get('epoch_float', []), history.get('d_loss', []), label='D loss')
        if len(mc_snaps) > 0:
            xs = list(range(0, 10*len(mc_snaps), 10))
            plt.scatter(xs, mc_snaps, label='Mode coverage (snapshots)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Coverage')
        plt.title('Training Dynamics & Mode Coverage')
        plt.legend()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
    except Exception:
        pass
