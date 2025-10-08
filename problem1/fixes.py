import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from pathlib import Path

def _save_grid(samples, path):
    import torch as T, imageio.v2 as imageio, numpy as np
    samples = (samples.clamp(-1,1)+1.0)*0.5
    N = samples.size(0); H=W=28; pad=2; nrow=10; ncol=(N+nrow-1)//nrow
    grid = T.ones(1, ncol*H + pad*(ncol-1), nrow*W + pad*(nrow-1))
    idx=0
    for r in range(ncol):
        for c in range(nrow):
            if idx>=N: break
            h0=r*(H+pad); w0=c*(W+pad)
            grid[:,h0:h0+H,w0:w0+W]=samples[idx]
            idx+=1
    arr=(grid.squeeze(0).cpu().numpy()*255).astype('uint8')
    imageio.imwrite(path, arr)

def train_gan_with_fix(generator, discriminator, data_loader, 
                       num_epochs=100, fix_type='feature_matching', device='cuda', z_dim=100, results_dir='results'):
    """
    Train GAN with mode collapse mitigation techniques.
    Supports: 'feature_matching' (implemented).
    (Unrolled/minibatch placeholders left as future work.)
    Returns history dict.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir)/'visualizations').mkdir(parents=True, exist_ok=True)
    history = defaultdict(list)

    g_opt = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5,0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5,0.999))
    bce = nn.BCELoss()

    for epoch in range(num_epochs):
        for i, (real, _) in enumerate(data_loader):
            real = real.to(device)*2.0 - 1.0
            bsz = real.size(0)
            real_y = torch.ones(bsz,1,device=device)
            fake_y = torch.zeros(bsz,1,device=device)

            # ---- D update ----
            d_opt.zero_grad()
            p_real = discriminator(real)
            d_loss_real = bce(p_real, real_y)

            z = torch.randn(bsz, getattr(generator, 'z_dim', z_dim), device=device)
            fake = generator(z).detach()
            p_fake = discriminator(fake)
            d_loss_fake = bce(p_fake, fake_y)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_opt.step()

            g_opt.zero_grad()
            z = torch.randn(bsz, getattr(generator, 'z_dim', z_dim), device=device)
            fake = generator(z)

            with torch.no_grad():
                real_feat = discriminator.extract_features(real).mean(dim=0)
            fake_feat = discriminator.extract_features(fake).mean(dim=0)
            fm_loss = torch.mean((fake_feat - real_feat)**2)

            adv = bce(discriminator(fake), real_y)
            g_loss = 0.5*fm_loss + 0.5*adv
            g_loss.backward()
            g_opt.step()

            if i % 10 == 0:
                history['d_loss'].append(float(d_loss.item()))
                history['g_loss'].append(float(g_loss.item()))
                history['epoch_float'].append(epoch + i/len(data_loader))

        if (epoch+1) in [10,30,50,100] or epoch==0:
            with torch.no_grad():
                z = torch.randn(100, getattr(generator, 'z_dim', z_dim), device=device)
                samples = generator(z).cpu()
                _save_grid(samples, f"{results_dir}/visualizations/fixed_generated_grid_epoch_{epoch+1}.png")

    return dict(history)
