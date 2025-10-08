import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image

@torch.no_grad()
def interpolation_experiment(generator, device, save_dir='results/visualizations'):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    z0 = torch.randn(1, getattr(generator, 'z_dim', 100), device=device)
    z1 = torch.randn(1, getattr(generator, 'z_dim', 100), device=device)
    alphas = torch.linspace(0, 1, steps=16, device=device).view(-1,1)
    z = (1-alphas)*z0 + alphas*z1
    imgs = generator(z).clamp(-1,1).cpu().numpy()
    imgs = (imgs + 1.0) * 0.5
    H=W=28; pad=2
    canvas = np.ones((H, 16*W + (16-1)*pad), dtype=np.float32)
    for i,im in enumerate(imgs):
        r = im[0]; x0 = i*(W+pad); canvas[:, x0:x0+W] = r
    plt.figure(figsize=(8,2))
    plt.imshow(canvas, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/interpolation.png", dpi=200)
    plt.close()

@torch.no_grad()
def style_consistency_experiment(conditional_generator, device, save_dir='results/visualizations'):
    if not getattr(conditional_generator, 'conditional', False):
        return
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    z = torch.randn(1, getattr(conditional_generator, 'z_dim', 100), device=device).repeat(26,1)
    eye = torch.eye(26, device=device)
    imgs = conditional_generator(z, class_label=eye).clamp(-1,1).cpu().numpy()
    imgs = (imgs+1.0)*0.5
    H=W=28; pad=2
    nrow=13; ncol=2
    canvas = np.ones((ncol*H + (ncol-1)*pad, nrow*W + (nrow-1)*pad), dtype=np.float32)
    for i,im in enumerate(imgs):
        r = i//nrow; c = i % nrow
        h0 = r*(H+pad); w0 = c*(W+pad)
        canvas[h0:h0+H, w0:w0+W] = im[0]
    plt.figure(figsize=(10,3))
    plt.imshow(canvas, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/style_consistency.png", dpi=200)
    plt.close()

def mode_recovery_experiment(generator_checkpoints, device, save_path='results/visualizations/mode_recovery.png'):
    if not generator_checkpoints:
        return
    xs = sorted(generator_checkpoints.keys())
    ys = [generator_checkpoints[x] for x in xs]
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Estimated Mode Coverage')
    plt.title('Mode coverage over checkpoints')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def compare_vanilla_fixed(vanilla_dir='results', fixed_dir='results_fixed', out_path='results/visualizations/vanilla_vs_fixed.png'):
    Path(Path(out_path).parent).mkdir(parents=True, exist_ok=True)
    def load_hist(d):
        p = Path(d) / 'training_log.json'
        if p.exists():
            with open(p, 'r') as f:
                return json.load(f)
        return {}
    hv = load_hist(vanilla_dir)
    hf = load_hist(fixed_dir)

    fig = plt.figure(figsize=(10,6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1,1.2])

    ax0 = fig.add_subplot(gs[0, :])
    xv = list(range(0, 10*len(hv.get('mode_coverage', [])), 10))
    xf = list(range(0, 10*len(hf.get('mode_coverage', [])), 10))
    ax0.plot(xv, hv.get('mode_coverage', []), marker='o', label='vanilla')
    ax0.plot(xf, hf.get('mode_coverage', []), marker='s', label='fixed')
    ax0.set_xlabel('Epoch (snapshot every 10)')
    ax0.set_ylabel('Mode coverage')
    ax0.set_title('Mode coverage: vanilla vs fixed')
    ax0.legend()

    def pick_last_grid(d, prefix):
        vis = Path(d) / 'visualizations'
        if not vis.exists():
            return None
        candidates = [f"{prefix}_epoch_100.png", f"{prefix}_epoch_50.png", f"{prefix}_epoch_30.png", f"{prefix}_epoch_10.png"]
        for c in candidates:
            p = vis / c
            if p.exists():
                return p
        return None

    v_img = pick_last_grid(vanilla_dir, 'generated_grid')
    f_img = pick_last_grid(fixed_dir, 'fixed_generated_grid')

    ax1 = fig.add_subplot(gs[1,0])
    if v_img and Path(v_img).exists():
        ax1.imshow(Image.open(v_img), cmap='gray')
        ax1.set_title('Vanilla - latest grid')
    else:
        ax1.text(0.5, 0.5, 'No grid found', ha='center')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1,1])
    if f_img and Path(f_img).exists():
        ax2.imshow(Image.open(f_img), cmap='gray')
        ax2.set_title('Fixed (Feature Matching) - latest grid')
    else:
        ax2.text(0.5, 0.5, 'No grid found', ha='center')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
