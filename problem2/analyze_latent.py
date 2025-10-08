import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def _to_numpy(t):
    return t.detach().cpu().numpy()

def visualize_latent_hierarchy(model, data_loader, results_dir='results/latent_analysis', device='cuda'):
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    zs_high, zs_low, labels = [], [], []
    with torch.no_grad():
        for x, y, _ in data_loader:
            x = x.to(device)
            recon, mu_l, logvar_l, mu_h, logvar_h = model(x)
            z_low  = mu_l  
            z_high = mu_h
            zs_low.append(z_low)
            zs_high.append(z_high)
            labels.append(y)
    zs_low  = torch.cat(zs_low).cpu().numpy()
    zs_high = torch.cat(zs_high).cpu().numpy()
    labels  = torch.cat([torch.as_tensor(l) for l in labels]).cpu().numpy()

    tsne = TSNE(n_components=2, init='random', learning_rate='auto', perplexity=30, max_iter=1000)
    emb = tsne.fit_transform(zs_high)
    plt.figure(figsize=(6,5))
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(emb[idx,0], emb[idx,1], s=12, label=f'style {c}', alpha=0.7)
    plt.legend()
    plt.title('t-SNE of z_high (style)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'tsne_z_high.png'), dpi=180)

    # Save arrays for report
    np.save(os.path.join(results_dir, 'z_high.npy'), zs_high)
    np.save(os.path.join(results_dir, 'z_low.npy'),  zs_low)
    np.save(os.path.join(results_dir, 'labels.npy'), labels)

def interpolate_styles(model, pattern1, pattern2, n_steps=10, results_dir='results/generated_patterns', device='cuda'):
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        x1 = pattern1.unsqueeze(0).to(device)  
        x2 = pattern2.unsqueeze(0).to(device)
        _, mu_l1, _, mu_h1, _ = model(x1)
        _, mu_l2, _, mu_h2, _ = model(x2)

        outs = []
        for i in range(n_steps):
            a = i / (n_steps - 1)
            z_h = (1 - a) * mu_h1 + a * mu_h2
            z_l = (1 - a) * mu_l1 + a * mu_l2
            logits = model.decode_hierarchy(z_h, z_l, temperature=0.9)
            probs = torch.sigmoid(logits)
            outs.append(probs.squeeze(0).cpu().numpy())

    arr = np.stack(outs, axis=0)  
    np.save(os.path.join(results_dir, 'interpolation.npy'), arr)

    dens = arr.sum(axis=-1)  
    plt.figure(figsize=(8,3))
    for i in range(len(dens)):
        plt.plot(dens[i], alpha=0.5)
    plt.title('Interpolation densities over time')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'interpolation_density.png'), dpi=180)

def measure_disentanglement(model, data_loader, results_dir='results/latent_analysis', device='cuda'):
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    ZH, ZL, Y = [], [], []
    with torch.no_grad():
        for x, y, _ in data_loader:
            x = x.to(device)
            _, mu_l, _, mu_h, _ = model(x)
            ZH.append(mu_h.cpu())
            ZL.append(mu_l.cpu())
            Y.append(y)
    ZH = torch.cat(ZH).numpy()
    ZL = torch.cat(ZL).numpy()
    Y  = torch.cat(Y).numpy()

    overall_var_high = ZH.var(axis=0).mean()
    overall_var_low  = ZL.var(axis=0).mean()

    within_high = []
    within_low  = []
    for c in np.unique(Y):
        idx = (Y == c)
        within_high.append(ZH[idx].var(axis=0).mean())
        within_low.append(ZL[idx].var(axis=0).mean())

    metrics = {
        'z_high_overall_var': float(overall_var_high),
        'z_low_overall_var':  float(overall_var_low),
        'z_high_within_var_mean': float(np.mean(within_high)),
        'z_low_within_var_mean':  float(np.mean(within_low)),
        'disentangle_score': float(np.mean(within_high) / (overall_var_high + 1e-8))
    }
    np.savez(os.path.join(results_dir, 'disentanglement_metrics.npz'), **metrics)
    return metrics

def controllable_generation(model, genre_labels, n_per_genre=10, temperature=0.9, results_dir='results/generated_patterns', device='cuda'):
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        genres = sorted(set(genre_labels))
        all_out = []
        for gid in genres:
            proto = torch.zeros(1, model.z_high_dim, device=device)
            idx = gid % model.z_high_dim
            proto[0, idx] = 1.0  
            logits = model.decode_hierarchy(proto, z_low=None, temperature=temperature)
            probs = torch.sigmoid(logits)
            samp = (probs > 0.5).float().cpu().numpy()
            all_out.append(samp)
        arr = np.concatenate(all_out, axis=0)  
        np.save(os.path.join(results_dir, 'controllable_generation.npy'), arr)
        return arr

import json

def compute_style_means(model, data_loader, device='cuda'):
    model.eval()
    sums = {}
    counts = {}
    with torch.no_grad():
        for x, y, _ in data_loader:
            x = x.to(device)
            _, _, _, mu_h, _ = model(x)  
            for i, sid in enumerate(y.tolist()):
                sums[sid] = sums.get(sid, 0) + mu_h[i].detach().cpu()
                counts[sid] = counts.get(sid, 0) + 1
    means = {sid: (sums[sid] / counts[sid]).unsqueeze(0) for sid in sums}
    return means  

def sample_10_per_style(model, style_means, n_per_style=10, temperature=0.9,
                        results_dir='results/generated_patterns', device='cuda'):
    os.makedirs(results_dir, exist_ok=True)
    out = {}
    model.eval()
    with torch.no_grad():
        for sid, z_high_mean in style_means.items():
            z_high = z_high_mean.to(device).repeat(n_per_style, 1)  
            logits = model.decode_hierarchy(z_high, z_low=None, temperature=temperature)
            probs = torch.sigmoid(logits)
            samples = (probs > 0.5).float().cpu().numpy()
            out[int(sid)] = samples
            np.save(os.path.join(results_dir, f'style_{int(sid)}_samples.npy'), samples)
    np.savez(os.path.join(results_dir, 'samples_10_per_style.npz'), **{f'style_{k}': v for k, v in out.items()})
    return out

def style_transfer_examples(model, data_loader, style_means, k_per_style=2, temperature=0.9,
                            results_dir='results/generated_patterns', device='cuda'):
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    by_style = {}
    with torch.no_grad():
        for x, y, _ in data_loader:
            for i, sid in enumerate(y.tolist()):
                if sid not in by_style: by_style[sid] = []
                if len(by_style[sid]) < k_per_style:
                    by_style[sid].append(x[i].clone())
            if all(len(v) >= k_per_style for v in by_style.values()) and len(by_style) >= len(style_means):
                break

    results = {}
    with torch.no_grad():
        for src_sid, patterns in by_style.items():
            for j, pat in enumerate(patterns):
                x = pat.unsqueeze(0).to(device)  
                _, mu_l, _, mu_h_src, _ = model(x)
                z_low = mu_l 
                for tgt_sid, z_high_mean in style_means.items():
                    if int(tgt_sid) == int(src_sid): 
                        continue
                    z_high = z_high_mean.to(device)
                    logits = model.decode_hierarchy(z_high, z_low, temperature=temperature)
                    probs = torch.sigmoid(logits)
                    sample = (probs > 0.5).float().cpu().numpy()[0]
                    key = f'src{int(src_sid)}_{j}_to_{int(tgt_sid)}'
                    results[key] = sample
    np.savez(os.path.join(results_dir, 'style_transfer_examples.npz'), **results)
    return results

def interpret_latent_dimensions(model, base_pattern, sweep_std=3.0, steps=7, which='both',
                                results_dir='results/latent_analysis', device='cuda'):
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        x = base_pattern.unsqueeze(0).to(device)
        _, mu_l, logvar_l, mu_h, logvar_h = model(x)
        std_h = torch.exp(0.5 * logvar_h)
        std_l = torch.exp(0.5 * logvar_l)
        grids = {}
        def sweep(z_base, std, tag):
            D = z_base.shape[-1]
            for d in range(D):
                vals = torch.linspace(-sweep_std, sweep_std, steps, device=device)
                outs = []
                for v in vals:
                    z = z_base.clone()
                    z[0, d] = z_base[0, d] + v * std[0, d]
                    if tag == 'z_high':
                        logits = model.decode_hierarchy(z, mu_l, temperature=0.9)
                    else:
                        logits = model.decode_hierarchy(mu_h, z, temperature=0.9)
                    probs = torch.sigmoid(logits)
                    outs.append((probs > 0.5).float().cpu().numpy()[0])
                grids[f'{tag}_dim_{d}'] = np.stack(outs, axis=0) 
        if which in ('both', 'high'):
            sweep(mu_h, std_h, 'z_high')
        if which in ('both', 'low'):
            sweep(mu_l, std_l, 'z_low')
    np.savez(os.path.join(results_dir, 'dimension_interpretation.npz'), **grids)
    with open(os.path.join(results_dir, 'dimension_interpretation_index.json'), 'w') as f:
        json.dump({'dims': list(grids.keys()), 'steps': steps}, f, indent=2)
    return grids
