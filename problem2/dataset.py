import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

class DrumPatternDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir: Path like 'data/drums'
            split: 'train' or 'val'
        """
        self.data_dir = data_dir
        self.split = split

        npz_path = os.path.join(data_dir, 'patterns.npz')
        meta_path = os.path.join(data_dir, 'patterns.json')

        if not os.path.exists(npz_path):
            npz_path = os.path.join(data_dir, 'drum_patterns.npz')

        data = np.load(npz_path, allow_pickle=True)

        if split == 'train':
            if 'train_patterns' in data and 'train_styles' in data:
                self.patterns = data['train_patterns']
                self.styles = np.array(data['train_styles'])
            else:
                self.patterns = data['patterns']
                self.styles = data['styles']
        else:
            if 'val_patterns' in data and 'val_styles' in data:
                self.patterns = data['val_patterns']
                self.styles = np.array(data['val_styles'])
            else:
                n = len(data['patterns'])
                n_train = int(0.8 * n)
                self.patterns = data['patterns'][n_train:]
                self.styles = data['styles'][n_train:]

        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.instrument_names = meta.get('instruments', [f'ch{i}' for i in range(9)])
            self.style_names = meta.get('styles', [f'style{i}' for i in range(5)])
        else:
            self.instrument_names = [f'ch{i}' for i in range(self.patterns.shape[-1])]
            self.style_names = [f'style{i}' for i in np.unique(self.styles)]

        assert self.patterns.ndim == 3 and self.patterns.shape[1] == 16, "Patterns must be [N,16,9]"

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx].astype(np.float32)
        style = int(self.styles[idx])

        pattern_tensor = torch.from_numpy(pattern)  
        density = float(pattern.sum()) / (16 * 9)

        return pattern_tensor, style, density

    def pattern_to_pianoroll(self, pattern):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        if torch.is_tensor(pattern):
            pattern = pattern.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 5))
        for t in range(16):
            for i in range(9):
                if pattern[t, i] > 0.5:
                    rect = patches.Rectangle((t, i), 1, 1, linewidth=0.8,
                                             edgecolor='black', facecolor='blue')
                    ax.add_patch(rect)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_xticks(range(17))
        ax.set_yticks(range(10))
        ax.set_yticklabels([''] + self.instrument_names)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Instrument')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        return fig
