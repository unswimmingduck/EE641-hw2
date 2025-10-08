import argparse
import os
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import FontDataset
from models import Generator, Discriminator
from training_dynamics import train_gan
from fixes import train_gan_with_fix

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=str, default='data/fonts')
    ap.add_argument('--results-dir', type=str, default='results')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-epochs', type=int, default=100)
    ap.add_argument('--z-dim', type=int, default=100)
    ap.add_argument('--experiment', type=str, default='vanilla', choices=['vanilla','fixed'])
    ap.add_argument('--fix-type', type=str, default='feature_matching', choices=['feature_matching'])
    ap.add_argument('--num-workers', type=int, default=2)
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir)/'visualizations').mkdir(parents=True, exist_ok=True)


    train_ds = FontDataset(args.data_dir, split='train_samples')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)


    G = Generator(z_dim=args.z_dim).to(device)
    D = Discriminator().to(device)


    if args.experiment == 'vanilla':
        history = train_gan(G, D, train_loader, num_epochs=args.num_epochs, device=str(device), z_dim=args.z_dim, results_dir=args.results_dir)
    else:
        history = train_gan_with_fix(G, D, train_loader, num_epochs=args.num_epochs, fix_type=args.fix_type, device=str(device), z_dim=args.z_dim, results_dir=args.results_dir)

    with open(f"{args.results_dir}/training_log.json", 'w') as f:
        json.dump(history, f, indent=2)

    torch.save({
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'config': vars(args),
        'final_epoch': args.num_epochs
    }, f"{args.results_dir}/best_generator.pth")

    print(f"Training complete. Results saved to {args.results_dir}/")


if __name__ == '__main__':
    main()
