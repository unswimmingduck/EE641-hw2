"""
Utility to compare vanilla and fixed runs and export a side-by-side figure.
Usage:
    python compare_runs.py --vanilla results --fixed results_fixed --out results/visualizations/vanilla_vs_fixed.png
"""
import argparse
from evaluate import compare_vanilla_fixed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vanilla', type=str, default='results')
    ap.add_argument('--fixed', type=str, default='results_fixed')
    ap.add_argument('--out', type=str, default='results/visualizations/vanilla_vs_fixed.png')
    args = ap.parse_args()
    compare_vanilla_fixed(args.vanilla, args.fixed, args.out)
    print(f"Saved comparison to {args.out}")

if __name__ == '__main__':
    main()
