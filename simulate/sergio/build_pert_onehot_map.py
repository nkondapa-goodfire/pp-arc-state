"""
build_pert_onehot_map.py — Build a pert onehot map from a dataset generation config.

Usage
-----
    uv run python build_pert_onehot_map.py --config generation_configs/dataset_tgt.json
    uv run python build_pert_onehot_map.py --config generation_configs/dataset_ppt.json --output configs/pert_onehot_map_ppt.pt

Output
------
Dict[str, torch.Tensor] saved as a .pt file:
  "non-targeting"        -> zeros(pool_size)
  "SYN_{i:04d}_{label}" -> one-hot at gene i, scaled by pert_strength
"""

import argparse
import json
from pathlib import Path

import torch


def build_map(cfg: dict) -> dict[str, torch.Tensor]:
    pool_size: int = cfg["pool_size"]
    pert_strengths: dict[str, float] = cfg["pert_strengths"]

    pert_map: dict[str, torch.Tensor] = {}
    pert_map["non-targeting"] = torch.zeros(pool_size)

    for gene_idx in range(pool_size):
        for label, strength in pert_strengths.items():
            key = f"SYN_{gene_idx:04d}_{label}"
            vec = torch.zeros(pool_size)
            vec[gene_idx] = strength
            pert_map[key] = vec

    return pert_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset generation JSON config")
    parser.add_argument("--output", default=None, help="Output .pt path (default: configs/pert_onehot_map_<config_stem>.pt)")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = json.load(f)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("configs") / f"pert_onehot_map_{config_path.stem}.pt"

    pert_map = build_map(cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pert_map, out_path)

    print(f"Saved {len(pert_map)} entries ({cfg['pool_size']}-dim) -> {out_path}")


if __name__ == "__main__":
    main()
