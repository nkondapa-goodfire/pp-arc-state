"""
build_ablation_dirs.py — Create symlink directories and TOML configs for each ablation run.

Reads manifest.csv from the merged dataset, applies a filter per ablation, symlinks matching
files into data/sergio_synthetic/ablations/{run_name}/, and writes configs/ablations/{run_name}.toml.

Usage:
    uv run python scripts/build_ablation_dirs.py \
        --merged-dir data/sergio_synthetic/mini_merged \
        --ablation-root data/sergio_synthetic/ablations \
        --config-root configs/ablations \
        --base-toml configs/sergio_mini_train.toml
"""

import argparse
import pathlib
import pandas as pd


HOME = "/mnt/polished-lake/home/nkondapaneni"
SERGIO_DIR = f"{HOME}/state/simulate/sergio"

# ---------------------------------------------------------------------------
# Ablation definitions: name -> filter function on manifest DataFrame
# ---------------------------------------------------------------------------
ABLATIONS = {
    "baseline_all": lambda df: df,
    "ablation_ba_only":               lambda df: df[df.grn_type == "BA"],
    "ablation_ba_plus_er":            lambda df: df[df.grn_type.isin(["BA", "ER"])],
    "ablation_kd_only":               lambda df: df[df.pert_type.str.startswith("KD")],
    "ablation_kd_plus_ko":            lambda df: df[
        df.pert_type.str.startswith("KD") | (df.pert_type == "KO")
    ],
    "ablation_noise_high":            lambda df: df[df.noise_label == "high"],
    "ablation_noise_high_plus_clean": lambda df: df,   # all noise levels
    "ablation_size_100":              lambda df: df[df.grn_size == 100],
    "ablation_size_100_plus_010":     lambda df: df[df.grn_size.isin([100, 10])],
}


def write_toml(run_name: str, ablation_dir: pathlib.Path, config_path: pathlib.Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f'[datasets]\n'
        f'{run_name} = "{ablation_dir}"\n\n'
        f'[training]\n'
        f'{run_name} = "train"\n\n'
        f'[zeroshot]\n'
        f'"{run_name}.bin_4" = "val"\n'
    )


def build_ablation(
    run_name: str,
    filtered: pd.DataFrame,
    merged_dir: pathlib.Path,
    ablation_root: pathlib.Path,
    config_root: pathlib.Path,
) -> None:
    ablation_dir = ablation_root / run_name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    created = skipped = 0
    for _, row in filtered.iterrows():
        src = (merged_dir / row.merged_file).resolve()
        dst = ablation_dir / row.merged_file
        if dst.exists():
            skipped += 1
        else:
            dst.symlink_to(src)
            created += 1

    config_path = config_root / f"{run_name}.toml"
    write_toml(run_name, ablation_dir, config_path)

    print(f"  {run_name}: {len(filtered)} files ({created} new symlinks, {skipped} existing) → {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir",    default=f"{SERGIO_DIR}/data/sergio_synthetic/mini_merged")
    parser.add_argument("--ablation-root", default=f"{SERGIO_DIR}/data/sergio_synthetic/ablations")
    parser.add_argument("--config-root",   default=f"{SERGIO_DIR}/configs/ablations")
    parser.add_argument("--only",          help="Build only this ablation (default: all)")
    args = parser.parse_args()

    merged_dir    = pathlib.Path(args.merged_dir).resolve()
    ablation_root = pathlib.Path(args.ablation_root)
    config_root   = pathlib.Path(args.config_root)

    manifest = pd.read_csv(merged_dir / "manifest.csv")
    print(f"Manifest: {len(manifest)} rows from {merged_dir}")

    ablations = ABLATIONS
    if args.only:
        ablations = {args.only: ABLATIONS[args.only]}

    for run_name, filter_fn in ablations.items():
        filtered = filter_fn(manifest).reset_index(drop=True)
        build_ablation(run_name, filtered, merged_dir, ablation_root, config_root)

    print(f"\nDone. TOML configs written to {config_root}/")


if __name__ == "__main__":
    main()
