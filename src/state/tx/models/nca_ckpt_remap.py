"""
Remap an NCA pre-pretraining checkpoint to the STATE transformer_backbone key namespace.

NCA checkpoints (CustomLlamaModel) store backbone weights at:
    layers.{i}.*  /  norm.weight

STATE's StateTransitionModel stores the backbone as self.transformer_backbone, so
the full-model state_dict uses:
    transformer_backbone.layers.{i}.*  /  transformer_backbone.norm.weight

Keys that exist only in NCA (input_proj, output_proj, wpe) are dropped — STATE
uses its own input encoder and decoder.

Usage:
    python -m state.tx.models.nca_ckpt_remap \\
        --nca_ckpt  /path/to/model_N.pth \\
        --out       /path/to/state_backbone_init.pt \\
        [--verify]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch


# Keys present in NCA but not relevant to STATE's backbone
_NCA_ONLY_PREFIXES = ("input_proj.", "output_proj.", "wpe.")


def remap_nca_to_state(
    nca_state_dict: dict[str, torch.Tensor],
    backbone_prefix: str = "transformer_backbone",
) -> dict[str, torch.Tensor]:
    """
    Remap NCA CustomLlamaModel state_dict keys to STATE's transformer_backbone namespace.

    Args:
        nca_state_dict: ``ckpt["model"]`` from an NCA .pth checkpoint.
        backbone_prefix: attribute name of the backbone inside StateTransitionModel
                         (default: ``"transformer_backbone"``).

    Returns:
        Remapped state dict ready for ``model.load_state_dict(..., strict=False)``.
    """
    remapped: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, tensor in nca_state_dict.items():
        if any(key.startswith(p) for p in _NCA_ONLY_PREFIXES):
            skipped.append(key)
            continue
        remapped[f"{backbone_prefix}.{key}"] = tensor

    return remapped, skipped


def verify_against_backbone(
    remapped: dict[str, torch.Tensor],
    backbone_kwargs: Optional[dict] = None,
) -> bool:
    """
    Instantiate a STATE LlamaBidirectionalModel with Replogle defaults and
    verify every remapped key exists with the correct shape.

    Returns True if all keys match, False otherwise.
    """
    try:
        from state.tx.models.utils import get_transformer_backbone
    except ImportError:
        print("WARNING: could not import state — skipping shape verification.", file=sys.stderr)
        return True

    defaults = dict(
        bidirectional_attention=True,
        max_position_embeddings=64,
        hidden_size=328,
        intermediate_size=3072,
        num_hidden_layers=8,
        num_attention_heads=12,
        num_key_value_heads=12,
        head_dim=64,
        use_cache=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        layer_norm_eps=1e-6,
        rotary_dim=0,
        use_rotary_embeddings=False,
    )
    if backbone_kwargs:
        defaults.update(backbone_kwargs)

    backbone, _ = get_transformer_backbone("llama", defaults)
    backbone_sd = backbone.state_dict()

    # Remap keys back to bare backbone namespace for comparison
    prefix = next(iter(remapped)).split(".")[0] + "."  # e.g. "transformer_backbone."
    bare = {k[len(prefix):]: v for k, v in remapped.items()}

    ok = True
    for key, tensor in bare.items():
        if key not in backbone_sd:
            print(f"  MISSING in STATE backbone : {key}")
            ok = False
        elif backbone_sd[key].shape != tensor.shape:
            print(f"  SHAPE MISMATCH : {key}  NCA={tuple(tensor.shape)}  STATE={tuple(backbone_sd[key].shape)}")
            ok = False
        else:
            print(f"  OK  {key:60s}  {tuple(tensor.shape)}")

    # Report STATE keys not covered by remapping
    uncovered = [k for k in backbone_sd if k not in bare]
    if uncovered:
        print(f"\n  STATE backbone keys not in NCA checkpoint (will keep random init):")
        for k in uncovered:
            print(f"    {k}")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Remap NCA checkpoint → STATE backbone init")
    parser.add_argument("--nca_ckpt", required=True, type=Path,
                        help="Path to NCA .pth checkpoint (e.g. model_100.pth)")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output path for remapped state dict (.pt)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify remapped keys against the Replogle backbone config")
    parser.add_argument("--backbone_prefix", default="transformer_backbone",
                        help="Attribute name of the backbone in StateTransitionModel")
    args = parser.parse_args()

    print(f"Loading NCA checkpoint: {args.nca_ckpt}")
    ckpt = torch.load(args.nca_ckpt, map_location="cpu", weights_only=False)
    nca_sd = ckpt["model"]
    print(f"  {len(nca_sd)} keys in NCA state dict  (epoch {ckpt.get('epoch', '?')})")

    remapped, skipped = remap_nca_to_state(nca_sd, backbone_prefix=args.backbone_prefix)

    print(f"\nRemapped {len(remapped)} keys, skipped {len(skipped)} NCA-only keys:")
    for k in skipped:
        print(f"  skipped  {k}")

    if args.verify:
        print("\nVerifying shapes against STATE Replogle backbone:")
        ok = verify_against_backbone(remapped)
        if ok:
            print("\nAll keys match.")
        else:
            print("\nVerification FAILED — see above.", file=sys.stderr)
            sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": remapped}, args.out)
    print(f"\nSaved remapped checkpoint to: {args.out}")


if __name__ == "__main__":
    main()
