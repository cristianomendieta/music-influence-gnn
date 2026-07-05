"""Load the Phase 2 grid-search-winning checkpoint as a MusicDiffusionGNN."""
from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN

GRID_BEST_CKPT = Path("results/phase2_experimentos_v2/grid_best_model.pt")


def load_grid_best_model(
    ckpt_path: Path | str = GRID_BEST_CKPT,
    g: HeteroData = None,
    pop_bank_regen: Tensor = None,
    *,
    device: str = "cpu",
) -> MusicDiffusionGNN:
    """Load the BEST Phase 2 grid config (W12_h128_l3_lr5e-04) for Phase 3 eval.

    ``ckpt_path`` must be the wrapper dict produced by the Phase 2 grid search
    (keys: config_str, W, hidden, layers, lr, dropout, val_mse, state_dict) —
    NEVER ``results/phase2_experimentos_v2/best_model.pt``, which is a
    different, weaker checkpoint (raw state_dict only, config W4_h64_l2) and
    must not be loaded by this function.

    The saved ``state_dict`` contains a ``pop_bank`` buffer frozen at Phase 2
    training time; it is dropped and replaced with ``pop_bank_regen`` (a
    freshly built bank over the full weekly timeseries) so eval always uses
    an up-to-date popularity bank. The two tensors are expected to match in
    practice — a value mismatch only warns, but a shape mismatch aborts since
    it would indicate an incompatible graph/timeseries snapshot.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MusicDiffusionGNN(
        g.metadata(),
        n_genre=g["genre"].num_nodes,
        hidden=ck["hidden"],
        layers=ck["layers"],
        dropout=ck["dropout"],
        pop_bank=pop_bank_regen,
    ).to(device)

    sd = ck["state_dict"]
    ckpt_pop_bank = sd.get("pop_bank")
    sd = dict(sd)
    sd.pop("pop_bank", None)

    if ckpt_pop_bank is not None and pop_bank_regen is not None:
        if ckpt_pop_bank.shape != pop_bank_regen.shape:
            raise ValueError(
                f"pop_bank shape mismatch: checkpoint {ckpt_pop_bank.shape} vs "
                f"regenerated {pop_bank_regen.shape} — incompatible graph/timeseries snapshot."
            )
        if not torch.allclose(
            ckpt_pop_bank, pop_bank_regen.to(device=ckpt_pop_bank.device, dtype=ckpt_pop_bank.dtype)
        ):
            warnings.warn(
                "Regenerated pop_bank differs from the checkpoint's frozen pop_bank "
                "(same shape, different values)."
            )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) <= {"pop_bank"}, f"unexpected missing keys: {missing}"
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"

    return model
