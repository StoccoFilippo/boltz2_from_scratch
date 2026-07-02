from src.boltz2.config import Config, SMALL
import torch

def fake_complex(n_tokens=8, n_atoms_per_token=3, n_msa=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_atoms = n_tokens * n_atoms_per_token [A]
    return {
        # per-token
        "token_type": torch.randint(0, 20, (n_tokens,), generator=g), [L]
        "residue_index": torch.arange(n_tokens), [L]
        "chain_index": torch.zeros(n_tokens, dtype=torch.long), [L]
        # per-atom
        "atom_coords": torch.randn(n_atoms, 3, generator=g),[A,3]
        "atom_to_token": torch.arange(n_tokens).repeat_interleave(n_atoms_per_token), [L*A]
        "ref_coords": torch.randn(n_atoms, 3, generator=g),[A,3]
        # msa
        "msa": torch.randint(0, 20, (n_msa, n_tokens), generator=g),[n_msa,L]
        # targets (for loss testing)
        "true_coords": torch.randn(n_atoms, 3, generator=g), [A,3] 
    }





