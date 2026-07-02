"""Hyperparameters for the model, as one flat dataclass.

Every dimension in the network lives here so there are no magic numbers inside
module bodies — a module reads its sizes off the config it's handed. Use
``Config()`` for full-size runs and ``SMALL`` for fast fake-data tests, where the
tiny dims make a forward+backward pass run in milliseconds.

    from boltz2.config import Config, SMALL

    model = Boltz2(Config())     # real dims
    model = Boltz2(SMALL)        # tiny dims for tests / overfitting fake data
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    # --- core representations -------------------------------------------------
    c_s: int = 384          # single (per-token) representation channels
    c_z: int = 128          # pair (per-token-pair) representation channels
    c_token: int = 768      # token channels inside the diffusion transformer
    c_atom: int = 128       # per-atom representation channels
    c_atompair: int = 16    # atom-pair representation channels

    # --- input / features -----------------------------------------------------
    n_token_types: int = 33     # amino acids + nucleotides + gap/unknown/ligand
    rel_pos_max: int = 32       # clamp range for relative-position encoding
    n_msa: int = 128            # number of MSA rows kept per example
    c_msa: int = 64             # MSA representation channels

    # --- trunk: Pairformer ----------------------------------------------------
    n_pairformer_blocks: int = 48
    n_recycles: int = 3
    c_hidden_tri_mul: int = 128     # hidden channels in triangle multiplication
    c_hidden_tri_attn: int = 32     # per-head channels in triangle attention
    n_heads_tri: int = 4            # heads in triangle attention
    n_heads_single: int = 16        # heads in the single-attention-with-pair-bias
    transition_n: int = 4           # expansion factor in Transition MLPs

    # --- atom attention (encoder/decoder) ------------------------------------
    atom_attn_query_size: int = 32  # atoms per query block (local window)
    atom_attn_key_size: int = 128   # atoms visible to each query block
    n_atom_attn_blocks: int = 3
    n_heads_atom: int = 4

    # --- diffusion module -----------------------------------------------------
    n_diffusion_blocks: int = 24
    n_heads_diffusion: int = 16
    sigma_data: float = 16.0        # EDM data scale (roughly the coord std)
    # EDM training noise distribution: log-sigma ~ Normal(p_mean, p_std)
    p_mean: float = -1.2
    p_std: float = 1.5
    # EDM sampling schedule
    n_sampling_steps: int = 200
    sigma_min: float = 4e-4
    sigma_max: float = 160.0
    rho: float = 7.0                # schedule curvature

    # --- heads ----------------------------------------------------------------
    n_bins_distogram: int = 64
    n_bins_pae: int = 64
    n_bins_pde: int = 64
    n_bins_plddt: int = 50
    dist_min: float = 2.0           # distogram/PAE binning range (Angstrom)
    dist_max: float = 22.0

    # --- affinity head --------------------------------------------------------
    c_affinity: int = 256
    n_affinity_blocks: int = 4


# Tiny preset for fake-data tests and overfitting: same code, milliseconds/step.
SMALL = replace(
    Config(),
    c_s=16,
    c_z=8,
    c_token=32,
    c_atom=16,
    c_atompair=8,
    n_msa=4,
    c_msa=8,
    n_pairformer_blocks=2,
    n_recycles=1,
    c_hidden_tri_mul=8,
    c_hidden_tri_attn=4,
    n_heads_tri=2,
    n_heads_single=2,
    atom_attn_query_size=4,
    atom_attn_key_size=8,
    n_atom_attn_blocks=1,
    n_heads_atom=2,
    n_diffusion_blocks=2,
    n_heads_diffusion=2,
    n_sampling_steps=8,
    n_affinity_blocks=1,
    n_bins_distogram=16,
    n_bins_pae=16,
    n_bins_pde=16,
    n_bins_plddt=10,
)
