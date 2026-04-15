"""Generative latent sampler using Conditional Flow Matching.

The sampler learns a transport map from Gaussian noise to the latent data
distribution, with importance weights proportional to the world-model
transition loss.  This biases the generator towards latent states where
the current dynamics model is *most inaccurate*, guiding acquisition
toward informative PDE solver queries.

Design choices:
    - Flow matching over DDPM: simpler loss (MSE on velocity), fast
      sampling (Euler ODE, 20-50 steps), no noise schedule tuning.
    - Fixed 512-dim compressed latent: a small conv encoder/decoder
      maps the AE latent (variable size) → 512-dim → (16, 32) spatial
      tensor for the velocity network.
    - ConvAttention velocity net: 1D conv blocks capture local structure,
      self-attention captures global correlations.  Time conditioning
      via FiLM, same pattern as the dynamics model.

References:
    - Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
    - DAS-PINN (Tang et al.) – loss-weighted density for PDE sampling
    - KRnet (Tang et al.) – transport maps for scientific computing
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pdewm.data.context_features import metadata_to_context_vector
from pdewm.data.schema import TransitionRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPRESSED_DIM = 512          # fixed compressed latent dimension
COMPRESSED_CHANNELS = 16      # spatial layout: (16, 32)
COMPRESSED_LENGTH = COMPRESSED_DIM // COMPRESSED_CHANNELS  # 32


# ---------------------------------------------------------------------------
# Latent compressor / decompressor
# ---------------------------------------------------------------------------


class LatentCompressor(nn.Module):
    """Map AE latent (C, L) → fixed-size flat vector of dim ``COMPRESSED_DIM``.

    Uses adaptive pooling + 1D conv to handle variable AE latent shapes
    while preserving spatial structure.
    """

    def __init__(self, latent_channels: int, latent_length: int) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_length = latent_length
        # Step 1: adaptive pool spatial to COMPRESSED_LENGTH
        self.pool = nn.AdaptiveAvgPool1d(COMPRESSED_LENGTH)
        # Step 2: 1x1 conv to reduce channels to COMPRESSED_CHANNELS
        self.proj = nn.Conv1d(latent_channels, COMPRESSED_CHANNELS, kernel_size=1)
        self.norm = nn.GroupNorm(1, COMPRESSED_CHANNELS)
        self.act = nn.SiLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, C, L) → (B, COMPRESSED_DIM)"""
        h = self.pool(z)
        h = self.act(self.norm(self.proj(h)))
        return h.reshape(h.shape[0], -1)


class LatentDecompressor(nn.Module):
    """Map flat vector (COMPRESSED_DIM,) → AE latent shape (C, L).

    Uses transposed conv + interpolation to restore the original spatial
    layout.
    """

    def __init__(self, latent_channels: int, latent_length: int) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_length = latent_length
        self.proj = nn.Conv1d(COMPRESSED_CHANNELS, latent_channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, latent_channels)
        self.act = nn.SiLU()

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """(B, COMPRESSED_DIM) → (B, C, L)"""
        h = x_flat.reshape(-1, COMPRESSED_CHANNELS, COMPRESSED_LENGTH)
        h = self.act(self.norm(self.proj(h)))
        # Interpolate spatial dim back to original length
        if h.shape[-1] != self.latent_length:
            h = F.interpolate(h, size=self.latent_length, mode="linear", align_corners=False)
        return h


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    """Map scalar time *t* ∈ [0, 1] to a fixed-width embedding vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ---------------------------------------------------------------------------
# ConvAttention velocity network  v_θ(x_t, t) → velocity
# ---------------------------------------------------------------------------


class FiLMConvBlock(nn.Module):
    """1D residual conv block with FiLM time conditioning."""

    def __init__(self, channels: int, time_dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(1, channels)
        self.film = nn.Linear(time_dim, channels * 2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        gamma, beta = self.film(t_emb).chunk(2, dim=-1)
        h = h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class SpatialSelfAttention(nn.Module):
    """Single-head self-attention over the spatial dimension."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)  # each (B, C, L)
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale  # (B, L, L)
        attn = attn.softmax(dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))  # (B, C, L)
        return x + self.proj(out)


@dataclass(slots=True)
class VelocityNetConfig:
    """Config for the convolutional velocity network."""
    hidden_channels: int = 64
    time_embed_dim: int = 128
    num_conv_blocks: int = 4
    kernel_size: int = 5
    use_attention: bool = True


class ConvAttentionVelocityNet(nn.Module):
    """Velocity field v_θ(x_t, t) using 1D convolutions + self-attention.

    Operates on the spatial tensor (B, COMPRESSED_CHANNELS, COMPRESSED_LENGTH),
    not a flat MLP.  This exploits local correlations in the latent PDE state
    while the attention layer captures global structure.
    """

    def __init__(self, config: VelocityNetConfig) -> None:
        super().__init__()
        self.config = config
        C = config.hidden_channels

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(config.time_embed_dim, C),
            nn.SiLU(),
            nn.Linear(C, C),
        )

        # Input projection
        self.input_proj = nn.Conv1d(COMPRESSED_CHANNELS, C, kernel_size=1)

        # Conv blocks with FiLM conditioning
        self.blocks = nn.ModuleList([
            FiLMConvBlock(C, C, kernel_size=config.kernel_size)
            for _ in range(config.num_conv_blocks)
        ])

        # Self-attention in the middle
        self.use_attention = config.use_attention
        if config.use_attention:
            self.attention = SpatialSelfAttention(C)

        # Output projection → velocity in compressed channel space
        self.output_proj = nn.Conv1d(C, COMPRESSED_CHANNELS, kernel_size=1)

    def forward(self, x_flat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """(B, COMPRESSED_DIM), (B,) → (B, COMPRESSED_DIM) velocity."""
        # Reshape flat → spatial
        x = x_flat.reshape(-1, COMPRESSED_CHANNELS, COMPRESSED_LENGTH)
        t_emb = self.time_proj(self.time_embed(t))

        h = self.input_proj(x)

        # First half of conv blocks
        mid = len(self.blocks) // 2
        for block in self.blocks[:mid]:
            h = block(h, t_emb)

        # Attention in the middle
        if self.use_attention:
            h = self.attention(h)

        # Second half of conv blocks
        for block in self.blocks[mid:]:
            h = block(h, t_emb)

        v = self.output_proj(h)
        return v.reshape(-1, COMPRESSED_DIM)


# ---------------------------------------------------------------------------
# Latent Flow Matching Sampler
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FlowMatchingSamplerConfig:
    # Velocity network
    hidden_channels: int = 64
    time_embed_dim: int = 128
    num_conv_blocks: int = 4
    kernel_size: int = 5
    use_attention: bool = True
    # Training
    training_epochs: int = 200
    learning_rate: float = 1e-3
    batch_size: int = 256
    # Sampling
    ode_steps: int = 30
    # Loss weighting
    temperature: float = 1.0


class LatentFlowMatchingSampler:
    """Train a flow-matching model on compressed latent vectors with loss-based weights.

    The sampler includes a compressor/decompressor pair to map between the
    AE's native latent space and a fixed 512-dim compressed space.

    Usage::

        sampler = LatentFlowMatchingSampler(config, latent_channels, latent_length, device)
        sampler.fit(latents_2d, transition_losses)   # latents_2d: (N, C, L)
        candidates_2d = sampler.sample(n=128)        # returns (N, C, L)
    """

    def __init__(
        self,
        config: FlowMatchingSamplerConfig,
        latent_channels: int,
        latent_length: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.latent_channels = latent_channels
        self.latent_length = latent_length

        # Compressor / decompressor (trained jointly with velocity net)
        self.compressor = LatentCompressor(latent_channels, latent_length).to(device)
        self.decompressor = LatentDecompressor(latent_channels, latent_length).to(device)

        # Velocity network (operates in compressed 512-dim space)
        vel_cfg = VelocityNetConfig(
            hidden_channels=config.hidden_channels,
            time_embed_dim=config.time_embed_dim,
            num_conv_blocks=config.num_conv_blocks,
            kernel_size=config.kernel_size,
            use_attention=config.use_attention,
        )
        self.velocity_net = ConvAttentionVelocityNet(vel_cfg).to(device)

        # Normalisation stats
        self.data_mean: torch.Tensor | None = None
        self.data_std: torch.Tensor | None = None

    def _all_parameters(self):
        """Collect parameters from all sub-modules."""
        yield from self.compressor.parameters()
        yield from self.decompressor.parameters()
        yield from self.velocity_net.parameters()

    # ---- training ----------------------------------------------------------

    def fit(
        self,
        latents_2d: np.ndarray,
        transition_losses: np.ndarray,
        *,
        epochs: int | None = None,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Train the flow model on AE latents weighted by transition losses.

        Parameters
        ----------
        latents_2d : (N, C, L) array — raw AE latent tensors.
        transition_losses : (N,) array of per-sample transition losses.
        epochs : override for ``config.training_epochs``.
        verbose : print training progress.

        Returns
        -------
        dict with average training loss and final loss.
        """
        epochs = epochs or self.config.training_epochs
        data = torch.from_numpy(latents_2d).float().to(self.device)  # (N, C, L)
        n = len(data)

        # Compress to 512-dim
        self.compressor.train()
        self.decompressor.train()
        self.velocity_net.train()

        # Build importance weights  p(i) ∝ loss_i^temperature
        losses_t = torch.from_numpy(transition_losses).float().to(self.device)
        if self.config.temperature > 0:
            weights = losses_t.pow(self.config.temperature)
            weights = weights / weights.sum()
        else:
            weights = torch.ones(n, device=self.device) / n

        optimizer = torch.optim.AdamW(
            list(self._all_parameters()),
            lr=self.config.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        batch_size = min(self.config.batch_size, n)
        total_loss = 0.0
        last_loss = 0.0

        for epoch in range(1, epochs + 1):
            # Sample indices with importance weights
            indices = torch.multinomial(weights, batch_size, replacement=True)
            batch_latents = data[indices]  # (B, C, L)

            # Compress → 512-dim
            compressed = self.compressor(batch_latents)  # (B, 512)

            # Normalise (compute running stats on first epoch)
            if epoch == 1:
                with torch.no_grad():
                    all_compressed = self.compressor(data)
                    self.data_mean = all_compressed.mean(dim=0).detach()
                    self.data_std = all_compressed.std(dim=0).clamp(min=1e-6).detach()

            x_1 = (compressed - self.data_mean) / self.data_std  # normalised data
            x_0 = torch.randn_like(x_1)  # noise source

            # OT interpolation
            t = torch.rand(batch_size, device=self.device)
            x_t = (1.0 - t[:, None]) * x_0 + t[:, None] * x_1

            # Target velocity (conditional vector field)
            u_t = x_1 - x_0

            # Predict velocity
            v_pred = self.velocity_net(x_t, t)
            flow_loss = F.mse_loss(v_pred, u_t)

            # Reconstruction regularisation: ensure compressor/decompressor
            # form a reasonable autoencoder pair
            reconstructed = self.decompressor(compressed)
            recon_loss = F.mse_loss(reconstructed, batch_latents)

            loss = flow_loss + 0.1 * recon_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self._all_parameters()), 1.0)
            optimizer.step()
            scheduler.step()

            loss_val = float(loss.detach().cpu())
            total_loss += loss_val
            last_loss = loss_val

            if verbose and epoch % max(1, epochs // 10) == 0:
                print(
                    f"  [FlowMatching] epoch {epoch}/{epochs}  "
                    f"flow={float(flow_loss.detach()):.6f}  recon={float(recon_loss.detach()):.6f}"
                )

        return {"mean_loss": total_loss / epochs, "final_loss": last_loss}

    # ---- sampling ----------------------------------------------------------

    @torch.no_grad()
    def sample(self, n: int, *, ode_steps: int | None = None) -> np.ndarray:
        """Generate *n* candidate latent tensors via Euler ODE integration.

        Returns
        -------
        (n, C, L) numpy array of latent tensors in AE space.
        """
        ode_steps = ode_steps or self.config.ode_steps
        self.velocity_net.eval()
        self.decompressor.eval()

        dt = 1.0 / ode_steps
        x = torch.randn(n, COMPRESSED_DIM, device=self.device)

        for step in range(ode_steps):
            t_val = step * dt
            t = torch.full((n,), t_val, device=self.device)
            v = self.velocity_net(x, t)
            x = x + dt * v

        # Un-normalise
        if self.data_mean is not None and self.data_std is not None:
            x = x * self.data_std + self.data_mean

        # Decompress back to AE latent shape (N, C, L)
        z = self.decompressor(x)
        return z.cpu().numpy()


# ---------------------------------------------------------------------------
# Transition loss computation
# ---------------------------------------------------------------------------


def compute_transition_losses(
    records: list[TransitionRecord],
    committee: list[tuple[Any, Any, dict[str, Any]]],
    *,
    device: torch.device,
    rollout_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-record transition losses using the ensemble committee.

    For each record, the loss is the *mean across ensemble members* of the
    MSE between the decoded prediction and the ground-truth next state.

    Returns
    -------
    latents_2d : (N, C, L) AE latent tensors for each record.
    losses : (N,) per-record mean transition loss.
    """
    latents_list: list[np.ndarray] = []
    losses_list: list[float] = []

    for record in records:
        state = torch.from_numpy(record.state).unsqueeze(0).to(device)
        next_state = torch.from_numpy(record.next_state).unsqueeze(0).to(device)
        metadata_dict = record.metadata.to_dict()

        member_losses: list[float] = []
        latent_np: np.ndarray | None = None

        for model, autoencoder, model_metadata in committee:
            with torch.no_grad():
                z = autoencoder.encode(state)
                if latent_np is None:
                    latent_np = z[0].detach().cpu().numpy().astype(np.float32)

                # Build context
                context_features = metadata_to_context_vector(
                    metadata_dict, model_metadata["context_features"]
                )
                context = torch.from_numpy(context_features).unsqueeze(0).to(device)
                pde_index = model_metadata["pde_to_index"][metadata_dict["pde_id"]]
                pde_ids = torch.tensor([pde_index], device=device, dtype=torch.long)

                # Forward dynamics
                z_pred = z
                for _ in range(rollout_steps):
                    z_pred = model(z_pred, pde_ids, context)

                # Decode and compare
                u_pred = autoencoder.decode(z_pred)
                loss = float(F.mse_loss(u_pred, next_state).cpu())
                member_losses.append(loss)

        assert latent_np is not None
        latents_list.append(latent_np)
        losses_list.append(float(np.mean(member_losses)))

    return np.stack(latents_list), np.array(losses_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Candidate generation from flow samples
# ---------------------------------------------------------------------------


def build_generative_candidates(
    sampler: LatentFlowMatchingSampler,
    committee: list[tuple[Any, Any, dict[str, Any]]],
    memory_latents_flat: np.ndarray,
    *,
    n_candidates: int,
    device: torch.device,
    alpha: float = 1.0,
    beta: float = 0.2,
    gamma: float = 0.5,
) -> list[Any]:
    """Generate ``CandidateState`` objects from the flow matching sampler.

    Each generated latent is decoded to physical space via the AE, then
    scored with uncertainty / novelty / risk — same signals as heuristic
    candidates for a fair comparison.
    """
    from pdewm.acquisition.heuristic import CandidateState

    # Sample latents from the flow model: (n, C, L)
    raw_latents = sampler.sample(n_candidates)
    model, autoencoder, model_metadata = committee[0]
    metadata_template = _build_minimal_metadata(model_metadata)

    candidates: list[CandidateState] = []
    # Pre-compute amplitude reference from memory bank
    mem_amplitude = float(np.max(np.abs(memory_latents_flat))) + 1e-6

    for i in range(n_candidates):
        z = torch.from_numpy(raw_latents[i : i + 1]).float().to(device)  # (1, C, L)

        # Decode to physical space
        with torch.no_grad():
            u = autoencoder.decode(z)
        candidate_state = u[0].cpu().numpy()

        # Re-encode for clean latent summary
        with torch.no_grad():
            z_clean = autoencoder.encode(u)
        latent_summary = z_clean[0].detach().cpu().reshape(-1).numpy().astype(np.float32)

        # Committee uncertainty
        uncertainty = _generative_committee_uncertainty(
            candidate_state, metadata_template, committee, device
        )

        # Novelty vs memory bank
        distances = np.linalg.norm(
            memory_latents_flat - latent_summary[None, :], axis=1
        )
        novelty = float(np.min(distances))

        # Risk: penalise extreme amplitudes
        amplitude = float(np.max(np.abs(candidate_state)))
        risk = max(0.0, amplitude - 2.0 * mem_amplitude)

        score = alpha * uncertainty + beta * novelty - gamma * risk

        candidates.append(
            CandidateState(
                state=candidate_state,
                metadata=metadata_template,
                latent_summary=latent_summary,
                uncertainty=uncertainty,
                novelty=novelty,
                risk=risk,
                score=score,
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generative_committee_uncertainty(
    candidate_state: np.ndarray,
    metadata: dict[str, Any],
    committee: list[tuple[Any, Any, dict[str, Any]]],
    device: torch.device,
) -> float:
    """Committee disagreement on a decoded candidate state."""
    predictions: list[np.ndarray] = []
    for model, autoencoder, model_metadata in committee:
        state = torch.from_numpy(candidate_state).unsqueeze(0).to(device)
        context_features = metadata_to_context_vector(
            metadata, model_metadata["context_features"]
        )
        context = torch.from_numpy(context_features).unsqueeze(0).to(device)
        pde_id = metadata.get("pde_id", "")
        pde_index = model_metadata.get("pde_to_index", {}).get(pde_id, 0)
        pde_ids = torch.tensor([pde_index], device=device, dtype=torch.long)
        with torch.no_grad():
            z = autoencoder.encode(state)
            z_next = model(z, pde_ids, context)
            u_pred = autoencoder.decode(z_next)[0].cpu().numpy()
        predictions.append(u_pred)
    stacked = np.stack(predictions, axis=0)
    return float(np.mean(np.var(stacked, axis=0)))


def _build_minimal_metadata(model_metadata: dict[str, Any]) -> dict[str, Any]:
    """Create minimal physical metadata for scoring generative candidates."""
    pde_to_index = model_metadata.get("pde_to_index", {})
    pde_id = next(iter(pde_to_index), "burgers_1d")
    context_features = model_metadata.get("context_features", ())
    pde_params: dict[str, float] = {}
    for feat in context_features:
        if feat not in ("dt", "forcing_amplitude", "forcing_mode", "grid_size", "domain_length"):
            pde_params[feat] = 0.0
    return {
        "pde_id": pde_id,
        "dt": model_metadata.get("dt", 0.0025),
        "pde_params": pde_params,
        "forcing_descriptor": {"amplitude": 0.0, "mode": 1.0},
        "grid_descriptor": {"grid_size": 64, "domain_length": 6.283185},
        "bc_descriptor": {"type": "periodic"},
        "trajectory_id": "generative",
        "split": "train",
        "sample_origin": "generative",
        "solver_status": "pending",
        "solver_runtime_sec": 0.0,
        "seed": 0,
        "time_index": 0,
    }
