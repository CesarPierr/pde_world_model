"""Tests for the flow matching generative sampler."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pdewm.acquisition.generative import (
    COMPRESSED_DIM,
    ConvAttentionVelocityNet,
    FlowMatchingSamplerConfig,
    LatentCompressor,
    LatentDecompressor,
    LatentFlowMatchingSampler,
    VelocityNetConfig,
)


class TestLatentCompressor:
    def test_output_shape(self) -> None:
        comp = LatentCompressor(32, 16)
        z = torch.randn(4, 32, 16)
        out = comp(z)
        assert out.shape == (4, COMPRESSED_DIM)

    def test_handles_larger_latent(self) -> None:
        comp = LatentCompressor(64, 32)
        z = torch.randn(2, 64, 32)
        out = comp(z)
        assert out.shape == (2, COMPRESSED_DIM)


class TestLatentDecompressor:
    def test_output_shape(self) -> None:
        decomp = LatentDecompressor(32, 16)
        x = torch.randn(4, COMPRESSED_DIM)
        out = decomp(x)
        assert out.shape == (4, 32, 16)

    def test_restores_larger_spatial(self) -> None:
        decomp = LatentDecompressor(64, 32)
        x = torch.randn(2, COMPRESSED_DIM)
        out = decomp(x)
        assert out.shape == (2, 64, 32)


class TestConvAttentionVelocityNet:
    def test_forward_shape(self) -> None:
        net = ConvAttentionVelocityNet(VelocityNetConfig())
        x = torch.randn(4, COMPRESSED_DIM)
        t = torch.rand(4)
        v = net(x, t)
        assert v.shape == (4, COMPRESSED_DIM)

    def test_no_attention(self) -> None:
        net = ConvAttentionVelocityNet(VelocityNetConfig(use_attention=False))
        x = torch.randn(4, COMPRESSED_DIM)
        t = torch.rand(4)
        v = net(x, t)
        assert v.shape == (4, COMPRESSED_DIM)


class TestLatentFlowMatchingSampler:
    @pytest.fixture()
    def sampler(self) -> LatentFlowMatchingSampler:
        config = FlowMatchingSamplerConfig(training_epochs=5, ode_steps=5)
        return LatentFlowMatchingSampler(config, 32, 16, torch.device("cpu"))

    def test_fit_returns_stats(self, sampler: LatentFlowMatchingSampler) -> None:
        latents = np.random.randn(12, 32, 16).astype(np.float32)
        losses = np.random.rand(12).astype(np.float32)
        stats = sampler.fit(latents, losses)
        assert "mean_loss" in stats
        assert "final_loss" in stats
        assert np.isfinite(stats["mean_loss"])

    def test_sample_shape(self, sampler: LatentFlowMatchingSampler) -> None:
        latents = np.random.randn(12, 32, 16).astype(np.float32)
        losses = np.random.rand(12).astype(np.float32)
        sampler.fit(latents, losses)
        samples = sampler.sample(8)
        assert samples.shape == (8, 32, 16)
        assert np.all(np.isfinite(samples))

    def test_uniform_temperature(self) -> None:
        config = FlowMatchingSamplerConfig(
            training_epochs=5, ode_steps=5, temperature=0.0,
        )
        sampler = LatentFlowMatchingSampler(config, 32, 16, torch.device("cpu"))
        latents = np.random.randn(12, 32, 16).astype(np.float32)
        losses = np.random.rand(12).astype(np.float32)
        sampler.fit(latents, losses)
        samples = sampler.sample(4)
        assert samples.shape == (4, 32, 16)
