"""
Tests for NICTO-GAN
Verifies all components work correctly.
"""

import sys
import torch
import pytest
sys.path.insert(0, r"C:\Users\BYU\Desktop\NICTO")

from nicto_ai.gan.generator import NICTOGenerator
from nicto_ai.gan.discriminator import NICTODiscriminator
from nicto_ai.gan.loss import R3GANLoss
from nicto_ai.gan.config import GANConfig, get_colab_t4_config


class TestGenerator:
    def test_forward(self):
        config = GANConfig(target_size=64, gen_channels=64, z_dim=128, style_dim=128, moe_experts=2)
        gen = NICTOGenerator(
            z_dim=config.z_dim,
            style_dim=config.style_dim,
            channels=config.gen_channels,
            target_size=config.target_size,
            moe_experts=config.moe_experts,
        )
        z = torch.randn(2, config.z_dim)
        out = gen(z)
        assert out.shape == (2, 3, 64, 64)
        assert out.min() >= -1 and out.max() <= 1

    def test_return_style(self):
        config = GANConfig(target_size=64, gen_channels=64, z_dim=128, style_dim=128, moe_experts=2)
        gen = NICTOGenerator(
            z_dim=config.z_dim,
            style_dim=config.style_dim,
            channels=config.gen_channels,
            target_size=config.target_size,
            moe_experts=config.moe_experts,
        )
        z = torch.randn(2, config.z_dim)
        out, w = gen(z, return_style=True)
        assert out.shape == (2, 3, 64, 64)
        assert w.shape == (2, config.style_dim)

    def test_gradient_flow(self):
        config = GANConfig(target_size=64, gen_channels=64, z_dim=128, style_dim=128, moe_experts=2)
        gen = NICTOGenerator(
            z_dim=config.z_dim,
            style_dim=config.style_dim,
            channels=config.gen_channels,
            target_size=config.target_size,
            moe_experts=config.moe_experts,
        )
        z = torch.randn(2, config.z_dim)
        out = gen(z)
        loss = out.mean()
        loss.backward()
        assert all(p.grad is not None for p in gen.parameters() if p.requires_grad)


class TestDiscriminator:
    def test_forward(self):
        config = GANConfig(target_size=64, disc_channels=64)
        disc = NICTODiscriminator(
            channels=config.disc_channels,
            input_size=config.target_size,
        )
        x = torch.randn(2, 3, 64, 64)
        out = disc(x)
        assert "score" in out
        assert out["score"].shape == (2, 1)

    def test_consciousness(self):
        config = GANConfig(target_size=64, disc_channels=64, consciousness=True)
        disc = NICTODiscriminator(
            channels=config.disc_channels,
            input_size=config.target_size,
            consciousness=True,
        )
        x = torch.randn(2, 3, 64, 64)
        out = disc(x)
        assert "uncertainty" in out
        assert out["uncertainty"].shape == (2, 1)
        assert out["uncertainty"].min() >= 0 and out["uncertainty"].max() <= 1


class TestLoss:
    def test_r3gan_loss(self):
        criterion = R3GANLoss(r1_gamma=10.0, r2_gamma=10.0)
        real_scores = torch.randn(4, 1, requires_grad=True)
        fake_scores = torch.randn(4, 1, requires_grad=True)
        result = criterion(real_scores, fake_scores)
        assert "g_loss" in result
        assert "d_loss" in result
        assert result["g_loss"].requires_grad
        assert result["d_loss"].requires_grad


class TestEndToEnd:
    def test_train_step(self):
        config = GANConfig(
            target_size=64,
            gen_channels=64,
            disc_channels=64,
            z_dim=128,
            style_dim=128,
            moe_experts=2,
            r1_gamma=10.0,
            r2_gamma=10.0,
        )

        gen = NICTOGenerator(
            z_dim=config.z_dim,
            style_dim=config.style_dim,
            channels=config.gen_channels,
            target_size=config.target_size,
            moe_experts=config.moe_experts,
        )
        disc = NICTODiscriminator(
            channels=config.disc_channels,
            input_size=config.target_size,
        )
        criterion = R3GANLoss(r1_gamma=config.r1_gamma, r2_gamma=config.r2_gamma)

        g_opt = torch.optim.Adam(gen.parameters(), lr=config.g_lr)
        d_opt = torch.optim.Adam(disc.parameters(), lr=config.d_lr)

        # One training step
        batch_size = 4
        real = torch.randn(batch_size, 3, 64, 64)

        # D step
        d_opt.zero_grad()
        z = torch.randn(batch_size, config.z_dim)
        fake = gen(z).detach()
        real_out = disc(real)
        fake_out = disc(fake)
        losses = criterion(real_out["score"], fake_out["score"], real, fake, gen, disc)
        losses["d_loss"].backward()
        d_opt.step()

        # G step
        g_opt.zero_grad()
        z = torch.randn(batch_size, config.z_dim)
        fake = gen(z)
        fake_out = disc(fake)
        g_loss = torch.nn.functional.softplus(-fake_out["score"]).mean()
        g_loss.backward()
        g_opt.step()

        assert True  # If we got here, training works


if __name__ == "__main__":
    test = TestGenerator()
    test.test_forward()
    print("Generator forward: OK")

    test.test_return_style()
    print("Generator return_style: OK")

    test.test_gradient_flow()
    print("Generator gradient: OK")

    test_d = TestDiscriminator()
    test_d.test_forward()
    print("Discriminator forward: OK")

    test_d.test_consciousness()
    print("Discriminator consciousness: OK")

    test_loss = TestLoss()
    test_loss.test_r3gan_loss()
    print("R3GAN loss: OK")

    test_e2e = TestEndToEnd()
    test_e2e.test_train_step()
    print("End-to-end train step: OK")

    print("\nAll tests passed!")
