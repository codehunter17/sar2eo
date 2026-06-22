"""
models/losses.py — Loss Functions
===================================
Two loss components for Pix2Pix training.

1. Adversarial Loss (GAN loss)
   ---------------------------
   The generator and discriminator play a minimax game:
   - Discriminator tries to tell REAL EO from FAKE (generated) EO.
   - Generator tries to fool the discriminator into calling FAKE = REAL.

   We use least-squares GAN (LSGAN) instead of vanilla binary cross-entropy:
     - LSGAN minimises the Pearson χ² divergence
     - More stable training (no vanishing gradients for confident predictions)
     - Originally from Mao et al. (2017) "Least Squares Generative Adversarial Networks"

   LSGAN discriminator loss:
     L_D = 0.5 * E[(D(real) - 1)²] + 0.5 * E[(D(fake) - 0)²]
   LSGAN generator loss:
     L_G_adv = 0.5 * E[(D(fake) - 1)²]  (generator wants D to output 1 for fake)

2. L1 Reconstruction Loss
   ------------------------
   L_L1 = E[|EO_real - EO_fake|]

   Why L1 and not L2?
   - L2 minimises mean squared error → the model learns to predict the
     mean over all plausible outputs, producing blurry images.
   - L1 minimises mean absolute error → slightly less blurry, but still
     suffers from the same mode-averaging issue.
   - Neither L1 nor L2 alone produces sharp images; the adversarial loss
     is what adds perceptual sharpness.  L1 anchors the overall structure.

Total generator loss:
   L_G = L_G_adv + λ_L1 * L_L1    (λ_L1 = 100 per Isola et al.)
"""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    LSGAN loss (least-squares).

    Automatically creates real/fake label tensors that match the
    shape of the discriminator output (patch grid, not scalar).
    """

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        self.loss = nn.MSELoss()   # MSE → LSGAN

    def _make_target(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        """Create a label tensor of the same shape as prediction."""
        label = self.real_label if is_real else self.fake_label
        return label.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = self._make_target(prediction, is_real)
        return self.loss(prediction, target)


class Pix2PixLoss(nn.Module):
    """
    Combined Pix2Pix loss for the generator:
        L_G = λ_adv * L_GAN + λ_l1 * L_L1

    For the l1_only ablation mode, the adversarial component is
    simply set to zero and no discriminator is needed.

    Args:
        lambda_l1:  Weight on the L1 reconstruction term (default 100).
        mode:       'full_gan' | 'l1_only'
    """

    def __init__(self, lambda_l1: float = 100.0, mode: str = "full_gan"):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.mode = mode
        self.gan_loss = GANLoss()
        self.l1_loss  = nn.L1Loss()

    def generator_loss(
        self,
        fake_pred:   torch.Tensor,    # discriminator output on generated EO
        fake_eo:     torch.Tensor,    # generated EO image
        real_eo:     torch.Tensor,    # ground truth EO image
    ) -> dict:
        """
        Compute generator loss.
        Returns a dict with individual terms for logging.
        """
        l1 = self.l1_loss(fake_eo, real_eo)

        if self.mode == "l1_only":
            return {"g_total": l1, "g_l1": l1, "g_adv": torch.tensor(0.0)}

        # Adversarial: generator wants discriminator to output 1 (real)
        adv = self.gan_loss(fake_pred, is_real=True)
        total = adv + self.lambda_l1 * l1

        return {
            "g_total": total,
            "g_adv":   adv,
            "g_l1":    l1,
        }

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,   # discriminator output on real (SAR, real_EO)
        fake_pred: torch.Tensor,   # discriminator output on fake (SAR, generated_EO)
    ) -> dict:
        """
        Compute discriminator loss.
        D tries to output 1 for real pairs and 0 for fake pairs.
        """
        if self.mode == "l1_only":
            # No discriminator in l1_only mode
            return {"d_total": torch.tensor(0.0), "d_real": torch.tensor(0.0), "d_fake": torch.tensor(0.0)}

        d_real = self.gan_loss(real_pred, is_real=True)
        d_fake = self.gan_loss(fake_pred, is_real=False)
        total  = 0.5 * (d_real + d_fake)   # average as in original Pix2Pix

        return {
            "d_total": total,
            "d_real":  d_real,
            "d_fake":  d_fake,
        }
