from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

# ---------------------------------------------------------------------------
# Spatial attention
# ---------------------------------------------------------------------------

class SpatialChangeAttention(nn.Module):
    """Lightweight cross-spatial attention focused on inter-frame difference.

    Given two feature maps (start, end), this module learns a spatial weight
    map conditioned on their concatenation and signed difference, then applies
    it to both streams.  Architecture follows CBAM-style spatial attention
    (Woo et al., ECCV 2018) adapted for two-stream inversion.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        inner = max(channels // 16, 32)
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 3, inner, 1, bias=False),
            nn.GroupNorm(min(32, inner), inner),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat_start: torch.Tensor,
        feat_end: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        diff = feat_end - feat_start
        weight = self.proj(torch.cat([feat_start, feat_end, diff], dim=1))
        attended_start = feat_start * weight
        attended_end = feat_end * weight
        return attended_start, attended_end


# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------

def _build_encoder(
    backbone: str,
    pretrained: bool,
) -> tuple[nn.Module, int, int]:
    """Return ``(stem, feature_dim, spatial_size)``.

    ``stem`` is everything up to (but excluding) the final AdaptiveAvgPool2d
    and fc, so the output is a spatial feature map.
    """
    if backbone == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet18(weights=weights)
        # drop last two modules: AdaptiveAvgPool2d + Linear
        stem = nn.Sequential(*list(resnet.children())[:-2])
        return stem, 512, 7  # 224 px -> 7x7
    if backbone == "resnet34":
        weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet34(weights=weights)
        stem = nn.Sequential(*list(resnet.children())[:-2])
        return stem, 512, 7
    if backbone == "resnet50":
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet50(weights=weights)
        stem = nn.Sequential(*list(resnet.children())[:-2])
        return stem, 2048, 7
    raise ValueError(f"unknown backbone: {backbone}")


def _freeze_prefixes(backbone: str) -> list[str]:
    """Parameters whose name starts with any of these prefixes are frozen."""
    if backbone.startswith("resnet"):
        return [
            "stem.0",   # conv1
            "stem.1",   # bn1
            "stem.4.0", # layer1.0
            "stem.4.1", # layer1.1
            "stem.5.0", # layer2.0
            "stem.5.1", # layer2.1
        ]
    return []


# ---------------------------------------------------------------------------
# Rotation loss helpers
# ---------------------------------------------------------------------------

# Indices of rotation components in the canonical 14-dim action layout used
# by test.py.  The 6-dim layout used by train.py has no rotation axes.
_ROTATION_INDICES_14: list[int] = [3, 4, 5, 10, 11, 12]

_GRIPPER_INDICES_14: list[int] = [6, 13]


def _guess_rotation_indices(action_dim: int) -> list[int] | None:
    """Best-effort heuristic: if the layout matches the 14-dim format."""
    if action_dim == 14:
        return _ROTATION_INDICES_14
    return None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimpleActionModel(nn.Module):
    """Visual inverse-dynamics model for bimanual manipulation.

    A shared, pretrained ResNet encodes *start* and *end* images independently
    (Siamese stream).  Spatial feature maps are fed through a lightweight
    cross-spatial attention module that focuses on inter-frame change, then
    pooled and mapped to the action space by a deep MLP.

    Parameters
    ----------
    action_dim:
        Dimensionality of the output action vector (6 for position-only, 14
        for full bimanual pose).
    backbone:
        ResNet variant used as the visual encoder.
    pretrained:
        When True, load ImageNet-pretrained weights.
    hidden_dim:
        Width of the first MLP hidden layer.
    rotation_indices:
        If provided, a periodic cosine penalty is added to the regression loss
        for these dimensions.  When *None* a heuristic based on ``action_dim``
        is used.
    """

    def __init__(
        self,
        action_dim: int = 6,
        backbone: str = "resnet18",
        pretrained: bool = True,
        hidden_dim: int = 768,
        rotation_indices: list[int] | None = None,
        mag_weight: bool = True,
        mag_weight_median: float = 0.08,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.backbone = backbone

        # -- vision stem (outputs spatial feature map) --
        self.stem, feat_dim, self._spatial_size = _build_encoder(backbone, pretrained)

        # -- spatial attention --
        self.spatial_attn = SpatialChangeAttention(feat_dim)

        # -- pooling --
        self.pool = nn.AdaptiveAvgPool2d(1)

        # -- action head --
        # The head sees: [pooled(start), pooled(end), pooled(attended_start),
        #                 pooled(attended_end), pooled(attended_diff)]
        fused_dim = feat_dim * 5
        self.action_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, action_dim),
        )

        # -- rotation-aware loss config --
        if rotation_indices is not None:
            self.rotation_indices: list[int] | None = rotation_indices
        else:
            self.rotation_indices = _guess_rotation_indices(action_dim)

        # -- magnitude-aware loss config --
        # Register a buffer so the median is tracked in state_dict but not
        # updated by the optimizer.  The default (0.08) is the median action
        # norm measured on the RoboTwin training set (position deltas, meters).
        self.mag_weight = mag_weight
        self.register_buffer("action_norm_median", torch.tensor(mag_weight_median))

        # -- apply freezing policy --
        self._apply_freeze_policy(backbone)

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def _apply_freeze_policy(self, backbone: str) -> None:
        """Freeze early layers and all BatchNorm-like modules.

        Early conv layers learn generic texture/edge detectors that transfer
        well; later layers adapt to domain-specific semantics.  Frozen BN
        prevents unstable running-statistic updates under small batches and
        mixed-precision training.
        """
        freeze = _freeze_prefixes(backbone)
        if not freeze:
            return

        frozen_params = 0
        trainable_params = 0
        for name, param in self.named_parameters():
            if any(name.startswith(p) for p in freeze):
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                trainable_params += param.numel()

        # Put all BN / GroupNorm in eval mode so running stats are not updated.
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                m.requires_grad_(False)

        if frozen_params > 0:
            total = frozen_params + trainable_params
            print(
                f"[SimpleActionModel] frozen {frozen_params / 1e6:.1f}M / "
                f"{total / 1e6:.1f}M params "
                f"({100 * frozen_params / total:.0f}%)"
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, start_image: torch.Tensor, end_image: torch.Tensor) -> torch.Tensor:
        # Siamese encoding -> spatial feature maps
        f_start = self.stem(start_image)  # [B, C, H, W]
        f_end = self.stem(end_image)

        # Cross-spatial attention
        a_start, a_end = self.spatial_attn(f_start, f_end)
        a_diff = a_end - a_start

        # Pool each stream independently
        v_start = self.pool(f_start)
        v_end = self.pool(f_end)
        v_astart = self.pool(a_start)
        v_aend = self.pool(a_end)
        v_adiff = self.pool(a_diff)

        fused = torch.cat([v_start, v_end, v_astart, v_aend, v_adiff], dim=1)
        return self.action_head(fused)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def regression_loss(
        self,
        start_image: torch.Tensor,
        end_image: torch.Tensor,
        target_action: torch.Tensor,
    ) -> torch.Tensor:
        pred = self(start_image, end_image)

        # Per-dimension SmoothL1 loss (kept per-sample so we can reweight).
        per_dim = F.smooth_l1_loss(pred, target_action, reduction="none")  # [B, D]
        per_sample = per_dim.mean(dim=1)  # [B]

        # --- magnitude-aware sample reweighting ---
        if self.mag_weight:
            norm = target_action.norm(dim=1)  # [B], L2 norm of the action vector
            # Scale each sample by how far its motion magnitude is from the
            # median.  Near-zero frames are down-weighted; large motions are
            # up-weighted.  Clamped to [0.1, 3.0] to keep training stable.
            weight = torch.clamp(norm / self.action_norm_median, 0.1, 3.0)
            loss = (per_sample * weight).mean()
        else:
            loss = per_sample.mean()

        # Periodic cosine penalty on rotation axes (14-dim layout only).
        if self.rotation_indices is not None:
            rot_pred = pred[:, self.rotation_indices]
            rot_targ = target_action[:, self.rotation_indices]
            loss = loss + 0.1 * (1.0 - torch.cos(rot_pred - rot_targ)).mean()

        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(
        self,
        start_image: torch.Tensor,
        end_image: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        return self(start_image, end_image)
