"""FreeInpaint: learnable mask attention and confidence-weighted support tokens."""

from typing import Optional
import logging
import torch
from torch import nn
import torch.nn.functional as F
from depth_anything_3.api import DepthAnything3


from depth_anything_3.model.reference_view_selector import (
    reorder_by_reference,
    restore_original_order,
)


def view_order_aware_provider(base):
    """Keep source masks aligned if DA3 chooses a different internal anchor view."""
    if base is None:
        return None
    selected = None

    def set_view_order(indices):
        nonlocal selected
        selected = indices

    def provider(x, layer):
        original = restore_original_order(x, selected) if selected is not None else x
        bias = base(original, layer)
        return reorder_by_reference(bias, selected) if selected is not None else bias

    provider.set_view_order = set_view_order
    return provider


class SoftMaskRefiner(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """g_l(F): an unscaled, signed convolutional residual."""
        return self.net(feat)


class _ResidualMaskBias:
    """Binary-initialized recurrent mask logits."""

    def _configure_mask_bias(self, alpha, beta, epsilon):
        if alpha <= 0 or beta <= 0 or (not 0 < epsilon < 0.5):
            raise ValueError("LMA requires alpha,beta > 0 and 0 < epsilon < 0.5")
        self.mask_alpha = float(alpha)
        self.mask_beta = float(beta)
        self.mask_epsilon = float(epsilon)

    def _initial_mask_logits(self, mask):
        binary_mask = (mask > 0.5).float()
        return torch.logit(
            self.mask_epsilon + (1 - 2 * self.mask_epsilon) * binary_mask
        )


class FreeInpaint(_ResidualMaskBias, nn.Module):
    """Checkpoint-compatible inference wrapper; cameras are always predicted.

    P0 = logit(epsilon + (1 - 2 epsilon) M)
    Pl = P(l-1) + alpha * g_l(F_l)
    Bl = -beta * sigmoid(Pl)

    The bias acts on keys of cross-view attention. Reference and special tokens
    have zero bias; support keys use their opacity-derived confidence penalties.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        mask_hidden_dim: int = 8,
        mask_alpha: float = 1.5,
        mask_beta: float = 10.0,
        mask_epsilon: float = 0.02,
        patch_size: int = 14,
    ):
        super().__init__()
        self.device = device
        self.patch_size = patch_size
        self._configure_mask_bias(mask_alpha, mask_beta, mask_epsilon)
        print(f"[FreeInpaint] Building {model_name} architecture...")
        # freeinpaint.pth contains the complete model, including the backbone.
        # Build from the bundled architecture config; do not download DA3 again.
        self.da3_api = DepthAnything3(model_name=model_name)
        self.model = self.da3_api.model
        self.global_block_indices = self._get_global_block_indices()
        self.embed_dim = self._get_embed_dim()
        self.mask_refiners = nn.ModuleList(
            [
                SoftMaskRefiner(feat_dim=self.embed_dim, hidden_dim=mask_hidden_dim)
                for _ in self.global_block_indices
            ]
        )
        logging.getLogger("depth_anything_3").setLevel(logging.WARNING)
        logging.getLogger("depth_anything_3.model.dinov2.vision_transformer").setLevel(
            logging.WARNING
        )

    def _get_global_block_indices(self) -> list[int]:
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            return []
        pretrained = getattr(backbone, "pretrained", backbone)
        alt_start = getattr(pretrained, "alt_start", -1)
        blocks = getattr(pretrained, "blocks", [])
        if alt_start == -1:
            return []
        return [idx for idx in range(len(blocks)) if idx >= alt_start and idx % 2 == 1]

    def _get_embed_dim(self) -> int:
        backbone = getattr(self.model, "backbone", None)
        pretrained = getattr(backbone, "pretrained", backbone)
        blocks = getattr(pretrained, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise RuntimeError(
                "Unable to infer backbone embed dim for Learnable Mask Attention."
            )
        return blocks[0].norm1.normalized_shape[0]

    def _resize_token_maps(
        self, maps: torch.Tensor, patch_h: int, patch_w: int, mode: str
    ):
        if maps is None:
            return None
        (batch_size, num_views) = maps.shape[:2]
        interpolate_kwargs = {"size": (patch_h, patch_w), "mode": mode}
        if mode in {"bilinear", "bicubic"}:
            interpolate_kwargs["align_corners"] = False
        resized = F.interpolate(maps.flatten(0, 1), **interpolate_kwargs)
        return resized.view(batch_size, num_views, 1, patch_h, patch_w)

    def _prepare_support_penalty(
        self, support_masks, support_confidences, patch_h, patch_w
    ):
        if support_masks is None or support_confidences is None:
            return None
        support_masks_small = self._resize_token_maps(
            support_masks, patch_h, patch_w, mode="nearest"
        )
        support_conf_small = self._resize_token_maps(
            support_confidences, patch_h, patch_w, mode="bilinear"
        )
        support_visibility = (support_masks_small * support_conf_small).clamp(0, 1)
        support_penalty = 1.0 - support_visibility
        return support_penalty

    def _extract_source_features(
        self, x: torch.Tensor, num_src: int, patch_h: int, patch_w: int
    ) -> torch.Tensor:
        total_tokens = x.shape[2]
        patch_tokens = patch_h * patch_w
        patch_start = max(total_tokens - patch_tokens, 0)
        src_patch_tokens = x[:, 1 : 1 + num_src, patch_start:, :]
        return src_patch_tokens.transpose(-1, -2).reshape(
            x.shape[0] * num_src, self.embed_dim, patch_h, patch_w
        )

    def _build_dynamic_attn_provider(
        self,
        src_masks: torch.Tensor,
        height: int,
        width: int,
        support_masks: Optional[torch.Tensor] = None,
        support_confidences: Optional[torch.Tensor] = None,
    ):
        if len(self.mask_refiners) == 0:
            return None
        (batch_size, num_src, _, _, _) = src_masks.shape
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        src_masks_small = self._resize_token_maps(
            src_masks, patch_h, patch_w, mode="nearest"
        )
        src_logits = self._initial_mask_logits(src_masks_small)
        support_penalty = self._prepare_support_penalty(
            support_masks, support_confidences, patch_h, patch_w
        )
        num_support = 0 if support_penalty is None else support_penalty.shape[1]
        ref_tokens = torch.zeros(
            batch_size,
            1,
            1 + patch_h * patch_w,
            device=src_masks.device,
            dtype=src_masks.dtype,
        )

        def provider(x: torch.Tensor, layer_idx: int):
            nonlocal src_logits
            src_feat_maps = self._extract_source_features(
                x, num_src, patch_h, patch_w
            ).to(src_logits.dtype)
            residual = self.mask_refiners[layer_idx](src_feat_maps).view_as(src_logits)
            src_logits = src_logits + self.mask_alpha * residual.float()
            src_soft_masks = src_logits.sigmoid()
            src_tokens = src_soft_masks.flatten(3).squeeze(2)
            src_tokens = torch.cat(
                [
                    torch.zeros(
                        batch_size,
                        num_src,
                        1,
                        device=src_masks.device,
                        dtype=src_masks.dtype,
                    ),
                    src_tokens,
                ],
                dim=2,
            )
            token_groups = [ref_tokens, src_tokens]
            if num_support > 0:
                support_tokens = support_penalty.flatten(3).squeeze(2)
                support_tokens = torch.cat(
                    [
                        torch.zeros(
                            batch_size,
                            num_support,
                            1,
                            device=src_masks.device,
                            dtype=src_masks.dtype,
                        ),
                        support_tokens,
                    ],
                    dim=2,
                )
                token_groups.append(support_tokens)
            full_tokens = torch.cat(token_groups, dim=1)
            return -self.mask_beta * full_tokens

        return provider

    def forward(
        self,
        ref_img,
        masked_src_imgs,
        src_masks,
        support_imgs=None,
        support_masks=None,
        support_confidences=None,
        ref_view_strategy=None,
        **kwargs,
    ):
        (height, width) = masked_src_imgs.shape[-2:]
        groups = [ref_img.unsqueeze(1), masked_src_imgs]
        if support_imgs is not None:
            groups.append(support_imgs)
        provider = view_order_aware_provider(
            self._build_dynamic_attn_provider(
                src_masks, height, width, support_masks, support_confidences
            )
        )
        strategy = (
            {}
            if ref_view_strategy is None
            else {"ref_view_strategy": ref_view_strategy}
        )
        return self.model(
            torch.cat(groups, dim=1),
            extrinsics=None,
            intrinsics=None,
            infer_gs=True,
            attn_mask=None,
            global_attn_masks=None,
            dynamic_global_attn_provider=provider,
            **strategy,
        )
