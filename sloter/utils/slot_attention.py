import torch
from torch import nn
import torch.nn.functional as F
import math
from PIL import Image
import numpy as np


class SlotAttention(nn.Module):
    def __init__(
        self,
        num_classes,
        slots_per_class,
        dim,
        iters=3,
        eps=1e-8,
        vis=False,
        vis_id=0,
        loss_status=1,
        power=1,
        to_k_layer=1,
        num_heads=8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.slots_per_class = slots_per_class
        self.num_slots = num_classes * slots_per_class
        self.iters = iters
        self.eps = eps
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert (
            self.head_dim * num_heads == dim
        ), "Hidden dimension must be divisible by num_heads"

        self.scale = self.head_dim**-0.5
        self.loss_status = loss_status

        # Initialize slots
        slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))
        mu = slots_mu.expand(1, self.num_slots, -1)
        sigma = slots_sigma.expand(1, self.num_slots, -1)
        self.initial_slots = nn.Parameter(torch.normal(mu, sigma))

        # Linear projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(
            dim, dim, bias=False
        )  # Optional: Use if you want a separate projection for V

        # GRU for slot update
        self.gru = nn.GRU(dim, dim)

        self.vis = vis
        self.vis_id = vis_id
        self.power = power

    def forward(self, inputs, inputs_x):
        """
        inputs: positional encoded inputs [batch_size, num_patches, dim]
        inputs_x: original inputs without positional encoding [batch_size, num_patches, dim]
        """
        b, n, d = inputs.shape
        slots = self.initial_slots.expand(b, -1, -1)  # [batch_size, num_slots, dim]
        k = self.to_k(inputs)  # [batch_size, num_patches, dim]
        v = (
            self.to_v(inputs_x) if hasattr(self, "to_v") else inputs_x
        )  # [batch_size, num_patches, dim]

        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(slots)  # [batch_size, num_slots, dim]

            # Reshape for multi-head attention using reshape()
            q = q.reshape(b, self.num_slots, self.num_heads, self.head_dim).transpose(
                1, 2
            )  # [b, num_heads, num_slots, head_dim]
            k = k.reshape(b, n, self.num_heads, self.head_dim).transpose(
                1, 2
            )  # [b, num_heads, n, head_dim]
            v = v.reshape(b, n, self.num_heads, self.head_dim).transpose(
                1, 2
            )  # [b, num_heads, n, head_dim]

            # Scaled dot-product attention
            dots = (
                torch.matmul(q, k.transpose(-2, -1)) * self.scale
            )  # [b, num_heads, num_slots, n]
            attn = torch.softmax(dots, dim=-1)  # [b, num_heads, num_slots, n]

            # Compute attention updates
            updates = torch.matmul(attn, v)  # [b, num_heads, num_slots, head_dim]
            updates = (
                updates.transpose(1, 2).contiguous().reshape(b, self.num_slots, -1)
            )  # [b, num_slots, dim]

            # Slot update with GRU
            self.gru.flatten_parameters()
            slots, _ = self.gru(
                updates.reshape(1, -1, self.dim), slots_prev.reshape(1, -1, self.dim)
            )
            slots = slots.reshape(b, -1, self.dim)

            if self.vis:
                # Visualization code can be added here if needed
                pass

        # Continue with the rest of your forward method...
        # Combine slots if slots_per_class > 1
        if self.slots_per_class > 1:
            new_updates = torch.zeros(
                (updates.size(0), self.num_classes, updates.size(-1)),
                device=updates.device,
            )
            for slot_class in range(self.num_classes):
                start_idx = self.slots_per_class * slot_class
                end_idx = self.slots_per_class * (slot_class + 1)
                new_updates[:, slot_class] = torch.sum(
                    updates[:, start_idx:end_idx], dim=1
                )
            updates = new_updates

        # Compute slot loss
        attn_relu = F.relu(attn)
        slot_loss = torch.sum(attn_relu) / (
            attn.size(0) * attn.size(1) * attn.size(2) * attn.size(3)
        )
        slot_loss = torch.pow(slot_loss, self.power)

        # Return the final outputs and attention loss
        return self.loss_status * torch.sum(updates, dim=1), slot_loss
