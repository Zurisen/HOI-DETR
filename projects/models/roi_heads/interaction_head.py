# projects/models/roi_heads/interaction_head.py
#interaction_head_2outs.py
import torch
import torch.nn as nn
from mmdet.models import build_loss


class InteractionHead(nn.Module):
    def __init__(
        self,
        in_dim=256,
        hidden_dim=256,
        final_hidden_dim=256,
        loss_interaction=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        ),
    ):
        super(InteractionHead, self).__init__()
        input_dim =in_dim #  2*in_dim + 11 #in_dim * 2 # , 4*in_dim + 11
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Linear(final_hidden_dim, 2)  # Binary classification: no-interact / interact
        )

        self.loss_interaction = build_loss(loss_interaction)
        self.init_weights()

    def forward(self, pair_embeddings, interaction_targets=None):
        """
        Args:
            pair_embeddings (Tensor): shape (N, 2*in_dim)
            interaction_targets (Tensor): shape (N,), optional

        Returns:
            logits (Tensor): shape (N, 2)
            loss (Tensor or None)
        """
        logits = self.mlp(pair_embeddings)
        loss = None
        if interaction_targets is not None:
            loss = self.loss_interaction(logits, interaction_targets)
        return logits, loss


    def forward_logits(self, pair_features):
        """Forward without computing loss (for inference)."""
        if hasattr(self, "mlp"):
            logits = self.mlp(pair_features)
        else:
            logits = self.fc(pair_features)

        # Convert to probabilities (class 1 = "interaction")
        probs = torch.softmax(logits, dim=-1)[:, 1]  # [N]
        return probs


    def init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
