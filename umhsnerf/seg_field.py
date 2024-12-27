from typing import Dict, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.field_components.mlp import MLP


class SemanticField(FieldComponent):
    """Semantic field using attention mechanisms and learnable endmember embeddings, with pre-norm attention."""

    def __init__(
        self,
        position_encoding: nn.Module,
        num_classes: int,
        feature_dim: int = 256,
        num_heads: int = 4,
        dir_embedding_dim: int = 32,
        wavelengths: int = 21,
        hidden_dim: int = 64,
        implementation: Literal["tcnn", "torch"] = "torch",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.position_encoding = position_encoding
        self.dir_embedding_dim = dir_embedding_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # ----------------------------------------------------------------------
        # 1) MLP to get "features" for each sample
        # ----------------------------------------------------------------------
        input_dim = self.position_encoding.get_out_dim() + self.dir_embedding_dim
        self.feature_mlp = MLP(
            in_dim=input_dim,
            num_layers=2,
            layer_width=hidden_dim,
            out_dim=feature_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        # ----------------------------------------------------------------------
        # 2) Projection MLP for endmembers
        # ----------------------------------------------------------------------
        self.proj_mlp_endmembers = MLP(
            in_dim=wavelengths,
            num_layers=2,
            layer_width=hidden_dim,
            out_dim=feature_dim,
            activation=nn.ReLU(),
            out_activation=nn.Tanh(),
            implementation=implementation,
        )

        # ----------------------------------------------------------------------
        # 3) Learnable endmember embeddings
        #    shape: (num_classes, 1, wavelengths)
        # ----------------------------------------------------------------------
        self.register_parameter(
            "endmembers",
            nn.Parameter(torch.rand(num_classes, 1, wavelengths))
        )

        # ----------------------------------------------------------------------
        # 4) Cross-attention: endmembers <--> features
        # ----------------------------------------------------------------------
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ----------------------------------------------------------------------
        # 5) Self-attention (for the aggregated endmember latents)
        # ----------------------------------------------------------------------
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ----------------------------------------------------------------------
        # 6) Output: small feedforward for class logits
        # ----------------------------------------------------------------------
        self.output_layer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout)

        # ----------------------------------------------------------------------
        # 7) LayerNorm modules for pre-norm:
        #    - We'll apply them before cross-attn / self-attn
        # ----------------------------------------------------------------------
        self.norm_keyval = nn.LayerNorm(feature_dim)  # for the features
        self.norm_query = nn.LayerNorm(feature_dim)   # for the endmembers
        self.norm_self = nn.LayerNorm(feature_dim)    # for the self-attention input
        self.norm_out = nn.LayerNorm(feature_dim)     # optional final norm

    def forward(
        self,
        positions: Tensor,             # (N, 3)
        density_embedding: Optional[Tensor] = None,  # (N, dir_embed_dim)
    ) -> Tensor:
        """Forward pass for semantic prediction.

        Args:
            positions: shape (N, 3)
            density_embedding: shape (N, dir_embedding_dim), from the main field's 'geo_feat_dim'
        Returns:
            probabilities: shape (N, num_classes)
        """

        if density_embedding is None:
            raise ValueError("density_embedding is required")

        # (N, pos_enc_dim)
        positions_flat = self.position_encoding(positions.view(-1, 3))

        # Concatenate position + direction embeddings
        features_input = torch.cat([positions_flat, density_embedding], dim=-1)
        # => (N, feature_dim)
        features = self.feature_mlp(features_input)

        # 1) Turn features into shape (1, N, feature_dim) for cross-attn
        features = features.unsqueeze(0)  # => (1, N, feature_dim)

        # 2) Project endmembers -> shape (num_classes, wavelengths) -> (num_classes, feature_dim)
        endmembers = self.endmembers.squeeze(1)            # => (num_classes, wavelengths)
        endmembers_proj = self.proj_mlp_endmembers(endmembers)
        endmembers_proj = endmembers_proj.unsqueeze(0)     # => (1, num_classes, feature_dim)

        # ----------------------
        # Cross-Attention (Pre-norm)
        # ----------------------
        # Norm queries
        x_query = self.norm_query(endmembers_proj)  # shape (1, num_classes, feature_dim)
        # Norm keys/values
        x_keyval = self.norm_keyval(features)        # shape (1, N, feature_dim)

        # cross-attention
        attn_output, _ = self.cross_attention(
            query=x_query,
            key=x_keyval,
            value=x_keyval
        )
        # Residual
        cross_out = endmembers_proj + self.dropout(attn_output)

        # ----------------------
        # Self-Attention (Pre-norm)
        # ----------------------
        x_self = self.norm_self(cross_out)
        refined_output, _ = self.self_attention(
            query=x_self,
            key=x_self,
            value=x_self,
        )
        final = cross_out + self.dropout(refined_output)

        # ----------------------
        # Output
        # ----------------------
        # optional final norm
        final_normed = self.norm_out(final)  # shape (1, num_classes, feature_dim)

        # linear => shape (1, num_classes, 1)
        logits = self.output_layer(final_normed).squeeze(-1)  # => (1, num_classes)
        # Expand to match (N, num_classes)
        # we want shape (N, num_classes) => replicate across the N samples
        N = features.size(1)
        logits = logits.expand(N, -1)

        # convert to probabilities
        probabilities = F.softmax(logits, dim=-1)
        return probabilities
