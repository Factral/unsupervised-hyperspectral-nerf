"""
Semantic field using cross-attention between learnable endmember embeddings and position-appearance features.
"""

from typing import Dict, Optional, Literal, Type, Sequence
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.field_components.mlp import MLP

import torch
import torch.nn as nn
from typing import Optional, Type

        

class SemanticField(FieldComponent):
    """Semantic field using attention mechanisms and learnable endmember embeddings."""
    
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
        
        # Feature extraction MLP
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

        #endmembers projection to feature_dim
        self.proj_mlp_endmembers = MLP(
            in_dim=wavelengths,
            num_layers=2,
            layer_width=hidden_dim,
            out_dim=feature_dim,
            activation=nn.ReLU(),
            out_activation=nn.Tanh(),
            implementation=implementation,
        )
        
        # Learnable endmember embeddings (queries)
        self.register_parameter(
            'endmembers', 
            nn.Parameter(torch.rand(num_classes, 1, wavelengths))
        )
        
        # Cross-attention between endmembers and features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Final self-attention on class predictions
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output layer for class logits
        self.output_layer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
    def forward(
        self, 
        positions,
        density_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for semantic prediction."""

        positions_flat = self.position_encoding(positions.view(-1, 3))

        if density_embedding is None:
            raise ValueError("density embedding is required")

        features_input = torch.cat([
            positions_flat,
            density_embedding.view(-1, self.dir_embedding_dim)
        ], dim=-1)


        # Extract features
        features = self.feature_mlp(features_input)  # (N, feature_dim)

        features = features.unsqueeze(0)  # (1, N, feature_dim)
    
        # Cross-attention between endmember queries and features
        endmembers = self.endmembers.permute(1, 0, 2)  # (1, num_classes, feature_dim)
        endmembers = self.proj_mlp_endmembers(endmembers)
        
        attn_output, _ = self.cross_attention(
            query=endmembers,
            key=features,
            value=features
        )  # (1, num_classes, feature_dim)
        
        # Self-attention for final refinement
        attn_output = self.norm2(attn_output)
        refined_output, _ = self.self_attention(
            query=attn_output,
            key=attn_output,
            value=attn_output
        )  # (1, num_classes, feature_dim)
        
        # Generate class logits
        logits = self.output_layer(refined_output).squeeze(-1)  # (1, num_classes)
        logits = logits.expand(features.size(1), -1)  # (N, num_classes)
        
        probabilities = F.softmax(logits, dim=-1)
        
        return probabilities