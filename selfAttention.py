import torch
import torch.nn as nn
import torch.nn.functional as F


# Arquitetura avançada TCC


class Rebuilt_SAIDNN(nn.Module):
    def __init__(
        self,
        n_features: int,
        sequence_length: int = 10,
        embed_dim: int = 160,
        num_heads: int = 3,
        num_blocks: int = 2,
        dropout: float = 0.2,
        use_attention_pooling: bool = True,
    ):
        super(Rebuilt_SAIDNN, self).__init__()
        # CNNs com conexões residuais: três blocos convolucionais com skip connections.
        self.input_projection = nn.Linear(n_features, embed_dim)

        NUMBER_CNN_BLOCKS = 3
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    kernel_size=3,
                    padding=1,
                )
                for _ in range(NUMBER_CNN_BLOCKS)
            ]
        )

        # Positional embeddings aprendíveis: substituição do encoding sinusoidal fixo por parâmetros treináveis
        self.positional_embedding = nn.Parameter(
            torch.randn(1, sequence_length, embed_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)

        # Pre-Layer Normalization: normalização aplicada antes das subcamadas, em vez de depois.
        self.conv_layer_norms = nn.LayerNorm(embed_dim)

        #  Attention pooling: agregação ponderada de toda a sequência, ao invés de considerar apenas o último token.
        self.attention_blocks = nn.ModuleList(
            [AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.use_attention_pooling = use_attention_pooling
        if self.use_attention_pooling:
            self.attention_pool = AttentionPooling(embed_dim)

        # MLP de saída profundo: três camadas (128 →256 →128 →1) no lugar de uma projeção linear simples. -> Na experimental (na qual me baseio) é 160 -> 320 -> 160 -> 1
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        # Layer normalization após CNNs: adicionada para estabilização adicional das ativações.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, n_features)
        # 1. Project input features to embedding dimension
        x = self.input_projection(x)  # (batch_size, sequence_length, embed_dim)

        # 2. Applu convolution with residual connections
        x_conv = x.permute(0, 2, 1)

        for conv in self.convs:
            x_conv = x_conv + F.relu(conv(x_conv))

        x = x_conv.permute(0, 2, 1)
        x = self.conv_layer_norms(x)

        # 3. Add learnable positional encoding
        x = x + self.positional_embedding
        x = self.pos_dropout(x)

        # 4. Pass throught attention blocks
        for block in self.attention_blocks:
            x = block(x)

        # 5. Aggregate sequence information
        if self.use_attention_pooling:
            x = self.attention_pool(x)
        else:
            x = x[:, -1, :]  # Use last token

        return self.output_mlp(x)


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, embed_dim)
        weights = self.attention_weights(x)  # (batch_size, sequence_length, 1)
        weights = F.softmax(weights, dim=1)  # Normalizar pesos
        pooled = (weights * x).sum(dim=1)  # Agregar com pesos
        return pooled


class AttentionBlock(nn.Module):
    """Enhanced Transformer block with pre-normalization"""

    def __init__(
        self,
        embed_dimmension: int,
        num_heads: int,
        dropout: float = 0.1,
        ffn_expansion: int = 4,
    ):
        super(AttentionBlock, self).__init__()

        # Pre-normalization
        self.norm1 = nn.LayerNorm(embed_dimmension)
        self.norm2 = nn.LayerNorm(embed_dimmension)

        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dimmension, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dimmension, embed_dimmension * ffn_expansion),
            nn.GELU(),  # GELU instead of ReLU
            nn.Dropout(dropout),
            nn.Linear(embed_dimmension * ffn_expansion, embed_dimmension),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)

        # Pre-norm FFN with residual
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)

        return x
