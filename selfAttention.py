import numpy as np
import polars as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# Arquitetura avançada TCC
class Rebuilt_SAIDNN(nn.Module):
    def __init__(
        self,
        embedded_dimmension=160,  # 64, 128, 256
        num_heads=10,  # 4, 8, 16
        num_blocks=3,  # 2, 3, 4
        learning_rate=0.005,  # 0.00001, 0.0005, 0.0003
        dropout=0.2,  # 0.1, 0.2, 0.3
        number_features=5,  # 4, 5, 6f
        use_attention_polling=True,
        sequence_length=10,
    ):
        # CNNs com conexões residuais: três blocos convolucionais com skip connections.

        self.input_projection = nn.Linear(number_features, embedded_dimmension)

        NUMBER_CNN_BLOCKS = 3
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedded_dimmension,
                    out_channels=embedded_dimmension,
                    kernel_size=3,
                    padding=1,
                )
                for _ in range(NUMBER_CNN_BLOCKS)
            ]
        )

        # Positional embeddings aprendíveis: substituição do encoding sinusoidal fixo por parâmetros treináveis
        self.positional_embedding = nn.Parameter(
            torch.randn(1, sequence_length, embedded_dimmension)
        )
        self.pos_dropout = nn.Dropout(dropout)

        # Pre-Layer Normalization: normalização aplicada antes das subcamadas, em vez de depois.
        self.conv_layer_norms = nn.LayerNorm(embedded_dimmension)

        #  Attention pooling: agregação ponderada de toda a sequência, ao invés de considerar apenas o último token.
        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(embedded_dimmension, num_heads, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.use_attention_polling = use_attention_polling
        if self.use_attention_polling:
            self.attention_pool = AttentionPooling(embedded_dimmension)

        # MLP de saída profundo: três camadas (128 →256 →128 →1) no lugar de uma projeção linear simples. -> Na experimental (na qual me baseio) é 160 -> 320 -> 160 -> 1
        self.output_mlp = nn.Sequential(
            nn.Linear(embedded_dimmension, embedded_dimmension * 2),
            nn.ReLU(),
            nn.Linear(embedded_dimmension * 2, embedded_dimmension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedded_dimmension, 1),
        )

        # Layer normalization após CNNs: adicionada para estabilização adicional das ativações.

    def foward(self, x):
        # x shape: (batch_size, sequence_length, number_features)
        # 1. Project input features to embedding dimension
        x = self.input_projection(
            x
        )  # (batch_size, sequence_length, embedded_dimmension)

        # 2. Applu convolution with residual connections
        x_conv = x.permutate(0, 2, 1)

        for conv in self.convs:
            x_conv += F.relu(conv(x_conv))

        x = x_conv.permutate(0, 2, 1)
        x = self.conv_layer_norms(x)

        # 3. Add learnable positional encoding
        x = x + self.positional_embedding
        x = self.pos_dropout(x)

        # 4. Pass throught attention blocks
        for block in self.attention_blocks:
            x = block(x)

        # 5. Aggregate sequence information
        if self.use_attention_polling:
            x = self.attention_pool(x)
        else:
            x = x[:, -1, :]  # Use last token

        return self.output_mlp(x)


class AttentionPooling(nn.Module):
    def __init__(self, embedded_dimmension):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.linear(embedded_dimmension, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embedded_dimmension)
        weights = self.attention_weights(x)  # (batch_size, sequence_length, 1)
        weights = F.softmax(weights, dim=1)  # Normalizar pesos
        pooled = (weights * x).sum(dim=1)  # Agregar com pesos
        return pooled


class AttentionBlock(nn.Module):
    """Enhanced Transformer block with pre-normalization"""

    def __init__(self, embed_dimmension, num_heads, dropout=0.1, ffn_expansion=4):
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

    def forward(self, x):
        # Pre-norm attention with residual
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)

        # Pre-norm FFN with residual
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)

        return x
