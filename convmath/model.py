import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConvMathConfig:
    vocab_size: int = 544
    d_model: int = 512
    num_decoder_layers: int = 7
    max_seq_len: int = 256
    start_token: int = 1
    end_token: int = 2


class ResidualBlock(nn.Module):
    """Basic residual block."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ImageEncoder(nn.Module):
    """ResNet-based image encoder for feature extraction."""

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.blocks = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128, stride=2),

            ResidualBlock(128, 256, stride=1),
            ResidualBlock(256, 256, stride=2),

            ResidualBlock(256, 512, stride=1),
            ResidualBlock(512, 512, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.blocks(x)


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)                                 # x: (batch, seq_len, d_model)
        pe = self.pe[:seq_len, :].to(x.dtype).to(x.device)  # (seq_len, d_model)
        return x + pe.unsqueeze(0)                          # broadcast to (batch, seq_len, d_model)
    

class CausalGLU(nn.Module):
    """Causal Gated Linear Unit with 1D convolution."""

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        
        # Use causal padding (left-padding only)
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # Full left padding
        self.conv = nn.Conv1d(d_model, 2 * d_model, kernel_size=kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)  # (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        
        # Apply causal padding manually
        x_t = F.pad(x_t, (self.padding, 0))
        
        conv_out = self.conv(x_t)  # (batch, 2*d_model, seq_len)
        conv_out = conv_out.transpose(1, 2)  # (batch, seq_len, 2*d_model)
        
        a, b = conv_out.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class ConvDecoderLayer(nn.Module):
    """Single convolutional decoder layer with causal GLU, layer norm, residual, and cross-attention."""

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()

        self.glu = CausalGLU(d_model, kernel_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Cross-attention projections
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # First sublayer: Causal GLU with residual and norm
        residual = x
        x = self.glu(x)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        # Second sublayer: Cross-attention with residual and norm
        residual = x
        
        queries = self.query_proj(x)
        keys = self.key_proj(encoder_outputs)
        values = self.value_proj(encoder_outputs)
        
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(x.size(-1))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.bmm(attn_weights, values)
        context = self.out_proj(context)
        context = self.dropout(context)
        
        x = self.norm2(residual + context)
        
        return x, attn_weights


class ConvMath(nn.Module):
    """ConvMath model for mathematical expression recognition."""

    def __init__(self, config: Optional[ConvMathConfig] = None):
        super().__init__()

        config = config or ConvMathConfig()
        self.config = config
        self.d_model = config.d_model

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = PositionalEmbedding(config.d_model, max_len=config.max_seq_len)

        self.image_encoder = ImageEncoder()
        self.feature_proj = nn.Linear(512, self.d_model)

        self.decoder_layers = nn.ModuleList([
            ConvDecoderLayer(config.d_model, kernel_size=3) for _ in range(config.num_decoder_layers)
        ])

        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        """Selective initialization - avoid BatchNorm and embeddings."""
        for name, module in self.named_modules():
            # Skip BatchNorm - it has good defaults
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                continue
            
            # Initialize embeddings with normal distribution
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)
            
            # Initialize linear and conv layers
            elif isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor, target_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            images: (batch, 1, H, W)
            target_tokens: (batch, tgt_len) (optional) - if provided, do teacher-forcing training forward
        """
        encoder_outputs = self._encode_images(images)

        if target_tokens is not None:
            return self._forward_train(encoder_outputs, target_tokens)
        else:
            return self._generate(encoder_outputs)

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors and add positional embeddings."""
        features = self.image_encoder(images)
        batch_size, channels, h, w = features.size()

        features = features.view(batch_size, channels, -1).transpose(1, 2)
        features = self.feature_proj(features)
        features = self.dropout(features)
        features = self.pos_embedding(features)

        return features

    def _forward_train(self, encoder_outputs: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Training forward pass with teacher forcing."""
        # Reduce scaling factor for stability
        token_emb = self.token_embedding(target_tokens)
        decoder_input = self.pos_embedding(token_emb)
        decoder_input = self.dropout(decoder_input)

        h = decoder_input

        for layer in self.decoder_layers:
            h, _ = layer(h, encoder_outputs)

        logits = self.output_proj(h)
        return logits

    def _generate(self, encoder_outputs: torch.Tensor, max_len: int = 256) -> torch.Tensor:
        """Autoregressive generation (greedy decoding)."""
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        generated = torch.full((batch_size, 1), self.config.start_token, dtype=torch.long, device=device)

        for _step in range(max_len):
            token_emb = self.token_embedding(generated)
            decoder_input = self.pos_embedding(token_emb)

            h = decoder_input
            for layer in self.decoder_layers:
                h, _ = layer(h, encoder_outputs)

            logits = self.output_proj(h[:, -1:, :])
            next_token = torch.argmax(logits, dim=-1)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token.squeeze(1) == self.config.end_token).all():
                break

        return generated[:, 1:]


if __name__ == "__main__":
    config = ConvMathConfig()
    model = ConvMath(config)
    
    images = torch.rand(2, 1, 128, 512)
    output = model(images)
    print(f"Generation output shape: {output.shape}")

    target_tokens = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(images, target_tokens)
    print(f"Training logits shape: {logits.shape}")