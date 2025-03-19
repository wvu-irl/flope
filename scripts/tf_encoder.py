import torch
import torch.nn as nn

# Define the Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, out_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, model_dim)
        
        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.out_layer = nn.Linear(model_dim, out_dim)

    def forward(self, x):
        x = self.embedding(x)
        transformer_out = self.transformer_encoder(x)
        output = self.out_layer(transformer_out)
        return output

# Example usage
if __name__ == "__main__":
    # Input dimensions
    batch_size = 8
    seq_len = 10
    input_dim = 16  # Input feature size
    model_dim = 32  # Embedding/Model feature size
    num_heads = 4   # Number of attention heads
    num_layers = 2  # Number of Transformer Encoder layers
    ff_dim = 64     # Feedforward network dimension
    out_dim = 9

    # Dummy input tensor (sequence_length, batch_size, input_dim)
    input_tensor = torch.rand(batch_size, seq_len, input_dim)

    # Instantiate the Transformer Encoder
    transformer_encoder = TransformerEncoder(
        input_dim, model_dim, out_dim,
        num_heads, num_layers, ff_dim
    )

    # Forward pass
    output = transformer_encoder(input_tensor)
    print("Output shape:", output.shape)
