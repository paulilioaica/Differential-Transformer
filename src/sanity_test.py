from transformer import Transformer
import torch

def test_transformer_with_embeddings():
    # Hyperparameters
    num_layers = 2
    num_heads = 4
    seq_len = 10
    vocab_size = 50  # Let's assume the size of vocabulary is 50 for this test
    embedding_dim = 64  # Embedding dimension

    # Create the model
    model = Transformer(num_layers, num_layers, embedding_dim, num_heads, seq_len, vocab_size, embedding_dim)
    
    # Create some inputs
    x = torch.randint(vocab_size, (1, seq_len))  # batch size = 1
    y = torch.randint(vocab_size, (1, seq_len))  # batch size = 1

    # Forward pass
    output = model(x, y)

    # Check output size
    assert output.shape == torch.Size([1, seq_len, vocab_size]), f"Expected output shape (1, {seq_len}, {vocab_size}), but got {output.shape}"
    
    print("Transformer with embeddings test passed.")
test_transformer_with_embeddings()


