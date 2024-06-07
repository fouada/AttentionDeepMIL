# Add this to the bottom of model.py
from model import Attention
if __name__ == "__main__":
    # Example usage
    import torch

    # Create dummy input data
    dummy_data = torch.randn(1, 1, 28, 28)  # Example for MNIST size (1, 1, 28, 28)
    
    # Initialize model
    model = Attention()
    
    # Forward pass
    y_prob, y_hat, attention_weights = model(dummy_data)
    
    print(f"y_prob: {y_prob}")
    print(f"y_hat: {y_hat}")
    print(f"attention_weights: {attention_weights}")
