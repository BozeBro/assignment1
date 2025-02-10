import numpy as np


def softmax_loss(X, Y, mini_batch):
    """
    Compute the average softmax (cross-entropy) loss.

    Args:
        X: Input logits, shape (mini_batch, num_classes)
        Y: One-hot encoded labels, shape (mini_batch, num_classes)
        mini_batch: Number of rows in X and Y (batch size)

    Returns:
        Average loss (scalar)
    """
    # Compute softmax probabilities
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # Numerical stability
    softmax_probs = exp_X / np.sum(exp_X, axis=1, keepdims=True)

    # Compute cross-entropy loss
    log_probs = np.log(softmax_probs)
    loss = -np.sum(Y * log_probs) / mini_batch  # Average loss

    return loss


# Example inputs
X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])  # Logits (mini_batch=2, num_classes=3)
Y = np.array([[0, 1, 0], [0, 0, 1]])  # One-hot labels (mini_batch=2, num_classes=3)
mini_batch = 2  # Number of rows in X and Y


loss = softmax_loss(X, Y, mini_batch)
print(loss)
