import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import entropy


def get_inception_probs(images: torch.Tensor, batch_size: int = 64, device: str = "cuda") -> np.ndarray:
    """
    Extract softmax class probabilities from Inception v3.

    Args:
        images: Tensor (N, 3, H, W) in [0, 1]. Any resolution — resized to 299x299 internally.
        batch_size: Images per forward pass.
        device: 'cuda' or 'cpu'.

    Returns:
        numpy array (N, 1000) of softmax class probabilities.
    """
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    model.aux_logits = False
    model.eval().to(device)

    resize = transforms.Resize((299, 299), antialias=True)

    probs = []
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch,) in loader:
            batch = resize(batch).to(device)
            logits = model(batch)
            p = F.softmax(logits, dim=1)
            probs.append(p.cpu().numpy())

    return np.concatenate(probs, axis=0)


def compute_is_from_probs(probs: np.ndarray, splits: int = 10) -> tuple:
    """
    Compute IS from pre-extracted softmax probabilities.

    IS = exp( E_x[ KL( p(y|x) || p(y) ) ] )

    Args:
        probs: (N, 1000) softmax probabilities.
        splits: Number of splits for mean/std estimation.

    Returns:
        (mean_is, std_is) — higher is better.
    """
    if len(probs) < splits:
        raise ValueError(
            f"Number of images ({len(probs)}) must be >= splits ({splits}). "
            "Reduce --is_splits or generate more samples."
        )
    chunks = np.array_split(probs, splits)
    split_scores = []
    for part in chunks:
        p_y = part.mean(axis=0)
        kl_divs = np.array([entropy(p_yx, p_y) for p_yx in part])
        split_scores.append(np.exp(kl_divs.mean()))
    return float(np.mean(split_scores)), float(np.std(split_scores))


def compute_is(images: torch.Tensor, batch_size: int = 64, device: str = "cuda", splits: int = 10) -> tuple:
    """
    End-to-end IS computation.

    Args:
        images: Tensor (N, 3, H, W) in [0, 1].
        batch_size: Batch size for Inception forward passes.
        device: 'cuda' or 'cpu'.
        splits: Splits for mean/std estimation.

    Returns:
        (mean_is, std_is).
    """
    probs = get_inception_probs(images, batch_size=batch_size, device=device)
    return compute_is_from_probs(probs, splits=splits)
