import numpy as np
import torch
from ot import sinkhorn2


def wasserstein(
    predictive_samples: torch.Tensor,
    observation: torch.Tensor,
    reg: float = 1e-1,
) -> torch.Tensor:
    """Compute Wasserstein distance between predictive samples and observation.
    Uses Sinkhorn distance with squared Euclidean cost and uniform weights.

    Uses NumPy implementation, see [1] for discussion of differences.

    Args:
        predictive_samples: Predictive samples
        observation: Observation

    Returns:
        Wasserstein distance
    """

    n = predictive_samples.shape[0]

    M = torch.norm(predictive_samples[:, None] - observation[None, :], dim=2)

    w1 = np.ones(n) / n
    w2 = np.ones(n) / n

    return torch.tensor([sinkhorn2(w1, w2, M.numpy(), reg).astype(np.float32)])
