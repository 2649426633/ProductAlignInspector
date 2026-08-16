from __future__ import annotations

import math

import numpy as np
import torch


def approximate_greedy_coreset(
    features: np.ndarray,
    *,
    ratio: float = 0.10,
    device: str | torch.device = "cpu",
    projection_dim: int = 64,
    starting_points: int = 10,
    seed: int = 42,
    max_candidates: int = 20000,
) -> np.ndarray:
    """PatchCore-style approximate greedy coreset selection.

    The implementation is adapted from the previous ``patchcores`` project's
    ``ApproximateGreedyCoresetSampler``. It adds deterministic seeding and an
    optional candidate cap so ROI-DINO memory construction remains practical on
    CPU when many normal source images are supplied.
    """
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"features must be non-empty [N,D], got {array.shape}")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if ratio >= 0.999999:
        return np.ascontiguousarray(array)

    rng = np.random.default_rng(seed)
    original = array

    if len(array) > max_candidates:
        candidate_indices = rng.choice(len(array), size=max_candidates, replace=False)
        array = array[candidate_indices]
    else:
        candidate_indices = np.arange(len(array))

    target_count = max(1, int(math.ceil(len(original) * ratio)))
    target_count = min(target_count, len(array))

    torch_device = torch.device(device)
    tensor = torch.from_numpy(array).to(torch_device)

    if tensor.shape[1] > projection_dim:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        projection = torch.randn(
            tensor.shape[1],
            projection_dim,
            generator=generator,
            dtype=torch.float32,
        ) / math.sqrt(float(projection_dim))
        projected = tensor @ projection.to(torch_device)
    else:
        projected = tensor

    n_start = min(int(starting_points), len(projected))
    starts_np = rng.choice(len(projected), size=n_start, replace=False)
    starts = torch.as_tensor(starts_np, dtype=torch.long, device=torch_device)

    with torch.no_grad():
        initial = torch.cdist(projected, projected[starts], p=2)
        min_dist = initial.mean(dim=1)
        selected: list[int] = []

        for _ in range(target_count):
            idx = int(torch.argmax(min_dist).item())
            selected.append(idx)
            dist = torch.cdist(projected, projected[idx : idx + 1], p=2).squeeze(1)
            min_dist = torch.minimum(min_dist, dist)

    selected_original = candidate_indices[np.asarray(selected, dtype=np.int64)]
    return np.ascontiguousarray(original[selected_original], dtype=np.float32)
