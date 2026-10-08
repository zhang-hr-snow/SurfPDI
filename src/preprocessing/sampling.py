from __future__ import annotations

import numpy as np


def farthest_point_indices(points: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    if len(points) <= k:
        return np.arange(len(points), dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = np.empty(k, dtype=np.int64)
    selected[0] = int(rng.integers(0, len(points)))
    min_dist2 = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for i in range(1, k):
        selected[i] = int(np.argmax(min_dist2))
        dist2 = np.sum((points - points[selected[i]]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return selected


def residue_wise_sample(
    surf_xyz: np.ndarray,
    surf_res_idx: np.ndarray,
    residue_pos: np.ndarray | None,
    method: str,
    max_points_per_residue: int | None,
    seed: int = 42,
) -> np.ndarray:
    if method == "full" or max_points_per_residue is None:
        return np.arange(len(surf_xyz), dtype=np.int64)

    selected_global = []
    rng = np.random.default_rng(seed)
    for residue_idx in np.unique(surf_res_idx):
        local_indices = np.where(surf_res_idx == residue_idx)[0]
        if len(local_indices) <= max_points_per_residue:
            selected_global.append(local_indices)
            continue

        local_xyz = surf_xyz[local_indices]
        if method == "fps":
            local_selected = farthest_point_indices(
                local_xyz,
                max_points_per_residue,
                seed=seed + int(residue_idx),
            )
        elif method == "random":
            local_selected = rng.choice(len(local_indices), size=max_points_per_residue, replace=False)
        elif method == "nearest":
            if residue_pos is None:
                raise ValueError("nearest sampling requires residue_pos.")
            center = residue_pos[int(residue_idx)]
            dist2 = np.sum((local_xyz - center) ** 2, axis=1)
            local_selected = np.argsort(dist2)[:max_points_per_residue]
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        selected_global.append(local_indices[np.asarray(local_selected, dtype=np.int64)])

    return np.sort(np.concatenate(selected_global).astype(np.int64))

