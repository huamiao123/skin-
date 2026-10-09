from __future__ import annotations


def window_sample_counts(dataset_size: int, batch_size: int, accum_steps: int, drop_last: bool = False) -> list[int]:
    if dataset_size <= 0 or batch_size <= 0 or accum_steps <= 0:
        raise ValueError("dataset_size, batch_size and accum_steps must be positive")
    if drop_last:
        batches = [batch_size] * (dataset_size // batch_size)
    else:
        batches = [batch_size] * (dataset_size // batch_size)
        if dataset_size % batch_size:
            batches.append(dataset_size % batch_size)
    return [sum(batches[i:i + accum_steps]) for i in range(0, len(batches), accum_steps)]


def batch_weight(dataset_size: int, batch_size: int, accum_steps: int, batch_index: int, drop_last: bool = False) -> float:
    windows = window_sample_counts(dataset_size, batch_size, accum_steps, drop_last)
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    window_index = batch_index // accum_steps
    if window_index >= len(windows):
        raise IndexError("batch_index is outside the loader")
    actual = batch_size
    if not drop_last and batch_index == sum(1 for _ in range(dataset_size // batch_size)):
        # This branch is only relevant when the final partial batch exists.
        actual = dataset_size % batch_size or batch_size
    return actual / windows[window_index]

