
import torch


def uniform_sampling(frames, budget):
    """
    Select frames uniformly across a video.
    """
    total = frames.shape[0]

    if budget < 1 or budget > total:
        raise ValueError("Invalid frame budget.")

    indices = torch.linspace(
        0, total - 1, steps=budget
    ).round().long()

    return frames[indices]


def temporal_difference_sampling(frames, budget):
    """
    Select frames using temporal RGB differences.
    The first frame is retained as an anchor.
    """
    total = frames.shape[0]

    if budget < 1 or budget > total:
        raise ValueError("Invalid frame budget.")

    if budget == 1:
        return frames[:1]

    differences = torch.abs(
        frames[1:] - frames[:-1]
    ).mean(dim=(1, 2, 3))

    selected = torch.topk(
        differences,
        k=budget - 1,
        largest=True,
        sorted=False
    ).indices + 1

    indices = torch.cat([
        torch.tensor([0], device=frames.device),
        selected
    ])

    indices = torch.sort(indices).values

    return frames[indices]
