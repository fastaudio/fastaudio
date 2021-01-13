import torch


def random_mask(shape, p, device=None):
    """Create a bool tensor with items set to 1 with probability ``p``.

    (Items are set individually.)

    ``p`` may be a float or a tensor broadcastable to ``shape``.

    """
    return torch.rand(shape, device=device) <= p
