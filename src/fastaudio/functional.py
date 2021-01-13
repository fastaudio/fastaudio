import torch


def region_mask(n, min_mask_size, max_mask_size, maskable_length, device=None):
    """Create a vector of ``n`` random region masks.

    Masks are returned in a vector with shape `(n, maskable_length)`.

    The mask vectors are boolean vectors of ``maskable_length`` with a
    continuous region of 1s between ``min_mask_size`` and ``max_mask_size``
    (inclusive).

    """
    # Generate the start & end positions for each mask, then compare these to
    # absolute indices to create the actual mask vectors.
    mask_sizes = (
        torch.rand([n, 1], device=device) * (max_mask_size - min_mask_size)
        + min_mask_size
    )
    mask_starts = torch.rand([n, 1], device=device) * (maskable_length - mask_sizes)
    mask_ends = mask_starts + mask_sizes
    indexes = torch.arange(0, maskable_length, device=device)
    return (mask_starts <= indexes) & (indexes < mask_ends)


def random_mask(shape, p, device=None):
    """Create a bool tensor with items set to 1 with probability ``p``.

    (Items are set individually.)

    ``p`` may be a float or a tensor broadcastable to ``shape``.

    """
    return torch.rand(shape, device=device) <= p
