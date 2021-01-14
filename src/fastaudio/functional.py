import torch

# Must be imported explicitly to override the top-level `torch.fft` function
import torch.fft
from torch import Tensor


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


def mask_along_axis_(specgrams, num_masks, min_size, max_size, mask_val=None, axis=2):
    """Apply SpecAugment masks emitting from one axis.

    ``specgrams`` should be a tensor of shape (batch, channels, freq, time).
    ``axis`` must be either `2` or `3`.

    Masks are replaced with the mean of the masked area. Provide ``mask_val`` to
    mask with a specific value.

    """
    device = specgrams.device

    if axis != 3:
        # Orient so the axis we're masking comes last
        specgrams = specgrams.transpose(axis, -1)

    n, _, _, a = specgrams.shape

    # First create the broadcastable masks. Each spectrogram gets a different
    # set of masks (but the same masks span across channels).
    if num_masks == 1:
        masks = region_mask(n, min_size, max_size, a, device=device)
    else:
        # To create multiple masks per spectrogram, create a larger batch,
        # reshape, then merge.
        masks = (
            region_mask(
                num_masks * n,
                min_size,
                max_size,
                a,
                device=device,
            )
            .view(num_masks, n, a)
            .amax(dim=0)
        )
    # Expand so it can be broadcasted.
    masks = masks.view(n, 1, 1, a)

    if mask_val:
        specgrams.masked_fill_(masks, mask_val)
    else:
        # Mask with the channel-wise mean. Note while each channel takes the
        # same mask shape, the replacement value is determined per-channel.

        # Take the mean of the masked area, not the whole spectrogram, so as not
        # to change the overall mean (although this will affect the standard
        # deviation).
        mask_vals = (
            specgrams.mul(masks).sum((-2, -1))
            # This will be broadcast, so we have to manually multiply
            # by the size of that dimension.
            / (masks.sum((-2, -1)) * specgrams.shape[-2])
        )

        # Alternate method: mean of whole channel, not just masked area.
        # mask_vals = specgrams.mean(-2, -1)

        specgrams = torch.where(masks, mask_vals[..., None, None], specgrams)
    if axis == 3:
        return specgrams
    else:
        # Restore original orientation
        return specgrams.transpose(axis, -1)


def random_mask(shape, p, device=None):
    """Create a bool tensor with items set to 1 with probability ``p``.

    (Items are set individually.)

    ``p`` may be a float or a tensor broadcastable to ``shape``.

    """
    return torch.rand(shape, device=device) <= p


# TODO: Remove this & all uses, replace with torch.fft.rfftfreq
def _rfftfreq(n, d=1.0, device=None):
    """Get the sample frequencies for a discrete fourier transform."""
    val = 1.0 / (n * d)
    results = torch.arange(0, n // 2 + 1, device=device, dtype=int)
    return results * val


class NoiseColor:
    """Exponent beta values for named noise colors (see `colored_noise`)."""

    Violet = -2
    Blue = -1
    White = 0
    Pink = 1
    Brown = 2

    @staticmethod
    def valid(value: int) -> bool:
        """Is ``value`` a valid color for noise?"""
        return value in [*range(-2, 3)]


def colored_noise(shape, exponent, fmin=0, device=None):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance

    Ported to PyTorch from Felix Patzelt's numpy implementation, MIT Licensed.
    https://github.com/felixpatzelt/colorednoise/blob/acc9ec529b181dec2bd7fb914f87f0a62a0553d3/colorednoise.py

    Parameters:
    -----------

    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. It is not actually
        zero, but 1/samples.

    Returns
    -------

    out : array
        The samples.

    Examples:
    ---------

    # generate 1/f noise == pink noise == flicker noise
    >>> y = colored_noise([1], 5)

    """
    # Use `is` to allow for tensor exponents
    if exponent is NoiseColor.White:
        # White noise is simple - use a faster method.
        return torch.randn(shape)

    # The number of samples in each time series
    nsamples = shape[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    # FIXME: torch.fft is missing this for some reason
    # f = torch.fft.rfftfreq(nsamples)
    f = _rfftfreq(nsamples, device=device)

    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1.0 / nsamples)  # Low frequency cutoff
    ix = (s_scale < fmin).sum()  # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-exponent / 2.0)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].clone()
    w[-1] *= (1 + (nsamples % 2)) / 2.0  # correct f = +-0.5
    sigma = 2 * (w ** 2).sum().sqrt() / nsamples

    # Adjust size to generate one Fourier component per frequency
    new_shape = (*shape[:-1], f.size(0))
    # Original numpy imp
    # shape[-1] = f.size(0)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(new_shape) - 1
    s_scale = s_scale[[None] * dims_to_add + [Ellipsis]]

    # Generate scaled random power + phase
    sr = torch.randn(new_shape, device=device) * s_scale
    si = torch.randn(new_shape, device=device) * s_scale

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (nsamples % 2):
        si[..., -1] = 0

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0

    # Combine power + corrected phase to Fourier components
    s = sr + 1j * si

    # Transform to real time series & scale to unit variance
    y = torch.fft.irfft(s, n=nsamples, dim=-1) / sigma

    return y


def add_noise_(x: Tensor, color: NoiseColor, min_level: float, max_level: float, p=1.0):
    """Add colored noise to a batch of tensors, randomising per-item.

    Levels are relative to the standard deviation of each item.

    """
    n = x.size(0)

    noise_levels = (
        # Note that noise is scaled per-item, not per-channel. This means loud
        # channels will dwarf quiet channels.
        random_mask([n], p, device=x.device)
        * x.reshape(n, -1).std(dim=1)
        * (torch.rand([n], device=x.device) * (max_level - min_level) + min_level)
    ).reshape([n] + [1] * (x.dim() - 1))
    x += noise_levels * (colored_noise(x.shape, exponent=color, device=x.device) - 0.5)
    return x
