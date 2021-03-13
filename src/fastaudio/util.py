import torch
from fastai.vision.augment import RandTransform
from functools import wraps
from math import pi
from torch import Tensor

from fastaudio.core.signal import AudioTensor

__all__ = ["test_audio_tensor"]


def create_sin_wave(seconds=5, sr=16000, freq=400):
    "Creates a sin wave to be used for tests"
    max_samples = freq * seconds * 2 * pi
    rate = 2 * pi * freq / sr
    samples = torch.arange(0, max_samples, rate, dtype=torch.float)
    sin_wave = torch.sin(samples)
    return sin_wave, sr


def test_audio_tensor(seconds=2, sr=16000, channels=1):
    "Generates an Audio Tensor for testing that is based on a sine wave"
    sin_wave, sr = create_sin_wave(seconds=seconds, sr=sr)
    sin_wave = sin_wave.repeat(channels, 1)
    return AudioTensor(sin_wave, sr)


def apply_transform(transform, inp):
    """Generate a new input, apply transform, and display/return both input and output"""
    inp_orig = inp.clone()
    out = (
        transform(inp_orig, split_idx=0)
        if isinstance(transform, RandTransform)
        else transform(inp_orig)
    )
    return inp.clone(), out


def auto_batch(item_dims):
    """Wrapper that always calls the underlying function with a batch.

    Single items are expanded before calling, then the result is squashed back
    to the original dimensions. Batches are passed without modification. This
    allows the underlying function to be written once (for batches) and
    automatically work on both.

    The overhead of this wrapper is very small and will be dwarfed by most
    operations on reasonably-sized tensors.

    ``item_dims`` specifies the number of dimensions in an item.

    """

    def wrapper(orig_func):
        @wraps(orig_func)
        def expand_and_do(self, x: Tensor):
            nonlocal orig_func, item_dims
            is_item = x.dim() == item_dims
            if is_item:
                # Expand to batch
                x = x.unsqueeze(0)

            x = orig_func(self, x)

            if is_item:
                # Restore original shape
                return x.squeeze(0)
            else:
                return x

        return expand_and_do

    return wrapper
