import librosa
import torch
import warnings
from fastai.imports import partial, random
from fastcore.transform import Transform
from fastcore.utils import ifnone
from torch.nn import functional as F
from torchaudio.functional import compute_deltas

from ..core.spectrogram import AudioSpectrogram, AudioTensor
from ..util import auto_batch
from .functional import mask_along_axis_
from .signal import AudioPadType


class SpectrogramTransform(Transform):
    "Helps prevent us trying to apply to Audio Tensors"

    def encodes(self, audio: AudioTensor) -> AudioTensor:
        warnings.warn(
            f"You are trying to apply a {type(self).__name__} to an AudioTensor {audio.shape}"
        )
        return audio


class CropTime(SpectrogramTransform):
    """Random crops full spectrogram to be length specified in ms by crop_duration"""

    def __init__(self, duration, pad_mode=AudioPadType.Zeros):
        self.duration = duration
        self.pad_mode = pad_mode

    def encodes(self, sg: AudioSpectrogram) -> AudioSpectrogram:
        sr, hop = sg.sr, sg.hop_length
        w_crop = int((sr * self.duration) / (1000 * hop)) + 1
        w_sg = sg.shape[-1]
        if w_sg == w_crop:
            sg_crop = sg
        elif w_sg < w_crop:
            sg_crop = _tfm_pad_spectro(sg, w_crop, pad_mode=self.pad_mode)
        else:
            crop_start = random.randint(0, int(w_sg - w_crop))
            sg_crop = sg[:, :, crop_start : crop_start + w_crop]
            sg_crop.sample_start = int(crop_start * hop)
            sg_crop.sample_end = sg_crop.sample_start + int(self.duration * sr)
        sg.data = sg_crop
        return sg


def _tfm_pad_spectro(sg, width, pad_mode=AudioPadType.Zeros):
    """Pad spectrogram to specified width, using specified pad mode"""
    c, y, x = sg.shape
    if pad_mode in [AudioPadType.Zeros, AudioPadType.Zeros_After]:
        padded = torch.zeros((c, y, width))
        start = random.randint(0, width - x) if pad_mode == AudioPadType.Zeros else 0
        padded[:, :, start : start + x] = sg.data
        return padded
    elif pad_mode == AudioPadType.Repeat:
        repeats = width // x + 1
        return sg.repeat(1, 1, repeats)[:, :, :width]
    else:
        raise ValueError(
            f"""pad_mode {pad_mode} not currently supported,
            only AudioPadType.Zeros, AudioPadType.Zeros_After,
            or AudioPadType.Repeat"""
        )


class MaskFreq(SpectrogramTransform):
    """Google SpecAugment frequency masking from https://arxiv.org/abs/1904.08779."""

    def __init__(self, num_masks=1, size=20, start=None, val=None):
        self.num_masks = num_masks
        self.size = size
        self.start = start
        self.val = val

    def encodes(self, sg: AudioSpectrogram) -> AudioSpectrogram:
        channel_mean = sg.contiguous().view(sg.size(0), -1).mean(-1)[:, None, None]
        mask_val = ifnone(self.val, channel_mean)
        c, y, x = sg.shape
        # Position of the first mask
        start = ifnone(self.start, random.randint(0, y - self.size))
        for _ in range(self.num_masks):
            mask = torch.ones(self.size, x) * mask_val
            if not 0 <= start <= y - self.size:
                raise ValueError(
                    f"Start value '{start}' out of range for AudioSpectrogram of shape {sg.shape}"
                )
            sg[:, start : start + self.size, :] = mask
            # Setting start position for next mask
            start = random.randint(0, y - self.size)
        return sg


class MaskTime(SpectrogramTransform):
    """Google SpecAugment time masking from https://arxiv.org/abs/1904.08779."""

    def __init__(self, num_masks=1, size=20, start=None, val=None):
        self.num_masks = num_masks
        self.size = size
        self.start = start
        self.val = val

    def encodes(self, sg: AudioSpectrogram) -> AudioSpectrogram:
        sg.data = torch.einsum("...ij->...ji", sg)
        sg.data = MaskFreq(
            self.num_masks,
            self.size,
            self.start,
            self.val,
        )(sg)
        sg.data = torch.einsum("...ij->...ji", sg)
        return sg


class _MaskAxisGPU(SpectrogramTransform):
    """Base class for GPU-based SpecAugment masking transforms."""

    def __init__(self, axis, num_masks, min_size, max_size, mask_val):
        if axis not in [2, 3]:
            raise ValueError("Can only mask the time or frequency axis (2 or 3)")
        self.num_masks = num_masks
        self.min_size = min_size
        self.max_size = max_size
        self.mask_val = mask_val
        self.axis = axis
        super().__init__()

    @auto_batch(3)
    def encodes(self, sg: AudioSpectrogram):
        return mask_along_axis_(
            sg,
            num_masks=self.num_masks,
            min_size=self.min_size,
            max_size=self.max_size,
            mask_val=self.mask_val,
            axis=self.axis,
        )


class MaskFreqGPU(_MaskAxisGPU):
    """Google SpecAugment frequency masking from https://arxiv.org/abs/1904.08779.

    This version runs on batches and can be run efficiently on the GPU.

    """

    def __init__(self, num_masks=1, min_size=1, max_size=10, mask_val=None):
        super().__init__(2, num_masks, min_size, max_size, mask_val)


class MaskTimeGPU(_MaskAxisGPU):
    """Google SpecAugment time masking from https://arxiv.org/abs/1904.08779.

    This version runs on batches and can be run efficiently on the GPU.

    """

    def __init__(self, num_masks=1, min_size=1, max_size=10, mask_val=None):
        super().__init__(3, num_masks, min_size, max_size, mask_val)


class SGRoll(SpectrogramTransform):
    """Shifts spectrogram along x-axis wrapping around to other side"""

    def __init__(self, max_shift_pct=0.5, direction=0):
        if int(direction) not in [-1, 0, 1]:
            raise ValueError("Direction must be -1(left) 0(bidirectional) or 1(right)")
        self.max_shift_pct = max_shift_pct
        self.direction = direction

    def encodes(self, sg: AudioSpectrogram) -> AudioSpectrogram:
        direction = random.choice([-1, 1]) if self.direction == 0 else self.direction
        w = sg.shape[-1]
        roll_by = int(w * random.random() * self.max_shift_pct * direction)
        sg.data = sg.roll(roll_by, dims=-1)
        return sg


def _torchdelta(sg: AudioSpectrogram, order=1, width=9):
    """Converts to numpy, takes delta and converts back to torch, needs torchification"""
    if sg.shape[1] < width:
        raise ValueError(
            f"""Delta not possible with current settings, inputs must be wider than
        {width} columns, try setting max_to_pad to a larger value to ensure a minimum width"""
        )
    return AudioSpectrogram(
        torch.from_numpy(librosa.feature.delta(sg.numpy(), order=order, width=width))
    )


class Delta(SpectrogramTransform):
    """Creates delta with order 1 and 2 from spectrogram
    and concatenate with the original"""

    def __init__(self, width=9):
        self.td = partial(_torchdelta, width=width)

    def encodes(self, sg: AudioSpectrogram):
        new_channels = [
            torch.stack([c, self.td(c, order=1), self.td(c, order=2)]) for c in sg
        ]
        sg.data = torch.cat(new_channels, dim=0)
        return sg


class DeltaGPU(SpectrogramTransform):
    """Adds extra channels with delta (orders 1 and 2) to spectrogram."""

    def __init__(self, width=9, mode="replicate"):
        self.width = width
        self.mode = mode

    @auto_batch(3)
    def encodes(self, sg: AudioSpectrogram):
        delta = compute_deltas(sg, win_length=self.width, mode=self.mode)
        delta2 = compute_deltas(delta, win_length=self.width, mode=self.mode)
        sg.data = torch.cat([sg, delta, delta2], dim=1).contiguous()
        return sg


class TfmResize(SpectrogramTransform):
    """Temporary fix to allow image resizing transform"""

    def __init__(self, size, interp_mode="bilinear"):
        self.size = size
        self.interp_mode = interp_mode

    def encodes(self, sg: AudioSpectrogram) -> AudioSpectrogram:
        if isinstance(self.size, int):
            self.size = (self.size, self.size)
        c, y, x = sg.shape
        sg.data = F.interpolate(
            sg.unsqueeze(0), size=self.size, mode=self.interp_mode, align_corners=False
        ).squeeze(0)
        return sg


class TfmResizeGPU(SpectrogramTransform):
    """Resize the spectrogram using interpolation."""

    def __init__(self, size, interp_mode="bilinear"):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interp_mode = interp_mode

    @auto_batch(3)
    def encodes(self, sg: AudioSpectrogram):
        sg.data = F.interpolate(
            sg, size=self.size, mode=self.interp_mode, align_corners=False
        )
        return sg
