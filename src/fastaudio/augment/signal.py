import colorednoise as cn
import torch
from fastai.imports import np, random
from fastai.vision.augment import RandTransform
from fastcore.transform import Transform
from fastcore.utils import mk_class, patch, store_attr

from ..core.signal import AudioTensor
from ..core.spectrogram import AudioSpectrogram

mk_class(
    "AudioPadType",
    **{o: o.lower() for o in ["Zeros", "Zeros_After", "Repeat"]},
    doc="All methods of padding audio as attributes to get tab-completion and typo-proofing",
)


class ResizeSignal(Transform):
    """Crops signal to be length specified in ms by duration, padding if needed"""

    def __init__(self, duration, pad_mode=AudioPadType.Zeros):  # noqa: F821
        store_attr()

    def encodes(self, ai: AudioTensor) -> AudioTensor:
        sig = ai.data
        orig_samples = ai.nsamples
        crop_samples = int((self.duration / 1000) * ai.sr)
        if orig_samples == crop_samples:
            return ai
        elif orig_samples < crop_samples:
            ai.data = _tfm_pad_signal(sig, crop_samples, pad_mode=self.pad_mode)
        else:
            crop_start = random.randint(0, int(orig_samples - crop_samples))
            ai.data = sig[:, crop_start : crop_start + crop_samples]
        return ai


def _tfm_pad_signal(sig, width, pad_mode=AudioPadType.Zeros):  # noqa: F821
    """Pad spectrogram to specified width, using specified pad mode"""
    c, x = sig.shape
    pad_m = pad_mode.lower()
    if pad_m in ["zeros", "zeros_after"]:
        zeros_front = random.randint(0, width - x) if pad_m == "zeros" else 0
        pad_front = torch.zeros((c, zeros_front))
        pad_back = torch.zeros((c, width - x - zeros_front))
        return torch.cat((pad_front, sig, pad_back), 1)
    elif pad_m == "repeat":
        repeats = width // x + 1
        return sig.repeat(1, repeats)[:, :width]
    else:
        raise ValueError(
            f"pad_mode {pad_m} not currently supported, only 'zeros', 'zeros_after', or 'repeat'"
        )


def _shift(sig, s):
    if s == 0:
        return sig
    out = torch.zeros_like(sig)
    if s < 0:
        out[..., :s] = sig[..., -s:]
    else:
        out[..., s:] = sig[..., :-s]
    return out


def shift_signal(t: torch.Tensor, shift, roll):
    # refactor 2nd half of this statement to just take and roll the final axis
    if roll:
        t.data = torch.from_numpy(np.roll(t.numpy(), shift, axis=-1))
    else:
        t.data = _shift(t, shift)
    return t


class SignalShifter(RandTransform):
    """Randomly shifts the audio signal by `max_pct` %.
    direction must be -1(left) 0(bidirectional) or 1(right).
    """

    def __init__(self, p=0.5, max_pct=0.2, max_time=None, direction=0, roll=False):
        if direction not in [-1, 0, 1]:
            raise ValueError("Direction must be -1(left) 0(bidirectional) or 1(right)")
        store_attr(but="p")
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.shift_factor = random.uniform(-1, 1)
        if self.direction != 0:
            self.shift_factor = self.direction * abs(self.shift_factor)

    def encodes(self, ai: AudioTensor):
        if self.max_time is None:
            s = self.shift_factor * self.max_pct * ai.nsamples
        else:
            s = self.shift_factor * self.max_time * ai.sr
        ai.data = shift_signal(ai.data, int(s), self.roll)
        return ai

    def encodes(self, sg: AudioSpectrogram):
        if self.max_time is None:
            s = self.shift_factor * self.max_pct * sg.width
        else:
            s = self.shift_factor * self.max_time * sg.sr
        return shift_signal(sg, int(s), self.roll)


mk_class(
    "NoiseColor",
    **{o: i - 2 for i, o in enumerate(["Violet", "Blue", "White", "Pink", "Brown"])},
    doc="All possible colors of noise as attributes to get tab-completion and typo-proofing",
)


class AddNoise(Transform):
    "Adds noise of specified color and level to the audio signal"

    def __init__(self, noise_level=0.05, color=NoiseColor.White):  # noqa: F821
        store_attr()

    def encodes(self, ai: AudioTensor) -> AudioTensor:
        # if it's white noise, implement our own for speed
        if self.color == 0:
            noise = torch.randn_like(ai.data)
        else:
            noise = torch.from_numpy(
                cn.powerlaw_psd_gaussian(exponent=self.color, size=ai.nsamples)
            ).float()
        scaled_noise = noise * ai.data.abs().mean() * self.noise_level
        ai.data += scaled_noise
        return ai


@patch
def apply_gain(ai: AudioTensor, gain):
    ai.data *= gain
    return ai


class ChangeVolume(RandTransform):
    "Changes the volume of the signal"

    def __init__(self, p=0.5, lower=0.5, upper=1.5):
        self.lower, self.upper = lower, upper
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.gain = random.uniform(self.lower, self.upper)

    def encodes(self, ai: AudioTensor):
        return apply_gain(ai, self.gain)


@patch
def cutout(ai: AudioTensor, cut_pct):
    mask = torch.zeros(int(ai.nsamples * cut_pct))
    mask_start = random.randint(0, ai.nsamples - len(mask))
    ai.data[:, mask_start : mask_start + len(mask)] = mask
    return ai


# @patch
# def cutout(sg:AudioSpectrogram, cut_pct):


class SignalCutout(RandTransform):
    "Randomly zeros some portion of the signal"

    def __init__(self, p=0.5, max_cut_pct=0.15):
        self.max_cut_pct = max_cut_pct
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.cut_pct = random.uniform(0, self.max_cut_pct)

    def encodes(self, ai: AudioTensor):
        return cutout(ai, self.cut_pct)


@patch
def lose_signal(ai: AudioTensor, loss_pct):
    mask = (torch.rand_like(ai.data[0]) > loss_pct).float()
    ai.data[..., :] *= mask
    return ai


class SignalLoss(RandTransform):
    "Randomly loses some portion of the signal"

    def __init__(self, p=0.5, max_loss_pct=0.15):
        self.max_loss_pct = max_loss_pct
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.loss_pct = random.uniform(0, self.max_loss_pct)

    def encodes(self, ai: AudioTensor):
        return lose_signal(ai, self.loss_pct)


# downmixMono was removed from torchaudio, we now just take the mean across channels
# this works for both batches and individual items
class DownmixMono(Transform):
    "Transform multichannel audios into single channel"

    def encodes(self, ai: AudioTensor) -> AudioTensor:
        downmixed = ai.data.contiguous().mean(-2).unsqueeze(-2)
        return AudioTensor(downmixed, ai.sr)
