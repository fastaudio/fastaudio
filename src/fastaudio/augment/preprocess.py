import torch
from enum import Enum
from fastai.imports import math, np
from fastcore.transform import Transform
from librosa.effects import split
from scipy.signal import resample_poly

from ..core.signal import AudioTensor


class RemoveType(Enum):
    "All methods of removing silence as attributes to get tab-completion and typo-proofing"
    Trim = "trim"
    All = "all"
    Split = "split"


def _merge_splits(splits, pad):
    clip_end = splits[-1][1]
    merged = []
    i = 0
    while i < len(splits):
        start = splits[i][0]
        while splits[i][1] < clip_end and splits[i][1] + pad >= splits[i + 1][0] - pad:
            i += 1
        end = splits[i][1]
        merged.append(np.array([max(start - pad, 0), min(end + pad, clip_end)]))
        i += 1
    return np.stack(merged)


class RemoveSilence(Transform):
    """Split signal at points of silence greater than 2*pad_ms """

    def __init__(self, remove_type=RemoveType.Trim, threshold=20, pad_ms=20):
        self.remove_type = remove_type
        self.threshold = threshold
        self.pad_ms = pad_ms

    def encodes(self, ai: AudioTensor) -> AudioTensor:
        if self.remove_type is None:
            return ai
        padding = int(self.pad_ms / 1000 * ai.sr)
        if padding > ai.nsamples:
            return ai
        if ai.shape[0] < 2:
            splits = split(ai[0].numpy(), top_db=self.threshold, hop_length=padding)
        else:
            splits = split(ai.numpy(), top_db=self.threshold, hop_length=padding)
        if self.remove_type == RemoveType.Split:
            sig = [
                ai[:, (max(a - padding, 0)) : (min(b + padding, ai.nsamples))]
                for (a, b) in _merge_splits(splits, padding)
            ]
        elif self.remove_type == RemoveType.Trim:
            sig = [ai[:, (max(splits[0, 0] - padding, 0)) : splits[-1, -1] + padding]]
        elif self.remove_type == RemoveType.All:
            sig = [
                torch.cat(
                    [
                        ai[:, (max(a - padding, 0)) : (min(b + padding, ai.nsamples))]
                        for (a, b) in _merge_splits(splits, padding)
                    ],
                    dim=1,
                )
            ]
        else:
            raise ValueError(
                f"""Valid options for silence removal are
                None, RemoveType.Split, RemoveType.Trim, RemoveType.All,
                but not '{self.remove_type}'."""
            )
        ai.data = torch.cat(sig, dim=-1)
        return ai


class Resample(Transform):
    """Resample using faster polyphase technique and avoiding FFT computation"""

    def __init__(self, sr_new):
        self.sr_new = sr_new

    def encodes(self, ai: AudioTensor) -> AudioTensor:
        if ai.sr == self.sr_new:
            return ai
        sig_np = ai.numpy()
        sr_gcd = math.gcd(ai.sr, self.sr_new)
        resampled = resample_poly(
            sig_np, int(self.sr_new / sr_gcd), int(ai.sr / sr_gcd), axis=-1
        )
        ai.data = torch.from_numpy(resampled.astype(np.float32))
        ai.sr = self.sr_new
        return ai
