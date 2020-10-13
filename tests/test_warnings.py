import pytest

from fastaudio.augment.spectrogram import (
    CropTime,
    Delta,
    MaskFreq,
    MaskTime,
    SGRoll,
    TfmResize
)
from fastaudio.util import test_audio_tensor


def invoke_class(cls, **args):
    audio = test_audio_tensor()
    with pytest.warns(Warning):
        cls(**args)(audio)


def test_show_warning_with_tfm_on_sig():
    """
    When we invoke a transform intended for a spectrogram on
    something that is most likely a signal, we show a warning.

    Check Issue #17
    """
    transforms = [CropTime, Delta, MaskFreq, MaskTime, SGRoll, TfmResize]

    for t in transforms:
        # Some of the transforms require init arguments
        if t == CropTime:
            invoke_class(t, duration=1)
        elif t == TfmResize:
            invoke_class(t, size=2)
        else:
            invoke_class(t)
