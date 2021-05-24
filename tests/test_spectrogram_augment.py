import pytest

import random
import torch
from fastai.data.all import test_close as _test_close
from fastai.data.all import test_eq as _test_eq
from fastai.data.all import test_fail as _test_fail
from fastai.data.all import test_ne as _test_ne
from unittest.mock import patch

from fastaudio.all import (
    AudioConfig,
    AudioPadType,
    AudioSpectrogram,
    AudioToSpec,
    CropTime,
    DeltaGPU,
    MaskFreqGPU,
    MaskTimeGPU,
    OpenAudio,
    Pipeline,
    ResizeSignal,
    SGRoll,
    SignalShifter,
    SpectrogramTransformer,
    TfmResizeGPU,
)
from fastaudio.util import apply_transform, test_audio_tensor


def test_crop_time():
    for i in [1, 2, 5]:
        a2s = AudioToSpec.from_cfg(AudioConfig.Voice())
        audio = test_audio_tensor(seconds=3)
        crop = CropTime(i * 1000)
        inp, out = apply_transform(crop, a2s(audio))
        _test_eq(i, round(out.duration))
        _test_close(out.width, int((i / inp.duration) * inp.width), eps=1.01)


def test_crop_time_with_pipeline():
    """
    AudioToSpec->CropTime and ResizeSignal->AudioToSpec
    will result in same size images
    """
    afn = "./test.wav"
    test_audio_tensor().save(afn)
    ex_files = [afn] * 4
    oa = OpenAudio(ex_files)
    crop_dur = random.randint(1000, 5000)
    DBMelSpec = SpectrogramTransformer(mel=True, to_db=True)
    pipe_cropsig = Pipeline([oa, DBMelSpec(hop_length=128), CropTime(crop_dur)])
    pipe_cropspec = Pipeline(
        [
            oa,
            ResizeSignal(crop_dur),
            DBMelSpec(hop_length=128),
        ]
    )
    for i in range(4):
        _test_eq(pipe_cropsig(i).width, pipe_cropspec(i).width)


def test_crop_time_after_padding():
    sg_orig = test_audio_tensor()
    a2s = AudioToSpec.from_cfg(AudioConfig.Voice())
    sg = a2s(sg_orig)
    crop_time = CropTime((sg.duration + 5) * 1000, pad_mode=AudioPadType.Zeros_After)
    inp, out = apply_transform(crop_time, sg.clone())
    _test_ne(sg.duration, sg_orig.duration)


def test_crop_time_repeat_padding():
    "Test that repeat padding works when cropping time"
    repeat = 3
    audio = test_audio_tensor()
    crop_12000ms_repeat = CropTime(
        repeat * 1000 * audio.duration, pad_mode=AudioPadType.Repeat
    )
    a2s = AudioToSpec.from_cfg(AudioConfig.Voice())
    sg = a2s(audio)
    inp, out = apply_transform(crop_12000ms_repeat, sg)
    _test_eq(inp.width, sg.width)
    _test_ne(sg.width, out.width)


def test_fail_bad_pad():
    # test bad pad_mode doesnt fail silently, correct is 'zeros_after'
    _test_fail(CropTime(12000, pad_mode="zerosafter"))


def test_mask_freq():
    c, f, t = 2, 120, 80

    # Create a random frequency mask and test that it is being correctly applied
    min_size = 5
    max_size = 7

    sg = AudioSpectrogram(torch.rand([c, f, t]))
    val = 10  # Use a value not in the original spectrogram
    gradient_sg = AudioSpectrogram(
        torch.linspace(0, 1, f).view(1, f, 1).repeat([c, 1, t])
    )
    ones = torch.ones_like(sg)

    # Test patching with mean
    with patch(
        "fastaudio.augment.functional.region_mask",
        side_effect=[
            torch.BoolTensor([[1] * 10 + [0] * (f - 10)]),
        ],
    ):
        mask_with_mean = MaskFreqGPU(
            min_size=min_size, max_size=max_size, mask_val=None
        )
        # Use a gradient so we can be sure the mean will never show up outside the mask
        inp, out = apply_transform(mask_with_mean, gradient_sg)
        channelwise_mean = inp[:, :10, :].mean(dim=(-2, -1)).reshape(-1, 1, 1)
        _test_eq(out[:, :10, :], channelwise_mean * ones[:, :10, :])
        assert not (out[:, 10:, :] == channelwise_mean).any(), out == channelwise_mean

    # Test multiple masks (and patching with value)
    with patch(
        "fastaudio.augment.functional.region_mask",
        side_effect=[
            torch.BoolTensor([[1] * 10 + [0] * (f - 10), [0] * (f - 10) + [1] * 10]),
        ],
    ):
        mask_with_val = MaskFreqGPU(
            min_size=min_size, num_masks=2, max_size=max_size, mask_val=val
        )
        inp, out = apply_transform(mask_with_val, sg)
        _test_eq(
            out[:, :10, :],
            val * ones[:, :10, :],
        )
        _test_eq(
            out[:, f - 10 :, :],
            val * ones[:, f - 10 :, :],
        )
        matches = out[:, 10 : f - 10] == val
        assert not matches.any(), matches


def test_mask_time():
    c, f, t = 2, 120, 80

    min_size = 5
    max_size = 7

    sg = AudioSpectrogram(torch.rand([c, f, t]))
    val = 10  # Use a value not in the original spectrogram
    gradient_sg = AudioSpectrogram(
        torch.linspace(0, 1, t).view(1, 1, t).repeat([c, f, 1])
    )
    ones = torch.ones_like(sg)

    # Test patching with mean
    with patch(
        "fastaudio.augment.functional.region_mask",
        side_effect=[
            torch.BoolTensor([[1] * 10 + [0] * (t - 10)]),
        ],
    ):
        mask_with_mean = MaskTimeGPU(
            min_size=min_size, max_size=max_size, mask_val=None
        )
        # Use a gradient so we can be sure the mean will never show up outside the mask
        inp, out = apply_transform(mask_with_mean, gradient_sg)
        channelwise_mean = inp[..., :10].mean(dim=(-2, -1)).reshape(-1, 1, 1)
        _test_close(
            out[..., :10],
            channelwise_mean * ones[..., :10],
        )
        assert not (out[..., 10:] == channelwise_mean).any(), out == channelwise_mean

    # Test multiple masks (and patching with value)
    with patch(
        "fastaudio.augment.functional.region_mask",
        side_effect=[
            torch.BoolTensor([[1] * 10 + [0] * (t - 10), [0] * (t - 10) + [1] * 10]),
        ],
    ):
        mask_with_val = MaskTimeGPU(
            min_size=min_size, num_masks=2, max_size=max_size, mask_val=val
        )
        inp, out = apply_transform(mask_with_val, sg)
        _test_eq(
            out[..., :10],
            val * ones[..., :10],
        )
        _test_eq(
            out[..., t - 10 :],
            val * ones[..., t - 10 :],
        )
        matches = out[..., 10 : t - 10] == val
        assert not matches.any(), matches


def test_resize_int():
    # Test when size is an int
    size = 224
    resize_int = TfmResizeGPU(size)
    audio = test_audio_tensor()
    a2s = AudioToSpec.from_cfg(AudioConfig.Voice())
    sg = a2s(audio)
    inp, out = apply_transform(resize_int, sg)
    _test_eq(out.shape[1:], torch.Size([size, size]))


def test_delta_channels():
    " nchannels for a spectrogram is how many channels its original audio had "
    delta = DeltaGPU()
    # Explicitly check more than one channel
    audio = test_audio_tensor(channels=2)
    a2s = AudioToSpec.from_cfg(AudioConfig.Voice())
    sg = a2s(audio)
    inp, out = apply_transform(delta, sg)

    _test_eq(out.nchannels, inp.nchannels * 3)
    _test_eq(out.shape[-2:], inp.shape[-2:])
    for i1, i2 in [(0, 2), (1, 3), (0, 4), (1, 5), (2, 4), (3, 5)]:
        assert not torch.allclose(out[i1], out[i2])


def test_signal_shift_on_sg():
    audio = test_audio_tensor()
    a2s = AudioToSpec.from_cfg(AudioConfig.BasicSpectrogram())
    shifter = SignalShifter(1, 1)
    inp, out = apply_transform(shifter, a2s(audio))
    _test_ne(inp, out)


def test_sg_roll():
    roll = SGRoll(1)
    audio = test_audio_tensor()
    a2s = AudioToSpec.from_cfg(AudioConfig.BasicSpectrogram())
    inp, out = apply_transform(roll, a2s(audio))
    _test_ne(inp, out)


def test_sg_roll_fails_direction():
    with pytest.raises(ValueError):
        SGRoll(direction=2)
