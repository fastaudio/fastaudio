import random

import pytest

import torch
from fastai.data.all import test_close as _test_close
from fastai.data.all import test_eq as _test_eq
from fastai.data.all import test_fail as _test_fail
from fastai.data.all import test_ne as _test_ne

from fastaudio.all import (
    AudioConfig,
    AudioPadType,
    AudioToSpec,
    CropTime,
    Delta,
    MaskFreq,
    MaskTime,
    OpenAudio,
    Pipeline,
    ResizeSignal,
    SGRoll,
    SignalShifter,
    SpectrogramTransformer,
    TfmResize
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
    # create a random frequency mask and test that it is being correctly applied
    size, start, val = [random.randint(1, 50) for i in range(3)]
    freq_mask_test = MaskFreq(size=size, start=start, val=val)
    sg_orig = test_audio_tensor()
    a2s = AudioToSpec.from_cfg(AudioConfig.Voice())
    sg = a2s(sg_orig)

    inp, out = apply_transform(freq_mask_test, sg)
    _test_eq(
        out[:, start : start + size, :],
        val * torch.ones_like(inp)[:, start : start + size, :],
    )


def test_mask_freq():
    # create a random time mask and test that it is being correctly applied
    size, start, val = [random.randint(1, 50) for i in range(3)]
    time_mask_test = MaskTime(size=size, start=start, val=val)
    audio = test_audio_tensor()
    a2s = AudioToSpec.from_cfg(AudioConfig.Voice())
    sg = a2s(audio)
    inp, out = apply_transform(time_mask_test, sg)
    _test_eq(
        out[:, :, start : start + size],
        val * torch.ones_like(inp)[:, :, start : start + size],
    )


def test_resize_int():
    # Test when size is an int
    size = 224
    resize_int = TfmResize(size)
    audio = test_audio_tensor()
    a2s = AudioToSpec.from_cfg(AudioConfig.Voice())
    sg = a2s(audio)
    inp, out = apply_transform(resize_int, sg)
    _test_eq(out.shape[1:], torch.Size([size, size]))


def test_delta_channels():
    " nchannels for a spectrogram is how many channels its original audio had "
    delta = Delta()
    audio = test_audio_tensor(channels=1)
    a2s = AudioToSpec.from_cfg(AudioConfig.Voice())
    sg = a2s(audio)
    inp, out = apply_transform(delta, sg)

    _test_eq(out.nchannels, inp.nchannels * 3)
    _test_eq(out.shape[1:], inp.shape[1:])
    _test_ne(out[0], out[1])


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
