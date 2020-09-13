import random

import pytest

import torch
from fastai.data.all import test_close as _test_close
from fastai.data.all import test_eq as _test_eq
from fastai.data.all import test_fail as _test_fail
from fastai.data.all import test_ne as _test_ne
from fastai.data.all import untar_data

from fastaudio.all import (
    AudioConfig,
    AudioPadType,
    AudioToSpec,
    CropSignal,
    CropTime,
    MaskFreq,
    MaskTime,
    OpenAudio,
    Pipeline,
    SpectrogramTransformer,
    TfmResize,
    URLs
)
from fastaudio.util import apply_transform, test_audio_tensor


@pytest.fixture(scope="session")
def ex_files():
    p = untar_data(URLs.SAMPLE_SPEAKERS10)
    return (p / "train").ls()


def test_path(ex_files):
    assert len(ex_files) > 0


def test_crop_time():
    for i in [1, 2, 5]:
        a2s = AudioToSpec.from_cfg(AudioConfig.Voice())
        audio = test_audio_tensor(seconds=3)
        crop = CropTime(i * 1000)
        inp, out = apply_transform(crop, a2s(audio))
        _test_eq(i, round(out.duration))
        _test_close(out.width, int((i / inp.duration) * inp.width), eps=1.01)


def test_crop_time_with_pipeline(ex_files):
    """
    AudioToSpec->CropTime and CropSignal->AudioToSpec
    will result in same size images
    """
    oa = OpenAudio(ex_files)
    crop_dur = random.randint(1000, 5000)
    DBMelSpec = SpectrogramTransformer(mel=True, to_db=True)
    pipe_cropsig = Pipeline([oa, DBMelSpec(hop_length=128), CropTime(crop_dur)])
    pipe_cropspec = Pipeline(
        [
            oa,
            CropSignal(crop_dur),
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
