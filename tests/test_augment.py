import pytest

from fastai.data.all import test_close as _test_close
from fastai.data.all import test_eq as _test_eq
from fastai.data.all import untar_data

from fastaudio.all import *


@pytest.fixture(scope="session")
def audio():
    return test_audio_tensor()


def test_path(audio):
    assert audio != None


def apply_transform(transform, inp):
    """Generate a new input, apply transform, and display/return both input and output"""
    inp_orig = inp.clone()
    out = (
        transform(inp, split_idx=0)
        if isinstance(transform, RandTransform)
        else transform(inp)
    )
    return inp_orig, out


def test_silence_removed(audio):
    "Add silence to a signal and test that it gets removed"
    silencer = RemoveSilence(threshold=20, pad_ms=20)
    orig, silenced = apply_transform(silencer, audio)
    assert silenced.nsamples <= orig.nsamples


def test_silence_not_removed(audio):
    "Test that nothing is removed from audio that doesnt contain silence"
    test_aud = AudioTensor(torch.ones_like(audio), 16000)
    orig_samples = test_aud.nsamples

    for rm_type in [RemoveType.All, RemoveType.Trim, RemoveType.Split]:
        silence_audio_trim = RemoveSilence(rm_type, threshold=20, pad_ms=20)(test_aud)
        assert orig_samples == silence_audio_trim.nsamples


def test_resample(audio):
    no_resample_needed = Resample(audio.sr)
    inp, out = apply_transform(no_resample_needed, audio)
    assert inp.sr == out.sr
    _test_eq(inp.data, out.data)


def test_resample_rates(audio):
    "Test and hear realistic sample rates"
    for rate in [2000, 4000, 8000, 22050, 44100]:
        resampler = Resample(rate)
        inp, out = apply_transform(resampler, audio)
        assert rate == out.sr
        assert out.nsamples == inp.duration * rate


def test_resample_multi_channel(audio):
    audio = test_audio_tensor(channels=3)
    resampler = Resample(8000)
    _, out = apply_transform(resampler, audio)
    _test_eq(out.nsamples, out.duration * 8000)
    _test_eq(out.nchannels, 3)
    _test_eq(out.sr, 8000)


def test_upsample(audio):
    for _ in range(10):
        random_sr = random.randint(16000, 72000)
        random_upsample = Resample(random_sr)(audio)
        num_samples = random_upsample.nsamples
        _test_close(num_samples, abs(audio.nsamples // (audio.sr / random_sr)), eps=1.1)


def test_cropping():
    "Can use the CropSignal Transform"
    audio = test_audio_tensor(seconds=10, sr=1000)

    inp, out1000 = apply_transform(CropSignal(1000), audio.clone())
    inp, out2000 = apply_transform(CropSignal(2000), audio.clone())
    inp, out5000 = apply_transform(CropSignal(5000), audio.clone())

    _test_eq(out1000.duration, 1)
    _test_eq(out2000.duration, 2)
    _test_eq(out5000.duration, 5)

    _test_eq(out1000.nsamples, out1000.duration * inp.sr)
    _test_eq(out2000.nsamples, out2000.duration * inp.sr)
    _test_eq(out5000.nsamples, out5000.duration * inp.sr)

    # Multi Channel Cropping
    inp, mc1000 = apply_transform(CropSignal(1000), audio.clone())
    inp, mc2000 = apply_transform(CropSignal(2000), audio.clone())
    inp, mc5000 = apply_transform(CropSignal(5000), audio.clone())

    _test_eq(mc1000.duration, 1)
    _test_eq(mc2000.duration, 2)
    _test_eq(mc5000.duration, 5)
