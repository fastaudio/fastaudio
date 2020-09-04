import pytest

from fastai.data.all import test_close as _test_close
from fastai.data.all import test_eq as _test_eq
from fastai.data.all import test_fail as _test_fail
from fastai.data.all import test_ne as _test_ne
from fastai.data.all import untar_data

from fastaudio.all import *
from fastaudio.util import apply_transform, test_audio_tensor


@pytest.fixture(scope="session")
def ex_files():
    p = untar_data(URLs.SAMPLE_SPEAKERS10)
    return (p/'train').ls()


def test_path(ex_files):
    assert len(ex_files) > 0


def test_crop_time():
    for i in [1, 2, 5]:
        a2s = AudioToSpec.from_cfg(AudioConfig.Voice())
        audio = test_audio_tensor(seconds=3)
        crop = CropTime(i*1000)
        inp, out = apply_transform(crop, a2s(audio))
        _test_eq(i, round(out.duration))

def test_crop_time_with_pipeline(ex_files):
    """
    AudioToSpec->CropTime and CropSignal->AudioToSpec
    will result in same size images
    """
    oa = OpenAudio(ex_files)
    crop_dur = random.randint(1000, 5000)
    DBMelSpec = SpectrogramTransformer(mel=True, to_db=True)
    pipe_cropsig = Pipeline(
        [oa, DBMelSpec(hop_length=128), CropTime(crop_dur)])
    pipe_cropspec = Pipeline(
        [oa, CropSignal(crop_dur), DBMelSpec(hop_length=128), ])
    for i in range(4):
        _test_eq(pipe_cropsig(i).width, pipe_cropspec(i).width)


def test_crop_time_after(ex_files):
    sg_orig = test_audio_tensor()
    a2s = AudioToSpec.from_cfg(AudioConfig.Voice())
    sg = a2s(sg_orig)
    crop_time = CropTime((sg.duration+5)*1000, pad_mode=AudioPadType.Zeros_After)
    inp, out = apply_transform(crop_time, sg.clone())
    _test_ne(sg.duration, sg_orig.duration)
