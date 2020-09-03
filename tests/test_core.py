import inspect
import math

import torch
from fastai2.data.all import test_close as _close
from fastai2.data.all import test_warns as t_warns
from fastai2.data.all import untar_data
from torchaudio.transforms import MelSpectrogram

from fastaudio.all import (
    AudioTensor,
    MelSpectrogram,
    SpectrogramTransformer,
    URLs,
    tar_extract_at_filename
)


def test_load_audio():
    p = untar_data(URLs.SPEAKERS10, extract_func=tar_extract_at_filename)

    item0 = AudioTensor.create(p / "f0001_us_f0001_00001.wav")
    DBMelSpec = SpectrogramTransformer(mel=True, to_db=True)
    a2s = DBMelSpec(f_max=20000, n_mels=137)
    sg = a2s(item0)

    assert type(item0.data) == torch.Tensor
    assert item0.sr == 16000
    assert item0.nchannels == 1
    assert item0.nsamples == 74880
    assert item0.duration == 4.68

    assert sg.f_max == 20000
    assert sg.hop_length == 512
    assert sg.sr == item0.sr
    assert sg.mel
    assert sg.to_db
    assert sg.nchannels == 1
    assert sg.height == 137
    assert sg.n_mels == sg.height
    assert sg.width == 147

    defaults = {
        k: v.default for k, v in inspect.signature(MelSpectrogram).parameters.items()
    }
    hop_length = 345
    a2s = DBMelSpec(f_max=20000, hop_length=hop_length)
    sg = a2s(item0)
    assert sg.n_mels == defaults["n_mels"]
    assert sg.n_fft == 1024
    assert sg.shape[1] == sg.n_mels
    assert sg.hop_length == hop_length

    # test the spectrogram and audio have same duration, both are computed
    # on the fly as transforms can change their duration
    _close(sg.duration, item0.duration, eps=0.1)


def test_spectrograms_right_side_up():
    DBMelSpec = SpectrogramTransformer(mel=True, to_db=True)
    a2s_5hz = DBMelSpec(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        f_min=0.0,
        f_max=20000,
        pad=0,
        n_mels=137,
    )
    sine_5hz = 0.5 * torch.cos(2 * math.pi * 5 * torch.arange(0, 1.0, 1.0 / 16000))
    at_5hz = AudioTensor(sine_5hz[None], 16000)
    sg_5hz = a2s_5hz(at_5hz)
    max_row = sg_5hz.max(dim=1).indices.mode().values.item()
    assert max_row < 2


def test_add_tensors():
    tst0 = AudioTensor(torch.ones(10), sr=120)
    tst1 = AudioTensor(torch.ones(10), sr=150)
    assert (tst0 + tst1).sr == 120


def test_check_nchannels():
    audio_tensor = AudioTensor(torch.ones(1, 3, 10), sr=120)
    assert audio_tensor.nchannels == 3
