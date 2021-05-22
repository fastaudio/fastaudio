import inspect
import math
import torch
from fastai.data.all import test_close as _close
from tempfile import TemporaryFile
from torchaudio.transforms import MelSpectrogram

from fastaudio.all import AudioConfig, AudioTensor, AudioToMFCC, SpectrogramTransformer
from fastaudio.util import test_audio_tensor


# TODO
def test_download_audio():
    pass


def test_load_audio():
    item0 = test_audio_tensor()
    DBMelSpec = SpectrogramTransformer(mel=True, to_db=True)
    a2s = DBMelSpec(f_max=20000, n_mels=137)
    sg = a2s(item0)

    assert type(item0) == AudioTensor
    assert item0.sr == 16000
    assert item0.nchannels == 1
    assert item0.nsamples == 32000
    assert item0.duration == 2

    assert sg.f_max == 20000
    assert sg.hop_length == 512
    assert sg.sr == item0.sr
    assert sg.mel
    assert sg.to_db
    assert sg.nchannels == 1
    assert sg.height == 137
    assert sg.n_mels == sg.height
    assert sg.width == 63

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


def test_saved_audiotensor_keeps_metadata():
    # This test is related to this issue: https://github.com/fastaudio/fastaudio/issues/95
    # What happens is that multiprocessing uses pickling to distribute the data
    # and the way it was done inside fastai breaks when loading the metadata
    audio_tensor = test_audio_tensor()

    with TemporaryFile("wb+") as f:
        torch.save(audio_tensor, f)
        f.seek(0, 0)  # Go back to the begining of the file to read
        new_audio_tensor = torch.load(f)
        assert new_audio_tensor.sr == audio_tensor.sr


def test_saved_spectrogram_keeps_metadata():
    # Same issue as the test above
    item0 = test_audio_tensor()
    DBMelSpec = SpectrogramTransformer(mel=True, to_db=True)
    a2s = DBMelSpec(f_max=20000, n_mels=137)
    sg = a2s(item0)

    with TemporaryFile("wb+") as f:
        torch.save(sg, f)
        f.seek(0, 0)  # Go back to the begining of the file to read
        new_sg = torch.load(f)
        assert new_sg.sr == item0.sr


def test_indexing_audiotensor():
    audio_tensor = test_audio_tensor()
    assert audio_tensor.data[:, :3000].shape[1] == 3000


def test_mfcc_transform():
    audio = test_audio_tensor()
    a2s = AudioToMFCC.from_cfg(AudioConfig.BasicMFCC())
    sg = a2s(audio)
    assert len(sg.shape) == 3


def test_show_spectrogram():
    audio = test_audio_tensor()
    a2s = AudioToMFCC.from_cfg(AudioConfig.BasicMFCC())
    sg = a2s(audio)
    sg.show()
