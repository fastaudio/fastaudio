import os

from fastaudio.all import (
    AudioConfig,
    AudioToSpec,
    OpenAudio,
    Pipeline,
    audio_item_tfms,
    preprocess_audio_folder
)
from fastaudio.util import test_audio_tensor


def test_basic_config():
    "Make sure mel setting is passed down and is false for normal spectro"
    sg_cfg = AudioConfig.BasicSpectrogram()
    assert sg_cfg.mel == False


def test_load_audio_with_basic_config():
    """
    Grab a random file, test that the n_fft are passed successfully
    via config and stored in sg settings
    """
    sg_cfg = AudioConfig.BasicSpectrogram(n_fft=2000, hop_length=155)
    a2sg = AudioToSpec.from_cfg(sg_cfg)
    audio = test_audio_tensor()
    sg = a2sg(audio)
    assert sg.n_fft == sg_cfg.n_fft
    assert sg.width == int(audio.nsamples / sg_cfg.hop_length) + 1


def test_basic_pipeline():
    cfg = {"mel": False, "to_db": False, "hop_length": 128, "n_fft": 400}
    test_audio_tensor().save("./test.wav")
    f = "./test.wav"
    oa = OpenAudio([f])
    a2s = AudioToSpec.from_cfg(cfg)
    db_mel_pipe = Pipeline([oa, a2s])
    assert db_mel_pipe(0).hop_length == cfg["hop_length"]


def test_basic_pre_audio():
    tfms = audio_item_tfms(8000, True, 4000)
    assert len(tfms) == 3


def test_pre_process_audio():
    d = "data_test"
    if not os.path.isdir(d):
        os.mkdir(d)
    test_audio_tensor().save(d + "/test.wav")
    preprocess_audio_folder(d)
