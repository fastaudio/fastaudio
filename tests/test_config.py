from fastai.data.all import untar_data

from fastaudio.all import (
    AudioConfig,
    AudioToSpec,
    OpenAudio,
    Pipeline,
    URLs,
    tar_extract_at_filename
)


def test_basic_config():
    "Make sure mel setting is passed down and is false for normal spectro"
    sg_cfg = AudioConfig.BasicSpectrogram()
    assert sg_cfg.mel == False


def test_load_audio_with_basic_config():
    """
    Grab a random file, test that the n_fft are passed successfully
    via config and stored in sg settings
    """
    p = untar_data(URLs.SPEAKERS10, extract_func=tar_extract_at_filename)
    f = p / "f0001_us_f0001_00001.wav"
    oa = OpenAudio([f])
    sg_cfg = AudioConfig.BasicSpectrogram(n_fft=2000, hop_length=155)
    a2sg = AudioToSpec.from_cfg(sg_cfg)
    sg = a2sg(oa(0))
    assert sg.n_fft == sg_cfg.n_fft
    assert sg.width == int(oa(0).nsamples / sg_cfg.hop_length) + 1


def test_basic_pipeline():
    cfg = {"mel": False, "to_db": False, "hop_length": 128, "n_fft": 400}

    p = untar_data(URLs.SPEAKERS10, extract_func=tar_extract_at_filename)
    f = p / "f0001_us_f0001_00001.wav"

    oa = OpenAudio([f])
    a2s = AudioToSpec.from_cfg(cfg)
    db_mel_pipe = Pipeline([oa, a2s])

    assert db_mel_pipe(0).hop_length == cfg["hop_length"]
