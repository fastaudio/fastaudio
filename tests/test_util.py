from fastaudio.util import create_sin_wave, test_audio_tensor


def test_create_sin_wave():
    wave = create_sin_wave()
    assert wave != None
    wave, sr = wave
    assert sr == 16000
    assert wave.shape[0] == 5 * sr


def test_shape_of_sin_wave_tensor():
    sr = 16000
    secs = 2
    ai = test_audio_tensor(secs, sr)
    assert ai.duration == secs
    assert ai.nsamples == secs * sr
