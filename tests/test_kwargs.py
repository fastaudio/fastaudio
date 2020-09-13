from fastaudio.core.spectrogram import get_usable_kwargs


def test_kwargs(a: int = 10, b: int = 20):
    pass


kwargs = {"a": 1, "b": 2}


def test_get_usable_for_function():
    assert get_usable_kwargs(test_kwargs, kwargs) == kwargs


def test_flatten_kwargs():
    extra_kwargs = {"z": 0, "a": 1, "b": 2, "c": 3}
    assert get_usable_kwargs(test_kwargs, extra_kwargs, []) == kwargs
