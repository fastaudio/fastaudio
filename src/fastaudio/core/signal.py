# flake8: noqa
__all__ = [
    "audio_extensions",
    "get_audio_files",
    "AudioGetter",
    "tar_extract_at_filename",
    "AudioTensor",
    "show_audio_signal",
    "OpenAudio",
]

import torchaudio
from fastai.data.all import *
from fastai.torch_basics import *
from IPython.display import Audio, display
from librosa.display import waveplot

audio_extensions = tuple(
    str.lower(k) for k, v in mimetypes.types_map.items() if v.startswith("audio/")
)


def get_audio_files(path, recurse=True, folders=None):
    "Get audio files in `path` recursively, only in `folders`, if specified."
    return get_files(
        path, extensions=audio_extensions, recurse=recurse, folders=folders
    )


def AudioGetter(suf="", recurse=True, folders=None):
    "Create `get_audio_files` partial function that searches path suffix `suf` and passes along `kwargs`, only in `folders`, if specified."

    def _inner(o, recurse=recurse, folders=folders):
        return get_audio_files(o / suf, recurse, folders)

    return _inner


URLs.SPEAKERS10 = "http://www.openslr.org/resources/45/ST-AEDS-20180100_1-OS.tgz"
URLs.ESC50 = "https://github.com/karoldvl/ESC-50/archive/master.zip"


def tar_extract_at_filename(fname, dest):
    "Extract `fname` to `dest`/`fname`.name folder using `tarfile`"
    dest = Path(dest) / Path(fname).with_suffix("").name
    tarfile.open(fname, "r:gz").extractall(dest)


class AudioTensor(TensorBase):
    @classmethod
    @delegates(torchaudio.load, keep=True)
    def create(cls, fn, cache_folder=None, **kwargs):
        if cache_folder is not None:
            fn = cache_folder / fn.name
        sig, sr = torchaudio.load(fn, **kwargs)
        return cls(sig, sr=sr)

    @property
    def sr(self):
        return self.get_meta("sr")

    def __new__(cls, x, sr, **kwargs):
        return super().__new__(cls, x, sr=sr, **kwargs)

    # This one should probably use set_meta() but there is no documentation,
    # and I could not get it to work. Even TensorBase.set_meta?? is pointing
    # to the wrong source because of fastai patch on Tensorbase to retain types
    @sr.setter
    def sr(self, val):
        self._meta["sr"] = val

    nsamples, nchannels = add_props(lambda i, self: self.shape[-1 * (i + 1)])

    @property
    def duration(self):
        return self.nsamples / float(self.sr)

    def hear(self):
        display(Audio(self, rate=self.sr))

    def show(self, ctx=None, hear=True, **kwargs):
        "Show audio clip using `merge(self._show_args, kwargs)`"
        if hear:
            self.hear()
        show_audio_signal(self, ctx=ctx, **kwargs)
        plt.show()


def _get_f(fn):
    def _f(self, *args, **kwargs):
        cls = self.__class__
        res = getattr(super(TensorBase, self), fn)(*args, **kwargs)
        return retain_type(res, self)

    return _f


setattr(AudioTensor, "__getitem__", _get_f("__getitem__"))


def show_audio_signal(ai, ctx, **kwargs):
    if ai.nchannels > 1:
        _, axs = plt.subplots(ai.nchannels, 1, figsize=(6, 4 * ai.nchannels))
        for i, channel in enumerate(ai):
            waveplot(channel.numpy(), ai.sr, ax=axs[i], **kwargs)
    else:
        axs = plt.subplots(ai.nchannels, 1)[1] if ctx is None else ctx
        waveplot(ai.squeeze(0).numpy(), ai.sr, ax=axs, **kwargs)


class OpenAudio(Transform):
    def __init__(self, items):
        self.items = items

    def encodes(self, i):
        o = self.items[i]
        return AudioTensor.create(o)

    def decodes(self, i) -> Path:
        return self.items[i]
