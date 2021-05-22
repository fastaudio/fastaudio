import random
import torch
import torchaudio
from collections import OrderedDict
from fastai.data.external import URLs
from fastai.data.transforms import Transform, get_files
from fastai.imports import Path, mimetypes, plt, tarfile
from fastai.torch_core import TensorBase, _fa_rebuild_qtensor, _fa_rebuild_tensor
from fastai.vision.data import get_grid
from fastcore.basics import patch
from fastcore.dispatch import typedispatch
from fastcore.meta import delegates
from fastcore.utils import ifnone
from IPython.display import Audio, display
from librosa.display import waveplot
from os import path

audio_extensions = tuple(
    str.lower(k) for k, v in mimetypes.types_map.items() if v.startswith("audio/")
)


def get_audio_files(path, recurse=True, folders=None):
    "Get audio files in `path` recursively, only in `folders`, if specified."
    return get_files(
        path, extensions=audio_extensions, recurse=recurse, folders=folders
    )


def AudioGetter(suf="", recurse=True, folders=None):
    """Create `get_audio_files` partial function that searches path suffix `suf`
    and passes along `kwargs`, only in `folders`, if specified."""

    def _inner(o, recurse=recurse, folders=folders):
        return get_audio_files(o / suf, recurse, folders)

    return _inner


URLs.SPEAKERS10 = "http://www.openslr.org/resources/45/ST-AEDS-20180100_1-OS.tgz"
URLs.ESC50 = "https://github.com/karoldvl/ESC-50/archive/master.zip"
URLs.SAMPLE_SPEAKERS10 = (
    "https://github.com/fastaudio/10_Speakers_Sample/archive/10_speakers_sample.zip"
)


def tar_extract_at_filename(fname, dest):
    "Extract `fname` to `dest`/`fname.name` folder using `tarfile`"
    dest = Path(dest) / Path(fname).with_suffix("").name
    tarfile.open(fname, "r:gz").extractall(dest)


# fix to preserve metadata for subclass tensor in serialization
# src: https://github.com/fastai/fastai/pull/3383
# TODO: remove this when #3383 lands and a new fastai version is created
def _rebuild_from_type(func, type, args, dict):
    ret = func(*args).as_subclass(type)
    ret.__dict__ = dict
    return ret


@patch
def __reduce_ex__(self: TensorBase, proto):
    torch.utils.hooks.warn_if_has_hooks(self)
    args = (
        type(self),
        self.storage(),
        self.storage_offset(),
        tuple(self.size()),
        self.stride(),
    )
    if self.is_quantized:
        args = args + (self.q_scale(), self.q_zero_point())
    args = args + (self.requires_grad, OrderedDict())
    f = _fa_rebuild_qtensor if self.is_quantized else _fa_rebuild_tensor
    return (_rebuild_from_type, (f, type(self), args, self.__dict__))


class AudioTensor(TensorBase):
    """
    Semantic torch tensor that represents an audio.
    Contains all of the functionality of a normal tensor,
    but additionally can be created from files and has
    extra properties. Also knows how to show itself.
    """

    @classmethod
    @delegates(torchaudio.load, keep=True)
    def create(cls, fn, cache_folder=None, **kwargs):
        "Creates audio tensor from file"
        if cache_folder is not None:
            fn = cache_folder / fn.name
        sig, sr = torchaudio.load(fn, **kwargs)
        return cls(sig, sr=sr)

    def __new__(cls, x, sr=None, **kwargs):
        return super().__new__(cls, x, sr=sr, **kwargs)

    @property
    def nsamples(self):
        return self.shape[-1]

    @property
    def nchannels(self):
        return self.shape[-2]

    @property
    def duration(self):
        return self.nsamples / float(self.sr)

    def hear(self):
        "Listen to audio clip. Creates a html player."
        display(Audio(self.cpu(), rate=self.sr))

    def show(self, ctx=None, hear=True, **kwargs):
        """Show audio clip using librosa.
        Pass `hear=True` to also display a html
        player to listen.
        """
        if hear:
            self.hear()
        return show_audio_signal(self, ctx=ctx, **kwargs)

    def apply_gain(self, gain):
        self.data *= gain
        return self

    def cutout(self, cut_pct):
        mask = torch.zeros(int(self.nsamples * cut_pct))
        mask_start = random.randint(0, self.nsamples - len(mask))
        self.data[:, mask_start : mask_start + len(mask)] = mask
        return self

    def lose_signal(self, loss_pct):
        mask = (torch.rand_like(self.data[0]) > loss_pct).float()
        self.data[..., :] *= mask
        return self

    def save(self, fn: Path, overwrite=True):
        "Save the audio into the specfied path"
        fn = path.expanduser(fn)
        if not overwrite and path.exists(fn):
            raise Exception("File already exists")
        torchaudio.save(fn, self.data, self.sr)


def show_audio_signal(ai, ctx, ax=None, title="", **kwargs):
    ax = ifnone(ax, ctx)
    if ax is None:
        _, ax = plt.subplots()
    ax.axis(False)
    for i, channel in enumerate(ai):
        # x_start, y_start, x_lenght, y_lenght, all in percent
        ia = ax.inset_axes((i / ai.nchannels, 0.2, 1 / ai.nchannels, 0.7))
        waveplot(channel.cpu().numpy(), ai.sr, ax=ia, **kwargs)
        ia.set_title(f"Channel {i}")
    ax.set_title(title)

    return ax


@typedispatch
def show_batch(
    x: AudioTensor,
    y,
    samples,
    ctxs=None,
    max_n=6,
    nrows=2,
    ncols=1,
    figsize=None,
    **kwargs,
):
    if figsize is None:
        figsize = (4 * x.nchannels, 6)

    if ctxs is None:
        ctxs = get_grid(
            min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize
        )
    ctxs = show_batch[object](
        x, y, samples, ctxs=ctxs, max_n=max_n, hear=False, **kwargs
    )
    return ctxs


class OpenAudio(Transform):
    """
    Transform that creates AudioTensors from a list of files.
    """

    def __init__(self, items):
        self.items = items

    def encodes(self, i):
        o = self.items[i]
        return AudioTensor.create(o)

    def decodes(self, i) -> Path:
        return self.items[i]
