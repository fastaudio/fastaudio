import torchaudio
import warnings
from dataclasses import asdict, is_dataclass
from fastai.data.core import TensorImageBase
from fastai.imports import inspect, partial, plt
from fastai.vision.data import get_grid
from fastcore.dispatch import typedispatch
from fastcore.meta import delegates
from fastcore.transform import Transform
from fastcore.utils import ifnone
from inspect import signature
from librosa.display import specshow
from torch import nn

from .signal import AudioTensor


class AudioSpectrogram(TensorImageBase):
    """
    Semantic torch tensor that represents an Audio Spectrogram.
    Contains all of the functionality of a normal tensor,
    but has extra properties and knows how to show itself.
    """

    @classmethod
    def create(cls, sg_tensor, settings=None):
        """Create an AudioSpectrogram from a torch tensor"""
        audio_sg = cls(sg_tensor)
        audio_sg._settings = settings
        return audio_sg

    @property
    def duration(self):
        # spectrograms round up length to fill incomplete columns,
        # so we subtract 0.5 to compensate, wont be exact
        return (self.hop_length * (self.shape[-1] - 0.5)) / self.sr

    @property
    def width(self):
        return self.shape[-1]

    @property
    def height(self):
        return self.shape[-2]

    @property
    def nchannels(self):
        return self.shape[-3]

    def _all_show_args(self, show_y: bool = True):
        proper_kwargs = get_usable_kwargs(
            specshow, self._settings, exclude=["ax", "kwargs", "data"]
        )
        if "mel" not in self._settings or not show_y:
            y_axis = None
        else:
            y_axis = "mel" if self.mel else "linear"
        proper_kwargs.update({"x_axis": "time", "y_axis": y_axis})
        proper_kwargs.update(self._show_args)
        return proper_kwargs

    @property
    def _colorbar_fmt(self):
        return "%+2.0f dB" if "to_db" in self._settings and self.to_db else "%+2.0f"

    def __getattr__(self, name):
        if name == "settings":
            return self._settings
        if not name.startswith("_"):
            return self._settings[name]
        raise AttributeError(
            f"{self.__class__.__name__} object has no attribute {name}"
        )

    def show(self, ctx=None, ax=None, title="", **kwargs):
        "Show spectrogram using librosa"
        return show_spectrogram(self, ctx=ctx, ax=ax, title=title, **kwargs)


def show_spectrogram(sg, title="", ax=None, ctx=None, **kwargs):
    ax = ifnone(ax, ctx)
    if ax is None:
        _, ax = plt.subplots()
    ax.axis(False)
    for i, channel in enumerate(sg):
        # x_start, y_start, x_lenght, y_lenght, all in percent
        ia = ax.inset_axes((i / sg.nchannels, 0.2, 1 / sg.nchannels, 0.7))
        z = specshow(
            channel.cpu().numpy(), ax=ia, **sg._all_show_args(show_y=i == 0), **kwargs
        )
        ia.set_title(f"Channel {i}")
        if i == 0:  # Only colorbar the first one
            plt.colorbar(z, format=sg._colorbar_fmt, ax=ax)
    ax.set_title(title)

    return ax


@typedispatch
def show_batch(
    x: AudioSpectrogram,
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
        figsize = (4 * x.nchannels, 4)

    if ctxs is None:
        ctxs = get_grid(
            min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize
        )
    ctxs = show_batch[object](x, y, samples, ctxs=ctxs, max_n=max_n, **kwargs)
    return ctxs


_GenSpec = torchaudio.transforms.Spectrogram
_GenMelSpec = torchaudio.transforms.MelSpectrogram
_GenMFCC = torchaudio.transforms.MFCC
_ToDB = torchaudio.transforms.AmplitudeToDB


class AudioToSpec(Transform):
    """
    Transform to create spectrograms from audio tensors.
    """

    def __init__(self, pipe, settings):
        self.pipe = pipe
        self.settings = settings

    @classmethod
    def from_cfg(cls, audio_cfg):
        "Creates AudioToSpec from configuration file"
        cfg = asdict(audio_cfg) if is_dataclass(audio_cfg) else dict(audio_cfg)
        transformer = SpectrogramTransformer(mel=cfg.pop("mel"), to_db=cfg.pop("to_db"))
        return transformer(**cfg)

    def encodes(self, audio: AudioTensor):
        self.pipe.to(audio.device)
        self.settings.update({"sr": audio.sr, "nchannels": audio.nchannels})
        return AudioSpectrogram.create(self.pipe(audio), settings=dict(self.settings))


def SpectrogramTransformer(mel=True, to_db=True):
    """Creates a factory for creating AudioToSpec
    transforms with different parameters"""
    sg_type = {"mel": mel, "to_db": to_db}
    transforms = _get_transform_list(sg_type)
    pipe_noargs = partial(fill_pipeline, sg_type=sg_type, transform_list=transforms)
    pipe_noargs.__signature__ = _get_signature(transforms)
    return pipe_noargs


def _get_transform_list(sg_type):
    """Builds a list of higher-order transforms with no arguments"""
    transforms = []
    if sg_type["mel"]:
        transforms.append(_GenMelSpec)
    else:
        transforms.append(_GenSpec)
    if sg_type["to_db"]:
        transforms.append(_ToDB)
    return transforms


def fill_pipeline(transform_list, sg_type, **kwargs):
    """Adds correct args to each transform"""
    kwargs = _override_bad_defaults(dict(kwargs))
    function_list = []
    settings = {}
    for f in transform_list:
        usable_kwargs = get_usable_kwargs(f, kwargs)
        function_list.append(f(**usable_kwargs))
        settings.update(usable_kwargs)
    warn_unused(kwargs, settings)
    return AudioToSpec(nn.Sequential(*function_list), settings={**sg_type, **settings})


def _get_signature(transforms):
    """Looks at transform list and extracts all valid args for tab completion"""
    delegations = [delegates(to=f, keep=True) for f in transforms]
    out = lambda **kwargs: None  # noqa: E731
    for d in delegations:
        out = d(out)
    return signature(out)


def _override_bad_defaults(kwargs):
    if "n_fft" not in kwargs or kwargs["n_fft"] is None:
        kwargs["n_fft"] = 1024
    if "win_length" not in kwargs or kwargs["win_length"] is None:
        kwargs["win_length"] = kwargs["n_fft"]
    if "hop_length" not in kwargs or kwargs["hop_length"] is None:
        kwargs["hop_length"] = int(kwargs["win_length"] / 2)
    return kwargs


def warn_unused(all_kwargs, used_kwargs):
    unused_kwargs = set(all_kwargs.keys()) - set(used_kwargs.keys())
    for kwarg in unused_kwargs:
        warnings.warn(f"{kwarg} is not a valid arg name and was not used")


def get_usable_kwargs(func, kwargs, exclude=None):
    exclude = ifnone(exclude, [])
    defaults = {
        k: v.default
        for k, v in inspect.signature(func).parameters.items()
        if k not in exclude
    }
    usable = {k: v for k, v in kwargs.items() if k in defaults}
    return {**defaults, **usable}


@delegates(_GenMFCC.__init__)
class AudioToMFCC(Transform):
    """
    Transform to create MFCC features from audio tensors.
    """

    def __init__(self, **kwargs):
        func_args = get_usable_kwargs(_GenMFCC, kwargs, [])
        self.transformer = _GenMFCC(**func_args)
        self.settings = func_args

    @classmethod
    def from_cfg(cls, audio_cfg):
        "Creates AudioToMFCC from configuration file"
        cfg = asdict(audio_cfg) if is_dataclass(audio_cfg) else audio_cfg
        return cls(**cfg)

    def encodes(self, x: AudioTensor):
        sg_settings = {"sr": x.sr, "nchannels": x.nchannels, **self.settings}
        return AudioSpectrogram.create(
            self.transformer(x).detach(), settings=sg_settings
        )
